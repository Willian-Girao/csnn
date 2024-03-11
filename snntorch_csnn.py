# author: williansoaresgirao
# code source: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools

### 1. SURROGATE GRADIENT DESCENT ###

# @TODO - the class defined bellow is not used later in the code. Try using it instead of the 'surrogate.fast_sigmoid(slope=25)'.

# Leaky neuron model, overriding the backward pass with a custom function
class LeakySigmoidSurrogate(nn.Module):
    def __init__(self, beta=0.5, threshold=1.0, k=25):

        self.beta = beta
        self.threshold = threshold
        self.surrogate_func = self.FastSigmoid.apply

    # the forward function is called each time we call Leaky
    def forward(self, input_, mem):
        spk = self.surrogate_func((mem-self.threshold))             # call the Heaviside function
        reset = (spk - self.threshold).detach()                     # reset will be '0' if the neuron has spiked
        mem = self.beta * mem + input_ - reset

        return spk, mem

    # Forward pass: Heaviside function
    # Backward pass: override Dirac Delta with gradient of fast sigmoid (surrogate gradient)
    """
        Nested classes in Python do not inherently have access to the variables of their containing class (unless explicitly passed), so 
    marking it as a static method doesn't make sense. The decorator is intended to be used before methods within a class.
    """
    #@staticmethod                                                  # commeted out since it's wrong to use this decorator for a nested class
    class FastSigmoid(torch.autograd.Function):
        @staticmethod                                               # conceptual method (doesn't access/modify class/instance variables)
        def forward(ctx, mem, k=25):
            ctx.save_for_backward(mem)                              # store the membrane potential for use in the backward pass
            ctx.k = k

            # @TODO - what is out here? for it to be the spike it should use the threshold. This is not clear to me.
            out = (mem > 0).float()                                 # Heaviside on the forward pass: Eq(1)

            return out
        
        @staticmethod
        def backward(ctx, grad_output):
            (mem,) = ctx.saved_tensors                              # retrieve membrane potential
            grad_input = grad_output.clone()
            grad = grad_input / (ctx.k * torch.abs(mem) + 1.0)**2   # (surrogate) gradient of fast sigmoid on backward pass: Eq(4)

            return grad, None

### 2. SETTING UP THE CSNN ###

### 2.1 DATA LOADERS ###

batch_size = 64
data_path = '/datasets/mnist'

dtype = torch.float
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

### 2.2 DEFINE THE NETWORK (12C5-MP2-64C5-MP2-1024FC10) ###

# neuron and simulation parameters
spike_grad = surrogate.fast_sigmoid(slope=25)                       # surrogate gradient for spike
beta = 0.5
num_steps = 50

# network definition
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # initializing layers
        self.conv1 = nn.Conv2d(1, 12, 5)                                              # 5x5 conv. kernel with 12 filters and 1 input channel
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)     # LIF neuron layer 1
        self.conv2 = nn.Conv2d(12, 64, 5)                                             # 5x5 conv. kernel with 64 filters and 12 input channels
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)     # LIF neuron layer 2
        self.fc1 = nn.Linear(64*4*4, 10)                                              # @TODO - why '64*4*4'?
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)     # LIF neuron layer 3 (output)

    def forward(self, x):

        # initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        cur1 = F.max_pool2d(self.conv1(x), 2)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool2d(self.conv2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc1(spk2.view(batch_size, -1))
        spk3, mem3 = self.lif3(cur3, mem3)

        return spk3, mem3
    
net = nn.Sequential(nn.Conv2d(1, 12, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(12, 64, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(64*4*4, 10),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)

### 2.3 FORWARD PASS ###

data, targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)

def forward_pass(net, num_steps, data):
  mem_rec = []
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(num_steps):
      spk_out, mem_out = net(data)
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)

  return torch.stack(spk_rec), torch.stack(mem_rec)

spk_rec, mem_rec = forward_pass(net, num_steps, data)

### 3. TRAINING LOOP ###

loss_fn = SF.ce_rate_loss()             # cross entropy loss to the output spike count in order train a rate-coded network

def batch_accuracy(train_loader, net, num_steps):
  '''
  Returns the accuracy on the entire DataLoader object.
  '''
  with torch.no_grad():
    total = 0
    acc = 0
    net.eval()

    train_loader = iter(train_loader)
    for data, targets in train_loader:
      data = data.to(device)
      targets = targets.to(device)
      spk_rec, _ = forward_pass(net, num_steps, data)

      acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
      total += spk_rec.size(1)

  return acc/total

optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
num_epochs = 1
loss_hist = []
test_acc_hist = []
dataset_percentage = []
counter = 0

# outer training loop
for epoch in range(num_epochs):

    print(f'>> epoch #: {epoch}')

    # training loop
    for data, targets in iter(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train()
        spk_rec, _ = forward_pass(net, num_steps, data)

        # initialize the loss & sum over time
        loss_val = loss_fn(spk_rec, targets)

        # gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # store loss history for future plotting
        loss_hist.append(loss_val.item())

        # test set
        if counter % 50 == 0:
            with torch.no_grad():
                net.eval()

                # Test set forward pass
                test_acc = batch_accuracy(test_loader, net, num_steps)
                print(f"training set percentage: {(((counter+1)*batch_size)/len(mnist_train))*100}, test accuracy: {test_acc * 100:.2f}%\n")
                test_acc_hist.append(test_acc.item())
                dataset_percentage.append((((counter+1)*batch_size)/len(mnist_train))*100)

        counter += 1

### 4. PLOT TEST ACCURACY ###

fig = plt.figure(facecolor="w")
plt.plot(test_acc_hist, dataset_percentage)
plt.title("test set accuracy (static mnist)")
plt.xlabel("training set percentage")
plt.ylabel("accuracy")
plt.show()