# author: williansoaresgirao
# code source: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html

from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

from LIF import LIFlayer

### 1. DATA LOADERS ###

batch_size = 128
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

### 2. DEFINE THE NETWORK (12C5-MP2-64C5-MP2-1024FC10) ###

# neuron and simulation parameters
spike_grad = surrogate.fast_sigmoid(slope = 25)                       # surrogate gradient for spike
beta = 0.5
num_steps = 50

class CSNN(nn.Module):
    def __init__(self):
        super().__init__()

        # initializing layers
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = LIFlayer()     
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = LIFlayer()     
        self.fc1 = nn.Linear(64*4*4, 10)
        self.lif3 = LIFlayer(output=True)

    def reset_states(self):
        self.lif1.reset_mem()
        self.lif2.reset_mem()
        self.lif3.reset_mem()

    def forward(self, x):

        cur1 = F.max_pool2d(self.conv1(x), 2)
        spk1 = self.lif1(cur1)

        cur2 = F.max_pool2d(self.conv2(spk1), 2)
        spk2 = self.lif2(cur2)

        cur3 = self.fc1(spk2.view(batch_size, -1))
        spk3, mem3 = self.lif3(cur3)

        return spk3, mem3

net = CSNN()

### 3. SINGLE BATCH FORWARD PASS ###

data, targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)

def forward_pass(net, num_steps, data):
  mem_rec = []
  spk_rec = []
  net.reset_states()  # resets hidden states for all LIF neurons in net

  for step in range(num_steps):
      spk_out, mem_out = net(data)
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)

  return torch.stack(spk_rec), torch.stack(mem_rec)

spk_rec, mem_rec = forward_pass(net, num_steps, data)

loss_fn = SF.ce_rate_loss()             # cross entropy loss to the output spike count in order train a rate-coded network
loss_val = loss_fn(spk_rec, targets)    # target neuron index as the second argument to generate a loss

print(f"> The loss from an untrained network is {loss_val.item():.3f}")

acc = SF.accuracy_rate(spk_rec, targets)    # predicted output spikes and actual targets are supplied as arguments

print(f"> The accuracy of a single batch using an untrained network is {acc*100:.3f}%")

### 3.1. TEST ACCURACY ###

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

test_acc = batch_accuracy(test_loader, net, num_steps)

print(f"> The total accuracy on the test set is: {test_acc * 100:.2f}%")

### 4. TRAINING LOOP ###

optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
num_epochs = 1
loss_hist = []
test_acc_hist = []
counter = 0

# outer training loop
for epoch in range(num_epochs):

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
                print(f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
                test_acc_hist.append(test_acc.item())

        counter += 1

fig = plt.figure(facecolor="w")
plt.plot(test_acc_hist)
plt.title("Test Set Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()