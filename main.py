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

from LIF import LIFlayer

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

    def forward(self, x):

        cur1 = F.max_pool2d(self.conv1(x), 2)
        spk1 = self.lif1(cur1)

        cur2 = F.max_pool2d(self.conv2(spk1), 2)
        spk2 = self.lif2(cur2)

        cur3 = self.fc1(spk2.view(batch_size, -1))
        spk3, mem3 = self.lif3(cur3)

        return spk3, mem3

net = CSNN()

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
loss_val = loss_fn(spk_rec, targets)    # target neuron index as the second argument to generate a loss

print(f"The loss from an untrained network is {loss_val.item():.3f}")