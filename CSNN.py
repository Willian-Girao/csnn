"""
Author: Willian Soares Girão
Contact: wsoaresgirao@gmail.com

Description:    Class implementing a convolutional spiking neural network (nn.Module) with LIF neurons.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from LIF import LIFlayer

class CSNN(nn.Module):
    def __init__(self, batch_size=128):
        super().__init__()

        # training parameters
        self.batch_size = batch_size

        # initializing layers
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = LIFlayer()
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = LIFlayer()
        self.fc1 = nn.Linear(64*4*4, 10)
        self.lif3 = LIFlayer(output=True)

    def reset_states(self):
        """
        Access all LIF layers and reset membrane state tensors.
        """
        self.lif1.reset_mem()
        self.lif2.reset_mem()
        self.lif3.reset_mem()

    def forward(self, x):
        """
        Forward pass on input tensor(s).
        """

        cur1 = F.max_pool2d(self.conv1(x), 2)
        spk1 = self.lif1(cur1)

        cur2 = F.max_pool2d(self.conv2(spk1), 2)
        spk2 = self.lif2(cur2)

        cur3 = self.fc1(spk2.view(self.batch_size, -1))     # @TODO '.view(self.batch_size, -1)'
        spk3, mem3 = self.lif3(cur3)

        return spk3, mem3
    
    @staticmethod
    def forward_pass(net, num_steps, data):
        mem_rec = []
        spk_rec = []
        net.reset_states()                                 # resets hidden states for all LIF neurons in net

        for step in range(num_steps):
            spk_out, mem_out = net.forward(data)
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec), torch.stack(mem_rec)