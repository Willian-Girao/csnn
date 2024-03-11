"""
Author: Willian Soares Girão
Contact: wsoaresgirao@gmail.com

Description: class implementing a (discrete time) LIF neuron.
"""

# resources to read:
# - https://discuss.pytorch.org/t/can-i-specify-backward-function-in-my-custom-layer-by-inheriting-nn-module/81231
# - https://pytorch.org/docs/stable/notes/extending.html

import numpy as np
import torch
import torch.nn as nn

from SurrogateGradient import SpkSurrogateGradFunction

class LIFlayer(nn.Module):
    def __init__(self, tau_mem=10e-3, threshold=1.0, t_step=6e-3, output=False):
        super().__init__()

        self.t_step = t_step

        self.tau_mem = tau_mem
        self.threshold = threshold

        self.beta = float(np.exp(-self.t_step/self.tau_mem))      # membrane decay

        self.spike_fn = SpkSurrogateGradFunction.apply            # surrogate gradient class implementing spk non-lin. (forward) and surr. grad. (backward)

        self.output = output                                      # if 'True' returns (spk, mem), else returns only spk
        
        self.forwarded = False                                    # flag to reset mem state between forward pases

    def reset_mem(self):
        """
        Sets a flag to re-initialize self.mem tensor before a new batch forward pass.
        """
        self.forwarded = False
    
    def forward(self, x):
        """
            Forward pass (LIF neuron layer computation): reads out membrane value to check if spike 
        is sent, followed by update of membrane value (or reset in case a spike was emitted).

        Arguments:
        - x: input to neuron (interpreted as a current injection)
        - mem: membrane potential value

        Returns:
        - spk: spikes emmited
        - mem: updated membrane potential value
        """

        if not hasattr(self, 'mem') or not self.forwarded:                            # triggered only on the first pass to initialize membrane tensor
            self.mem = torch.zeros_like(x, requires_grad=True)
            self.forwarded = True

        spk = self.spike_fn(self.mem, self.threshold)
        rst = spk.detach()                                      # no backprop through the membrane reset

        self.mem = (self.beta*self.mem + x)*(1.0 - rst)         # update mem if not in reset

        if self.output:
            return spk, self.mem
        else:
            return spk