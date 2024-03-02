# Class implementing a (discrete time) LIF neuron.

# resources to read:
# - https://discuss.pytorch.org/t/can-i-specify-backward-function-in-my-custom-layer-by-inheriting-nn-module/81231
# - https://pytorch.org/docs/stable/notes/extending.html

import numpy as np
import torch
import torch.nn as nn

class LIF(nn.Module):
    def __init__(self, size_in, size_out, tau_mem=10e-3, threshold=1.0, t_step=1e-3):
        super().__init__()

        self.size_in, self.size_out, self.t_step = size_in, size_out, t_step

        self.tau_mem = tau_mem
        self.threshold = threshold

        self.beta = np.float(np.exp(-self.time_step/self.tau_mem))      # membrane decay

    @staticmethod
    def spike_fn(mem, threshold):
        """
        In discrete-time the spiking non-linearity can be formulated as a Heaviside step function.
        """
        out = torch.zeros_like(mem)
        out[mem > threshold] = 1.0

        return out
    
    def forward(self, x, mem):
        """
        Forward pass.

        Arguments:
        - x: input to neuron (interpreted as a current injection)
        - mem: membrane potential value

        Returns:
        - spk: spikes emmited
        - mem: updated membrane potential value
        """
        spk = LIF.spike_fn(mem, self.threshold)
        rst = spk.detach()                          # no backprop through the membrane reset

        mem = (self.beta*mem + x)*(1.0 - rst)       # update mem if not in reset

        return spk, mem
    
    class SurrogateFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mem):
            ctx.save_for_backward(mem)