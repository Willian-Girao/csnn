"""
Author: Willian Soares Girão
Contact: wsoaresgirao@gmail.com

Description:    Class implementing a surrogate gradient function for the forward pass binary non-linearity 
                (i.e., partial derivative of a function which to some extent approximates the stepfunction for
                gradient computation).
"""

import torch
import numpy as np

class SpkSurrogateGradFunction(torch.autograd.Function):
    """
        Implements the binary non-linearity during __foward__ pass and the surrogate gradient function for
    the __backward__ pass (normalized negative part of fast sigmoid [Zenke & Ganguli (2018)], computing
    the partial derivate of the spikes wrt mem).
    """

    @staticmethod
    def forward(ctx, mem, threshold, k):
        """
        Spike non-linearity (Heaviside funtion of membrane voltage).

        Arguments:
        - context: object used to stash information needed to later backpropagate error signals
        - mem: membrane potential value (used for computing gradient of the spk wrt mem)

        Returns:
        - out: neurons that have spiked in the time-step
        """
        ctx.save_for_backward(mem)
        ctx.k = np.float64(k)                             # saving for access in the backward pass (controls steepness of surrogate gradient function)
        out = torch.zeros_like(mem)
        out[mem > threshold] = np.float64(1.0)

        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
            Computes the surrogate gradient of the spike non-linearity for gradient estimation (partial
        derivate of spike wrt mem).

        The backward method of a custom torch.autograd.Function subclass should return gradients for each 
        input to the forward method.

        Arguments:
        - grad_output: gradient of loss wrt spike (output of forward)

        Returns:
        - grad: gradient of loss wrt to mem
        - None: gradient of loss wrt to threshold
        - None: gradient of loss wrt to k
        """

        (mem,) = ctx.saved_tensors
        grad_input = grad_output.clone()                   # upstream gradient (gradient of the loss wrt the output of the 'forward' method (i.e. gradient of loss wrt spike))
        grad = grad_input/(ctx.k*torch.abs(mem) + 1.0)**2  # upstream gradient x local gradient
        
        return grad, None, None