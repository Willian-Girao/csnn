"""
Author: Willian Soares Girão
Contact: wsoaresgirao@gmail.com

Description:    ...
"""

import torch
import numpy as np

class Normalize(object):
    """ Normalizes pixel values between 0. and 1. """

    def __init__(self, maxp_val: float, dtype=float):
        """
        - maxp_val: maximum pixel value on the dataset.
        """
        self.maxp_val = maxp_val
        self.dtype = dtype

    def __call__(self, sample):
        x_ = np.array(sample[0], dtype=self.dtype)/self.maxp_val
        y_ = np.array(sample[1], dtype=self.dtype)

        return x_, y_
    
class Pixel2Spiketime(object):
    """..."""

    def __init__(self, tau_mem=20.0, thr=0.2, tmax=100, epsilon=1e-7):
        """
        - tau_mem: membrane time constant of LIF neuron.
        - thr: membrane's spike threshold
        - tmax: maximum time returned (neurons that did not fire)
        """
        self.tau_mem = tau_mem
        self.thr = thr
        self.tmax = tmax
        self.epsilon = epsilon

    def __call__(self, sample):
        x_, y_ = sample

        idx = x_ < self.thr                              # neurons that did not fire a spike
        x_ = np.clip(x_, self.thr+self.epsilon, 1e9)     # prevents invalid computations in the subsequent logarithmic calculation
        T = self.tau_mem*np.log(x_/(x_-self.thr))        # time mem takes to reach thr given input 'x' and time constant 'tau' for current-based LIF neuron
        T[idx] = self.tmax                               # neurons that did not fire a spike take the longest to first spike

        return np.array(T, dtype=int), y_

class SpikeDataset(torch.utils.data.Dataset):
    """..."""

    def __init__(self, torchv_dataset, transform):
        """
        Arguments:
        - transform (callable): transformations to be applied on a sample:
            - Normalize: normalizes pixel values between 0. and 1. (assumes a gray scale img)

        Returns:
        - 
        """
        self.torchv_dataset = torchv_dataset
        self.transform = transform
        self.maxval = np.max(np.array(self.torchv_dataset.data))

    def __len__(self):
        return len(np.array(self.torchv_dataset.data))
    
    def __getitem__(self, idx):
        """
        Returns single sample from the dataset.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_ = self.torchv_dataset.data[idx]
        y_ = self.torchv_dataset.targets[idx]

        sample = self.transform((x_, y_))

        return sample