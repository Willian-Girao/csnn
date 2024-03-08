"""
Author: Willian Soares Girão
Contact: wsoaresgirao@gmail.com

Description:    Defines a Dataset class (torch.utils.data.Dataset) that converts static
                images into spiking representation.
"""

import torch
import numpy as np
from torchvision import transforms
from Analog2SpikesTransforms import Normalize, Pixel2Latency, Latency2Spikes

class SpikeDataset(torch.utils.data.Dataset):
    """..."""

    def __init__(self, torchv_dataset,
                 tau_mem=20e-3,
                 thr=0.2,
                 epsilon=1e-7,
                 time_step=1e-3,
                 num_steps=100):
        """
        Arguments:
        - tau_mem: membrane time constant of LIF neuron.
        - thr: membrane's spike threshold
        - tmax: maximum time returned (neurons that did not fire)

        Returns:
        - 
        """
        self.torchv_dataset = torchv_dataset
        self.tau_mem = tau_mem
        self.thr = thr
        self.epsilon = epsilon
        self.time_step = time_step
        self.num_steps = num_steps

        if self.time_step == 0:
            raise ValueError("'time_step' cannot be zero.")
        
        if self.num_steps == 0:
            raise ValueError("'num_steps' cannot be zero.")

        self.maxval = np.max(np.array(self.torchv_dataset.data))
        self.input_dim = np.array(self.torchv_dataset.data)[0].shape

        self.transform = transforms.Compose([
            Normalize(self.maxval),
            Pixel2Latency(self.tau_mem/self.time_step,          # scaling (continuous-time) 'tau_mem' to fit the simulation's (discrete-time) step
                          self.thr,
                          self.num_steps,
                          self.epsilon),
            Latency2Spikes(self.input_dim,
                           self.time_step,
                           self.num_steps,
                           self.tau_mem)
            ])

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