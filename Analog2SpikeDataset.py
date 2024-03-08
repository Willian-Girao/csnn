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

import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        """ Returns single sample from the dataset. """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_ = self.torchv_dataset.data[idx]
        y_ = self.torchv_dataset.targets[idx]

        sample = self.transform((x_, y_))

        return sample
    
    def plot_sample(self, idx, save=False):
        """ Plots a single sample from the dataset. """
        sample = self.__getitem__(idx)

        x_ = np.array(sample[0].tolist())
        y_ = np.array(sample[1].tolist())

        x = np.arange(x_.shape[1])
        y = np.arange(x_.shape[2])
        X, Y = np.meshgrid(x, y)
        frames = range(x_.shape[0])

        def update(frame, plot):
            for p in plot:
                p.remove()
            Z = x_[frame]
            plt.title(f'class {y_}')
            plot[0] = ax.plot_surface(X, Y, np.full_like(X, frame), facecolors=np.where(Z == 1.0, 'k', 'none'))
            plot[0] = ax.plot_surface(X, Y, np.full_like(X, frame), color='grey', alpha=0.25)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        Z = x_[0]
        plot = [ax.plot_surface(X, Y, np.full_like(X, 0), facecolors=np.where(Z == 1.0, 'k', 'none'))]

        ax.set_xlabel('x-coor')
        ax.set_ylabel('y-coor')
        ax.set_zlabel('time step')

        ax.set_xlim(1, 28)
        ax.set_ylim(1, 28)
        ax.set_zlim(0, len(frames))

        ax.set_xticks(np.arange(1, 29, 1))
        ax.set_yticks(np.arange(1, 29, 1))

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        ani = animation.FuncAnimation(fig, update, frames=frames, fargs=(plot,), interval=1)

        ax.grid(False)

        if save:
            writer = animation.PillowWriter(fps=15,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
            ani.save(f'animation/spiking_digit_{y_}_sample.gif', writer=writer)
        else:
            plt.show()