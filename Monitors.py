"""
Author: Willian Soares Girão
Contact: wsoaresgirao@gmail.com

Description:    ...
"""

import torch, os
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import random

class SpikeMonitor():
    def __init__(self, spikes, layer):
        self.spikes = spikes
        self.conv_layer = False
        self.layer = layer

        self.time_steps = self.spikes.shape[0]
        self.batches = self.spikes.shape[1]

        if len(self.spikes) > 3:
            self.conv_layer = True

    def plot_layer(self, plot, path=None):

        root = os.path.join(path, 'layer_activity')
        if not os.path.exists(root):
            os.makedirs(root)

        root = os.path.join(root, f'{self.layer}')
        if not os.path.exists(root):
            os.makedirs(root)

        for s in range(self.batches):
            sample_s = self.spikes[:, s, :, :, :]

            plot_path = os.path.join(root, f'sample-{s}')
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)

            for c in range(sample_s.shape[1]):
                channel_c = sample_s[:, c, :, :].cpu().detach().numpy()
                name = f'layer_spk-{self.layer}_channel_{c}_sample_{s}'

                plot_path = os.path.join(plot_path, f'channel-{c}')
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)

                self.plot_layer_inner(channel_c, name, plot, plot_path)

    def plot_layer_inner(self, sample, name, save=False, path=None):
        """ . """

        x_ = sample

        x = np.arange(x_.shape[1])
        y = np.arange(x_.shape[2])
        X, Y = np.meshgrid(x, y)
        frames = range(x_.shape[0])

        def update(frame, plot):
            for p in plot:
                p.remove()
            Z = x_[frame]
            plot[0] = ax.plot_surface(X, Y, np.full_like(X, frame), facecolors=np.where(Z == 1.0, 'k', 'none'))
            plot[0] = ax.plot_surface(X, Y, np.full_like(X, frame), color='grey', alpha=0.25)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        Z = x_[0]
        plot = [ax.plot_surface(X, Y, np.full_like(X, 0), facecolors=np.where(Z == 1.0, 'k', 'none'))]

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('time step')
        plt.title(name)

        ax.set_xlim(1, x_.shape[1])
        ax.set_ylim(1, x_.shape[2])
        ax.set_zlim(0, len(frames))

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        ani = animation.FuncAnimation(fig, update, frames=frames, fargs=(plot,), interval=1)

        ax.grid(False)

        if save:
            writer = animation.PillowWriter(fps=15,
                                metadata=dict(artist='Me'),
                                bitrate=500)
            ani.save(os.path.join(path, f'layer-spikes_{name}.gif'), writer=writer)
            print(f'layer-spikes_{name}.gif')
        else:
            plt.show()

        plt.close()

class StateMonitor():
    def __init__(self, membranes, thr_voltage, layer):
        self.membranes = membranes
        self.conv_layer = False
        self.layer = layer
        self.thr_voltage = thr_voltage

        self.time_steps = self.membranes.shape[0]
        self.batches = self.membranes.shape[1]

        if len(self.membranes) > 3:
            self.conv_layer = True

    def plot_layer(self, plot, path=None):

        root = os.path.join(path, 'layer_activity')
        if not os.path.exists(root):
            os.makedirs(root)

        root = os.path.join(root, f'{self.layer}')
        if not os.path.exists(root):
            os.makedirs(root)

        for s in range(self.batches):
            sample_s = self.membranes[:, s, :, :, :]

            plot_path = os.path.join(root, f'sample-{s}')
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)

            for c in range(sample_s.shape[1]):
                channel_c = sample_s[:, c, :, :].cpu().detach().numpy()
                name = f'layer_mem-{self.layer}_channel_{c}_sample_{s}'

                plot_path = os.path.join(plot_path, f'channel-{c}')
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)

                self.plot_layer_inner(channel_c, name, plot, plot_path)

    def sample_membranes(self, path=None):

        root = os.path.join(path, 'layer_activity')
        if not os.path.exists(root):
            os.makedirs(root)

        root = os.path.join(root, f'{self.layer}')
        if not os.path.exists(root):
            os.makedirs(root)

        for s in range(self.batches):
            sample_s = self.membranes[:, s, :, :, :]

            plot_path = os.path.join(root, f'sample-{s}')
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)

            for c in range(sample_s.shape[1]):
                channel_c = sample_s[:, c, :, :].cpu().detach().numpy()
                name = f'sampled_mem-{self.layer}_channel_{c}_sample_{s}'

                plot_path = os.path.join(plot_path, f'channel-{c}')
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)

                self.plot_membranes_inner(channel_c, name, plot_path)

    def plot_membranes_inner(self, channel, name, path):

        x = random.sample(range(channel.shape[1]), channel.shape[1])
        y = random.sample(range(channel.shape[2]), channel.shape[2])

        fig, axs = plt.subplots(5, 5, figsize=(20, 10))
        axs = axs.flatten()

        plt.suptitle(name)

        for i, ax in enumerate(axs):
            ax.plot(np.arange(channel.shape[0]), channel[:, x[i], y[i]], color=np.random.rand(3,))
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.set_xlim(0, channel.shape[0])
            ax.set_ylim(0, 1)
            ax.axhline(y=self.thr_voltage, color='k', linestyle='--', lw=0.5)
            
            ax.set_title(f'({x[i]}, {y[i]})', fontsize=8)

            if i % 5 != 0:
                ax.set_yticklabels([])

            if i < (5*5)-5:
                ax.set_xticklabels([])

        plt.tight_layout()
        plt.savefig(os.path.join(path, f'sampled-membranes_{name}.png'))
        print(f'sampled-membranes_{name}.png')

    def plot_layer_inner(self, data, name, save=False, path=None):
        """ . """

        norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=self.thr_voltage, vmax=1)

        frames = range(data.shape[0])

        def update(frame, plot):
            plot[0] = ax.imshow(data[frame, :, :], cmap='seismic', norm=norm)
            plt.title(name + '\n' + r'$\Delta t$' + f' {frame}')

        fig = plt.figure()
        ax = fig.add_subplot()

        plot = [ax.imshow(data[0, :, :], cmap='seismic', norm=norm)]
        cbar = plt.colorbar(plot[0])
        cbar.ax.axhline(y=self.thr_voltage, color='k', linestyle='--')

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_xlim(0, data.shape[1]-1)
        ax.set_ylim(0, data.shape[2]-1)

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        ani = animation.FuncAnimation(fig, update, frames=frames, fargs=(plot,), interval=1)

        ax.grid(False)

        if save:
            writer = animation.PillowWriter(fps=15,
                                metadata=dict(artist='Me'),
                                bitrate=500)
            ani.save(os.path.join(path, f'membrane_{name}.gif'), writer=writer)
            print(f'layer-membranes_{name}.gif')
        else:
            plt.show()

        plt.close()