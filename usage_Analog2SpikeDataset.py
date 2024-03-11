"""
Author: Willian Soares Girão
Contact: wsoaresgirao@gmail.com

Description:    Script showcases how to transform an image torchvision.dataset into
                an equivalent spiking representation.
"""

import numpy as np
from torchvision import datasets
from Analog2SpikeDataset import SpikeDataset
from torch.utils.data import DataLoader

root = 'datasets'

train_dataset = datasets.MNIST(root, train=True, transform=None, target_transform=None, download=True)

spikedataset = SpikeDataset(train_dataset, num_steps=50)

sample = spikedataset[1]
x_ = sample[0]
y_ = sample[1]

x_ = np.array(x_.tolist())
y_ = np.array(y_.tolist())

print(f'x_: {x_.shape}')
print(f'y_: {y_} (shape {y_.shape})')

spikedataset.plot_sample(idx=1, save=True)

train_loader = DataLoader(spikedataset, batch_size=32, shuffle=True, drop_last=True)

data_loader = iter(train_loader)
for data, targets in data_loader:
    break

print('> done')