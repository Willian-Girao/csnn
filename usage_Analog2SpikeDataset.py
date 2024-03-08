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

spikedataset = SpikeDataset(train_dataset)

sample = spikedataset[1]
x_ = sample[0]
y_ = sample[1]

x_ = np.array(x_.tolist())
y_ = np.array(y_.tolist())

print(x_.shape)
print(f'x_: {x_.shape}')
print(f'y_: {y_} ({y_.shape})')

spikedataset.plot_sample(idx = 1)

train_loader = DataLoader(spikedataset, batch_size=32, shuffle=True, drop_last=True)

data_loader = iter(train_loader)
for data, targets in data_loader:
    break

print('> done')