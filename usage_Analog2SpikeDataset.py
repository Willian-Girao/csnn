"""
Author: Willian Soares Girão
Contact: wsoaresgirao@gmail.com

Description:    Script showcases how to transform an image torchvision.dataset into
                an equivalent spiking representation.
"""

import numpy as np
import utils
from torchvision import datasets
from Analog2SpikeDataset import SpikeDataset
from torch.utils.data import DataLoader

root = 'datasets'

train_dataset = datasets.MNIST(root, train=True, transform=None, target_transform=None, download=True)

spikedataset = SpikeDataset(train_dataset)

first_sample = spikedataset[1]
x_ = first_sample[0]
y_ = first_sample[1]

x_ = x_.to_dense()
x_ = np.array(x_.tolist())

y_ = y_.to_dense()
y_ = np.array(y_.tolist())

print(x_.shape)
print(f'x_: {x_.shape}')
print(f'y_: {y_} ({y_.shape})')

utils.plot_spiking_img(x_)

train_loader = DataLoader(spikedataset, batch_size=32, shuffle=True, drop_last=True)

data_loader = iter(train_loader)
for data, targets in data_loader:
    break

print('> done')