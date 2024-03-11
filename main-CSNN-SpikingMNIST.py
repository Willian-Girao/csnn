"""
Author: Willian Soares Girão
Contact: wsoaresgirao@gmail.com

Description: trains a CSNN on the MNIST dataset (converted into spiking data).
"""
import numpy as np
import matplotlib.pyplot as plt

from snntorch import functional as SF

import torch
import torchvision

from Analog2SpikeDataset import SpikeDataset
from torch.utils.data import DataLoader

from CSNN import CSNN


def main():

    if torch.cuda.is_available():               # check GPU availability
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    ### 1. DATA LOADING ###
        
    # --- 1.1. loading MNIST dataset ---
        
    batch_size = 64
    num_steps = 50
    root = 'datasets'
    
    train_dataset = torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=True)
    test_dataset = torchvision.datasets.MNIST(root, train=False, transform=None, target_transform=None, download=True)

    # --- 1.2. converting static images into spike data ---

    train_spks = SpikeDataset(train_dataset, num_steps=num_steps)
    test_spks = SpikeDataset(test_dataset, num_steps=num_steps)

    train_loader = DataLoader(train_spks, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_spks, batch_size=batch_size, shuffle=True, drop_last=True)        

    ### 2. CSNN INSTANTIATION ###

    net = CSNN(batch_size=batch_size, spk_threshold=1.0)

    loss_fn = SF.ce_rate_loss()             # cross entropy loss to the output spike count in order train a rate-coded network

    ### 3. TRAINING LOOP ###

    def batch_accuracy(data_loader, device, num_steps):
        '''
        Returns the accuracy on the entire DataLoader object.
        '''
        with torch.no_grad():
            total = 0
            acc = 0
            net.eval()

            data_loader = iter(data_loader)
            for data, targets in data_loader:
                data = data.to(device)
                targets = targets.to(device)
                spk_rec, _ = CSNN.forward_pass_spikes(net, data, num_steps)

                acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
                total += spk_rec.size(1)

        return acc/total

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
    num_epochs = 1
    loss_hist = []
    test_acc_hist = []
    dataset_percentage = []

    # outer training loop
    for epoch in range(num_epochs):

        counter = 0

        print(f'>> epoch #: {epoch}')

        for data, targets in iter(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            net.train()
            spk_rec, _ = CSNN.forward_pass_spikes(net, data, num_steps)

            # initialize the loss & sum over time
            loss_val = loss_fn(spk_rec, targets.long())

            # gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # store loss history for future plotting
            loss_hist.append(loss_val.item())

            # test set
            if counter % 50 == 0:
                with torch.no_grad():
                    net.eval()

                    # Test set forward pass
                    test_acc = batch_accuracy(test_loader, device, num_steps)
                    print(f"training set percentage: {np.round((((counter+1)*batch_size)/len(train_dataset))*100, 1)}%, test accuracy: {test_acc * 100:.2f}%\n")
                    test_acc_hist.append(test_acc.item())
                    dataset_percentage.append(np.round((((counter+1)*batch_size)/len(train_dataset))*100, 1))

            counter += 1

    fig = plt.figure(facecolor="w")
    plt.plot(test_acc_hist, dataset_percentage)
    plt.ylim(0, 1)
    plt.title("test set accuracy (static mnist)")
    plt.xlabel("training set percentage")
    plt.ylabel("accuracy")
    plt.show()
    
if __name__ == '__main__':
    main()