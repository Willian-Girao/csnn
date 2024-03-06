"""
Author: Willian Soares Girão
Contact: wsoaresgirao@gmail.com

Description: trains a CSNN on the Fashion MNIST dataset (converted into spiking data).
"""
import numpy as np
import matplotlib.pyplot as plt

from snntorch import functional as SF

import torch
import torchvision

from CSNN import CSNN
from FMNIST2spikes import sparse_data_gen

def main():

    if torch.cuda.is_available():               # check GPU availability
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    ### 1. DATA LOADING ###
        
    # --- 1.1. loading Fashion MNIST dataset ---
        
    batch_size = 64
    root = 'datasets'
    
    train_dataset = torchvision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None, download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root, train=False, transform=None, target_transform=None, download=True)

    # --- 1.2. standardizing dataset ---
    # @TODO - flattened here should be removed since first layer will be a convolutional layer.

    x_train = np.array(train_dataset.data, dtype=float)/255             # normalizing pixels
    y_train = np.array(train_dataset.targets, dtype=float)

    x_test = np.array(test_dataset.data, dtype=float)/255
    y_test = np.array(test_dataset.targets, dtype=float)

    print(f'x_train shape: {x_train.shape}')
    print(f'x_test shape: {x_test.shape}')

    print(f'y_train shape: {y_train.shape}')
    print(f'y_test shape: {y_test.shape}')

    # --- 1.3. analog to spike data ---
        

    ### 1. CSNN INSTANTIATION ###

    num_steps = 50

    net = CSNN(batch_size = batch_size)

    loss_fn = SF.ce_rate_loss()             # cross entropy loss to the output spike count in order train a rate-coded network

    ### 2. TRAINING LOOP ###

    def batch_accuracy(x_, y_, batch_size, device, num_steps = num_steps):
        '''
        Returns the accuracy on the entire DataLoader object.
        '''
        with torch.no_grad():
            total = 0
            acc = 0
            net.eval()

            for x_local, y_local in sparse_data_gen(x_, y_, batch_size, device, num_steps = num_steps):
                data = x_local.to_dense().to(device)
                targets = y_local.to_dense().to(device)
                spk_rec, _ = CSNN.forward_pass_spikes(net, data, num_steps)

                acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
                total += spk_rec.size(1)

        return acc/total

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
    num_epochs = 1
    loss_hist = []
    test_acc_hist = []
    counter = 0

    # outer training loop
    for epoch in range(num_epochs):

        print(f'>> epoch #: {epoch}')

        batch_idx = 0

        # training loop
        for x_local, y_local in sparse_data_gen(x_train, y_train, batch_size, device, num_steps = num_steps):
            data = x_local.to_dense().to(device)
            targets = y_local.to_dense().to(device)

            # forward pass
            net.train()
            spk_rec, _ = CSNN.forward_pass_spikes(net, data, num_steps)

            # print(f'       batch: [{batch_idx}]')

            batch_idx += 1

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
                    test_acc = batch_accuracy(x_test, y_test, batch_size, device, num_steps = num_steps)
                    print(f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
                    test_acc_hist.append(test_acc.item())

            counter += 1

    fig = plt.figure(facecolor="w")
    plt.plot(test_acc_hist)
    plt.title("Test Set Accuracy")
    plt.xlabel("batch number")
    plt.ylabel("Accuracy")
    plt.show()
    
if __name__ == '__main__':
    main()