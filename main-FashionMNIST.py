"""
Author: Willian Soares Girão
Contact: wsoaresgirao@gmail.com

Description: trains a CSNN on the Fashion MNIST dataset (converted into spiking data).
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

from CSNN import CSNN

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

    x_train = np.array(train_dataset.data, dtype=float)         # flattening image
    x_train = x_train.reshape(x_train.shape[0], -1)/255         # normalizing values
    y_train = np.array(train_dataset.targets, dtype=int)

    x_test = np.array(test_dataset.data, dtype=float)
    x_test = x_test.reshape(x_test.shape[0], -1)/255
    y_test = np.array(test_dataset.targets, dtype=int)

    # plt.imshow(x_train[1].reshape(28, 28), cmap=plt.cm.gray_r)
    # plt.show()

    print(f'x_train shape: {x_train.shape}')
    print(f'x_test shape: {x_test.shape}')

    print(f'y_train shape: {y_train.shape}')
    print(f'y_test shape: {y_test.shape}')

    # --- 1.3. analog to spike data ---

    def current2firing_time(x, tau_mem=20.0, thr=0.7, tmax=100, epsilon=1e-7):
        """
        Computes first spiking time latency for a current input x ('current injection') fed to a LIF neuron (current-based LIF).

        Arguments:
        - tau_mem: membrane time constant of LIF neuron.
        - thr: membrane's spike threshold
        - tmax: maximum time returned (neurons that did not fire)

        Returns:
        - T: time to first spike given each "current injection" represented in x.
        """

        idx = x < thr                           # neurons that did not fire a spike
        x = np.clip(x, thr+epsilon, 1e9)        # prevents invalid computations in the subsequent logarithmic calculation
        T = tau_mem*np.log(x/(x-thr))           # time mem takes to reach thr given input 'x' and time constant 'tau' for current-based LIF neuron
        T[idx] = tmax                           # neurons that did not fire a spike take the longest to first spike

        return  np.array(T, dtype=int)

    def sparse_data_gen(X, Y, batch_size: int, device, shuffle=True, num_units=28*28, time_step=1e-3, num_steps=100, tau_mem=20e-3):
        """
        Take a dataset in analog (continuous value) format and generates spikes tensors.

        Arguments:
        - batch_size: number of sample in each batch
        - shuffle: shuffle samples during batch creation
        - num_units: number of units to which data will be fed to
        - time_step: time taken by a single simulated time step (how finely time is discretized)
        - num_steps: number of discrete time steps simulated (each representing 'time_step' units of time)
        - tau_mem: membrane time constant (continuous-time)

        Returns:
        - 
        """

        labels_ = np.array(Y, dtype=int)
        num_batches = len(X)//batch_size
        sample_index = np.arange(len(X))

        if shuffle:
            np.random.shuffle(sample_index)

        # compute discrete firing times
            
        tau_mem_adjusted = tau_mem/time_step    # scaling (continuous-time) 'tau_mem' to fit the simulation's (discrete-time) step

        firing_times = current2firing_time(X, 
                                           tau_mem=tau_mem_adjusted, 
                                           tmax=num_steps)
        units_ids = np.arange(num_units)        

        total_batch_count = 0
        counter = 0

        while counter < num_batches:                                                    # building i-th batch of spiking data

            batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]       # gathering samples belonging to the i-th batch

            batch_temp = [ [] for i in range(3) ]                                       # matrix holding analog-latency conversion results of all samples within the i-th batch

            for enumerator, sample_idx in enumerate(batch_index):

                print(enumerator, sample_idx)

                c = firing_times[sample_idx] < num_steps                                # getting units that will fire within 'num_steps' simulation steps

                times, units = firing_times[sample_idx][c], units_ids[c]                # units that will fire and their respective time-to-first-spike 

                batch_sample = [ enumerator for _ in range(len(times)) ]                # within batch sample's index (same for each unit that will spike) - to what sample in the batch each unit belongs to

                batch_temp[0].extend(batch_sample)                                      # sample index within the batch (same for each unit)
                batch_temp[1].extend(times)                                             # time-to-first-spike of each unit that has reached the threshold
                batch_temp[2].extend(units)                                             # units' IDs)

            i = torch.LongTensor(batch_temp).to(device)                                 # ?
            v = torch.FloatTensor(np.ones(len(batch_temp[0]))).to(device)               # ?

            X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, num_steps, num_units])).to(device) # ?
            Y_batch = torch.tensor(labels_[batch_index], device)                                                # ?

            yield X_batch.to(device), Y_batch.to(device)                                                        # ?

            counter += 1

    sparse_data_gen(x_train, y_train, batch_size, device)
        

    ### 1. CSNN INSTANTIATION ###
        
    num_steps = 50
    
    

    
    
if __name__ == '__main__':
    main()