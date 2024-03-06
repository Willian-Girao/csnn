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
        
    batch_size = 1
    root = 'datasets'
    
    train_dataset = torchvision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None, download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root, train=False, transform=None, target_transform=None, download=True)

    # --- 1.2. standardizing dataset ---

    x_train = np.array(train_dataset.data, dtype=float)/255
    y_train = np.array(train_dataset.targets, dtype=int)

    x_test = np.array(test_dataset.data, dtype=float)/255
    y_test = np.array(test_dataset.targets, dtype=int)

    print(f'x_train shape: {x_train.shape}')
    print(f'x_test shape: {x_test.shape}')

    print(f'y_train shape: {y_train.shape}')
    print(f'y_test shape: {y_test.shape}')

    # --- 1.3. analog to spike data ---

    def current2firing_time(x, tau_mem=20.0, thr=0.2, tmax=100, epsilon=1e-7):
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

    def sparse_data_gen(X, Y, batch_size: int, device, shuffle=False, num_units=(28, 28), time_step=1e-3, num_steps=100, tau_mem=20e-3):
        """
        Take a dataset in analog (continuous value) format and generates spikes tensors.

        Arguments:
        - batch_size: number of sample in each batch
        - shuffle: shuffle samples during batch creation
        - num_units: dimensions of image
        - time_step: time taken by a single simulated time step (how finely time is discretized)
        - num_steps: number of discrete time steps simulated (each representing 'time_step' units of time)
        - tau_mem: membrane time constant (continuous-time)

        Returns:
        - X_batch: three dimensional tensor (# of sample in the batch, # of discrete time steps, # of units) containing spike representation of images dataset
        - Y_batch: the labels
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
        
        units_ids = np.indices(num_units)

        counter = 0

        while counter < num_batches:                                                    # building i-th batch of spiking data

            batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]       # gathering samples belonging to the i-th batch

            coo_ = [ [] for i in range(4) ]                                             # matrix holding analog-latency conversion results of all samples within the i-th batch

            for enumerator, sample_idx in enumerate(batch_index):

                c = firing_times[sample_idx] < num_steps                                # getting units that will fire within 'num_steps' simulation steps

                spks_tsteps, x_coor, y_coor = firing_times[sample_idx][c], units_ids[0][c], units_ids[1][c]

                batch_sample = [ enumerator for _ in range(len(spks_tsteps)) ]          # within batch sample's index (same for each unit that will spike) - to what sample in the batch each unit belongs to

                coo_[0].extend(batch_sample)                                            # sample dimension indexing
                coo_[1].extend(spks_tsteps)                                             # tstep dimension indexing
                coo_[2].extend(x_coor)                                                  # image's grid x-coordinate indexing
                coo_[3].extend(y_coor)                                                  # image's grid y-coordinate indexing

            indices = torch.tensor(coo_, dtype=torch.int64)
            values = torch.tensor(np.ones(len(coo_[0])), dtype=torch.float32)
            
            X_batch = torch.sparse_coo_tensor(indices, values, (batch_size, num_steps, num_units[0], num_units[0])).to(device=device)
            Y_batch = torch.tensor(labels_[batch_index], device=device)

            # Uncomment lines bellow to export the "spiking image" for the samples.
            # w = X_batch.to_dense()
            # wtolist = w.tolist()
            # counts = 0
            # for sample in wtolist:
            #     counts += 1
            #     countt = 0
            #     for tstep in sample:
            #         print(f'sample # {counts}, tstep: {countt}')
            #         plt.title(f'sample # {counts}, tstep: {countt}')
            #         plt.imshow(np.array(tstep), cmap=plt.cm.gray_r)
            #         plt.axis('off')
            #         plt.savefig(f'animation/tstep_{countt}.png')
            #         plt.close()
            #         countt += 1

            # input('stopped (v2)...')

            yield X_batch.to(device), Y_batch.to(device)

            counter += 1

    sparse_data_gen(x_train, y_train, batch_size, device)
    
if __name__ == '__main__':
    main()