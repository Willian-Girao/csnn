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
    # @TODO - flattened here should be removed since first layer will be a convolutional layer.

    x_train = np.array(train_dataset.data, dtype=float)         # flattening image
    x_train = x_train.reshape(x_train.shape[0], -1)/255         # normalizing values
    y_train = np.array(train_dataset.targets, dtype=int)

    x_test = np.array(test_dataset.data, dtype=float)
    x_test = x_test.reshape(x_test.shape[0], -1)/255
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

            coo_tensor = [ [] for i in range(3) ]                                       # matrix holding analog-latency conversion results of all samples within the i-th batch

            for enumerator, sample_idx in enumerate(batch_index):

                c = firing_times[sample_idx] < num_steps                                # getting units that will fire within 'num_steps' simulation steps

                times, units = firing_times[sample_idx][c], units_ids[c]                # units that will fire and their respective time-to-first-spike 

                batch_sample = [ enumerator for _ in range(len(times)) ]                # within batch sample's index (same for each unit that will spike) - to what sample in the batch each unit belongs to

                coo_tensor[0].extend(batch_sample)                                      # sample index within the batch (same for each unit)
                coo_tensor[1].extend(times)                                             # time-to-first-spike of each unit that has reached the threshold
                coo_tensor[2].extend(units)                                             # units' IDs)

            """
                In the following lines of code are creatig sparse tensors for improved memory usage (source: https://pytorch.org/docs/stable/sparse.html, 
            section "Construction"). They will effectively transform the latency encoding above into spiking data.

            The tensor 'X_batch' will have three dimensions (a - # of sample in the batch, b - # of discrete time steps, c - # of units). This means that, for each of
            the 'num_steps' discrete time steps, a unit 'c' in sample 'a' will have a '1' at X_batch[a][b][c] if at time_step 'b' it was above the thr value (it will be
            '0' otherwise).

            The data becomes "spiking" data because the 2nd argument of 'torch.sparse.FloatTensor' sets the entries (values) for 'X_batch' that are not supposed to be
            the default (zero) ones.

            The 3rd argument of 'torch.sparse.FloatTensor' defines the dimensions of the resulting tensor (3D in this case, as explined above). This means that each
            sample (1st dim.) is represented in a span of 'num_steps' (2nd dim.) discrete time by 'num_units' (3rd dim.) spiking units.
            """

            i = torch.LongTensor(coo_tensor).to(device)                                 # location for entries that do not have default value (zero by default)
            v = torch.FloatTensor(np.ones(len(coo_tensor[0]))).to(device)               # entries for locations without default value (setting to one)

            X_batch = torch.sparse.FloatTensor(i, 
                                               v, 
                                               torch.Size([batch_size, num_steps, num_units])
                                               ).to(device)
            
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
            #         plt.imshow(np.array(tstep).reshape(28, 28), cmap=plt.cm.gray_r)
            #         plt.axis('off')
            #         plt.savefig(f'animation/tstep_{countt}.png')
            #         plt.close()
            #         countt += 1

            yield X_batch.to(device), Y_batch.to(device)                                                        # ?

            counter += 1

    sparse_data_gen(x_train, y_train, batch_size, device)
        

    ### 1. CSNN INSTANTIATION ###
        
    num_steps = 50
    
    

    
    
if __name__ == '__main__':
    main()