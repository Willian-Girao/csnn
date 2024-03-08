"""
Author: Willian Soares Girão
Contact: wsoaresgirao@gmail.com

Description:    Defines transformations to be applied to an input image in order
                to convert it into a spike data representation (latency encoding
                of pixel values).
"""

import torch
import numpy as np

class Normalize(object):
    """ Normalizes pixel values between 0. and 1. """

    def __init__(self, maxp_val: float, dtype=float):
        """
        - maxp_val: maximum pixel value on the dataset.
        """
        self.maxp_val = maxp_val
        self.dtype = dtype

    def __call__(self, sample):
        x_ = np.array(sample[0], dtype=self.dtype)/self.maxp_val
        y_ = np.array(sample[1], dtype=self.dtype)

        return x_, y_
    
class Pixel2Latency(object):
    """..."""

    def __init__(self, tau_mem, thr, tmax, epsilon):
        """
        - tau_mem: membrane time constant of LIF neuron.
        - thr: membrane's spike threshold
        - tmax: maximum time returned (neurons that did not fire)
        """
        self.tau_mem = tau_mem
        self.thr = thr
        self.tmax = tmax
        self.epsilon = epsilon

    def __call__(self, sample):
        x_, y_ = sample

        idx = x_ < self.thr                              # neurons that did not fire a spike
        x_ = np.clip(x_, self.thr+self.epsilon, 1e9)     # prevents invalid computations in the subsequent logarithmic calculation
        T = self.tau_mem*np.log(x_/(x_-self.thr))        # time mem takes to reach thr given input 'x' and time constant 'tau' for current-based LIF neuron
        T[idx] = self.tmax                               # neurons that did not fire a spike take the longest to first spike

        return np.array(T, dtype=int), y_
    
class Latency2Spikes(object):
    """..."""

    def __init__(self, input_dim, time_step, num_steps, tau_mem):
        self.input_dim = input_dim
        self.time_step = time_step
        self.num_steps = num_steps
        self.tau_mem = tau_mem
        
        self.units_ids = np.indices(self.input_dim)

    def __call__(self, sample):
        x_, y_ = sample

        mask_ = x_ < self.num_steps         # getting units that will fire within 'num_steps' simulation steps

        spks_tsteps, x_coor, y_coor = x_[mask_], self.units_ids[0][mask_], self.units_ids[1][mask_]

        coo_ = [ [] for i in range(3) ]

        coo_[0] = spks_tsteps               # tstep dimension indexing
        coo_[1] = x_coor                    # image's grid x-coordinate indexing
        coo_[2] = y_coor                    # image's grid y-coordinate indexing

        indices = torch.tensor(coo_, dtype=torch.int64)                             # [[batch samples], [total time steps], [input gridcell x axis], [input gridcell y axis]]
        values = torch.tensor(np.ones(len(coo_[0])), dtype=torch.float32)           # spike as 1.0 at time step t (otherwise 0.0 default value)
        
        x_ = torch.sparse_coo_tensor(indices, values, (self.num_steps, self.input_dim[0], self.input_dim[1]))
        y_ = torch.from_numpy(np.array(y_))

        """
            Batches of sparse tensors are not currently supported by the default collate_fn. Have to 
        convert the sparse tensors to dense (defeating the purpose of using them :( ).
        source: https://github.com/pytorch/pytorch/issues/106837
        """
        return x_.to_dense(), y_.to_dense()