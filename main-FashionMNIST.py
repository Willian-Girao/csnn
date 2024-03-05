# author: williansoaresgirao
# code source: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html


from snntorch import functional as SF

import torch
from torch.utils.data import DataLoader

from CSNN import CSNN

def main():

    ### 1. DATA LOADERS ###

    batch_size = 128

    train_filename = "dvs-cifar10/train/"
    test_filename = "dvs-cifar10/test/"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    train_loader = DataLoader(DVSCifar10(train_filename),
                            batch_size=batch_size,
                            shuffle=False)

    test_loader = DataLoader(DVSCifar10(test_filename),
                            batch_size=batch_size,
                            shuffle=False)
    
    data, targets = next(iter(train_loader))
    data = data.to(device)
    targets = targets.to(device)

    print(data.shape)
    
    ### 2. INSTANTIATE CSNN ###

    csnn = CSNN(batch_size = batch_size)
    
if __name__ == '__main__':
    main()