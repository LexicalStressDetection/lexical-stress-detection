import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder


class CNNDataset(Dataset):
    def __init__(self, root):
        self.dataset_folder = DatasetFolder(root=root, loader=CNNDataset._npy_loader, extensions=('_mfcc.npy',))
        self.len_ = len(self.dataset_folder)

    @staticmethod
    def _npy_loader(path):
        mfcc = np.load(path)
        non_mfcc_file_path = path.replace('mfcc', 'other')
        non_mfcc = np.load(non_mfcc_file_path)

        # in_channels x height x width
        assert mfcc.shape == (3, 1, 27)
        assert non_mfcc.shape == (18, )

        mfcc = torch.from_numpy(mfcc).float()
        non_mfcc = torch.from_numpy(non_mfcc).float()

        return mfcc, non_mfcc

    def __getitem__(self, index):
        return self.dataset_folder[index]

    def __len__(self):
        return self.len_
