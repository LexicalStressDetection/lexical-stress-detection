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
        f1_f7_file_path = path.replace('mfcc', 'f1_f7')
        f1_f7 = np.load(f1_f7_file_path)

        # in_channels x height x width
        assert mfcc.shape == (3, 1, 27)
        assert f1_f7.shape == (21, )

        mfcc = torch.from_numpy(mfcc).float()
        f1_f7 = torch.from_numpy(f1_f7).float()

        return mfcc, f1_f7

    def __getitem__(self, index):
        return self.dataset_folder[index]

    def __len__(self):
        return self.len_
