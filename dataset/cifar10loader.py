import glob
import re
import os.path as osp
from torch.utils.data import DataLoader
import torch
import torchvision
import numpy as np
from sklearn.model_selection import train_test_split

defaultTransform = torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor()
                                       ])

class Cifar10Loader(object):
    def __init__(self, root_dir, batch_size, split_coef=0.2):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        dt = torchvision.datasets.CIFAR10(root_dir, download=True, transform=defaultTransform)
        self.dt = dt

        self.num_classes = 10
        self.train_sz = 500*self.num_classes // batch_size * batch_size
        self.test_sz = 1000*self.num_classes // batch_size * batch_size

        targets = np.array(self.dt.targets)
        target_indices = np.arange(len(targets))
        train_idx, val_idx = train_test_split(target_indices, train_size=500*self.num_classes, stratify=targets)
        query_idx, ret_idx = train_test_split(val_idx, train_size=1000*self.num_classes, stratify=targets[val_idx])

        train = torch.utils.data.Subset(self.dt, train_idx)
        query = torch.utils.data.Subset(self.dt, query_idx)
        retrieve = torch.utils.data.Subset(self.dt, ret_idx)

        self.train = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
        self.query = DataLoader(query, batch_size=batch_size, shuffle=True, drop_last=True)
        self.retrieve = DataLoader(retrieve, batch_size=batch_size, shuffle=True, drop_last=True)




