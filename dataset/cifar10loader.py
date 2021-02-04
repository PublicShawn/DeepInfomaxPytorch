import glob
import re
import os.path as osp
from torch.utils.data import DataLoader
import torch
import torchvision

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

        datalen = len(dt)
        trainlen = int(datalen*(1-split_coef))
        testlen = int(datalen-trainlen)
        train, test = torch.utils.data.random_split(dt, (trainlen, testlen))
        self.train = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test = DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=True)
        self.num_classes = 10
        self.train_sz = trainlen // batch_size * batch_size
        self.test_sz = testlen // batch_size * batch_size


