import glob
import re
import os.path as osp
from torch.utils.data import DataLoader
import torch
import torchvision

defaultTransform = torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.5,), (0.5,))
                                       ])

class Cifar10Loader(object):
    def __init__(self, root_dir, split_coef=0.2):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        loader = DataLoader(
            torchvision.datasets.CIFAR10(root_dir, train=True, download=True,
                                       transform=defaultTransform),
            batch_size=16, shuffle=False)

        datalen = len(loader.dataset)
        trainlen = int(datalen*(1-split_coef))
        testlen = int(datalen-trainlen)
        train, test = torch.utils.data.random_split(loader.dataset, (trainlen, testlen))
        self.loader = loader
        self.train = train
        self.test = test
        self.num_classes = 10


