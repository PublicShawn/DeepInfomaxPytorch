from dataset.cifar10loader import Cifar10Loader
class DataLoadingManager(object):
    def __init__(self, datasetname, root, split_coef):
        self.root = root
        self.split_coef = split_coef
        self.datasetname = datasetname
        self.mapping = {
            "cifar10": Cifar10Loader
        }

    def __call__(self):
        return self.mapping[self.datasetname](self.root, self.split_coef)