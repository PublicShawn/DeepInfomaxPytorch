import argparse

class ConfigParser(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='DeepInfomax pytorch')
        self.parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
        self.parser.add_argument('--device', default="cuda", type=str)
        self.parser.add_argument('--modelroot', default="checkpoint", type=str)
        self.parser.add_argument('--dataroot', default="data", type=str)
        self.parser.add_argument('--epochrestart', default=None, type=int)
        self.parser.add_argument('--saveepoch', default=1, type=int)
        self.parser.add_argument('--epochs', default=10, type=int)
        self.parser.add_argument('--dataset', default="cifar10", type=str)

    def get(self):
        return self.parser.parse_args()
