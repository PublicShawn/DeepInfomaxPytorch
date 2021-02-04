import torch
from model.models import Classifier


class LatentDIM(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = Classifier()

    def forward(self, x):
        z, features = self.encoder(x)
        z = z.detach()
        # to be sure there is hash
        z = torch.sign(z).float()
        return self.classifier((z, features))