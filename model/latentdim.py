import torch
from model.models import Classifier


class LatentDIM(torch.nn.Module):
    def __init__(self, encoder, hashbit):
        super().__init__()
        self.encoder = encoder
        self.classifier = Classifier(hashbit)

    def forward(self, x):
        z, features = self.encoder(x)
        z = z.detach()
        return self.classifier((z, features))