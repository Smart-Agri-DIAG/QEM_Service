import torchvision.models as models
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.model = models.mobilenet_v3_small(weights=weights)
        self.model.classifier[3] = nn.Linear(in_features=1024, out_features=1)

    def forward(self, x):
        x = self.model(x)
        return x
