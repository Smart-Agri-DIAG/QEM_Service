import torch
from torch import nn
from scripts.learning_modules.models.brixcolor.uresnet import UResNet



class BrixColorNet(nn.Module):
    """
    """

    def __init__(self, num_classes=6, num_channels=3, pretrained=False, resnet_version='resnet18'):
        super(BrixColorNet, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.pretrained = pretrained
        self.auto_encoder = UResNet(output_channels=3, restnet_version=resnet_version)
        if resnet_version == 'resnet18' or resnet_version == 'resnet34':
            self.neck = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            )
        else:
            self.neck = nn.Sequential(
                nn.Conv2d(4096, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.Conv2d(1024, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
        self.regressor = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, x):
        features, sc = self.auto_encoder.encode(x)
        ae_out = self.auto_encoder.decode(features, sc)
        bc_features = self.neck(features) # common brix anc color features
        return ae_out, self.classifier(bc_features), self.regressor(bc_features)
