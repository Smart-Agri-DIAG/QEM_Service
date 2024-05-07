import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models


class UpConv(nn.Module):

    def __init__(self, in_ch, out_ch, scale=2):
        super(UpConv, self).__init__()
        neck_ch = in_ch // 4
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=scale),
            nn.Conv2d(in_channels=in_ch, out_channels=neck_ch,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(neck_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=neck_ch, out_channels=neck_ch,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(neck_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=neck_ch, out_channels=out_ch,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.up_conv(x)
        return x
    

class BottleneckConv(nn.Module):
    
    def __init__(self, in_ch, out_ch, stride=1):
        super(BottleneckConv, self).__init__()
        neck_ch = in_ch // 4
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=neck_ch,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(neck_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=neck_ch, out_channels=neck_ch,
                      kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(neck_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=neck_ch, out_channels=out_ch,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x


class UResNet(nn.Module):
    """
    TODO
    """

    def __init__(self, output_channels=1, pretrained=True, restnet_version='resnet50'):
        """
        TODO: Rendere modulare il decoder
        :param use_gpu:
        """
        super(UResNet, self).__init__()
        weights = None
        if pretrained:
            weights = 'DEFAULT'
        # ==================================================================
        # ENCODER
        # ==================================================================
        # select resnet version
        if restnet_version == 'resnet18':
            base_model = models.resnet18()
        elif restnet_version == 'resnet34':
            base_model = models.resnet34(weights=weights)
        elif restnet_version == 'resnet50':
            base_model = models.resnet50(weights=weights)
        elif restnet_version == 'resnet101':
            base_model = models.resnet101(weights=weights)
        elif restnet_version == 'resnet152':
            base_model = models.resnet152(weights=weights)
        else:
            raise ValueError("Invalid resnet version")
        # extract layers from resnet
        self.initial_conv = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu
        )
        self.pool = base_model.maxpool

        self.encoder_list = nn.ModuleList()
        for layer in [base_model.layer1, base_model.layer2, base_model.layer3, base_model.layer4]:
            self.encoder_list.append(layer)
        if restnet_version == 'resnet18' or restnet_version == 'resnet34':
            self.bottleneck = BottleneckConv(512, 1024, stride=2)
        else:
            self.bottleneck = BottleneckConv(2048, 4096, stride=2)
            
        # ==================================================================
        # DECODER
        # ==================================================================
        self.decoder_list = nn.ModuleList()
        # decoder layers
        if restnet_version == 'resnet18' or restnet_version == 'resnet34':
            features = [512, 256, 128, 64]
        else:
            features = [2048, 1024, 512, 256]
        # add upsampling and bottleneck layers corresponding to the four encoder layers
        for i in range(len(features)):
            self.decoder_list.append(UpConv(features[i]*2, features[i]))
            self.decoder_list.append(BottleneckConv(features[i] * 2, features[i]))
        # add upsampling and bottleneck layers corresponding to the last encoder layer
        if restnet_version == 'resnet18' or restnet_version == 'resnet34':
            self.decoder_list.append(UpConv(64, 32))
            self.decoder_list.append(BottleneckConv(96, 64))
        else:
            self.decoder_list.append(UpConv(256, 64))
            self.decoder_list.append(BottleneckConv(128, 64))
        
        # final upsampling and convolutions
        self.final_conv =  nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, output_channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, sc = self.encode(x)
        x = self.decode(x, sc)
        return x

    def encode(self, x):
        skip_connections = list()
        # encoding
        x = self.initial_conv(x)
        skip_connections.append(x)
        x = self.pool(x)

        for layer in self.encoder_list:
            x = layer(x)
            skip_connections.append(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        return x, skip_connections

    def decode(self, x, skip_connections):
        # decoding
        for i in range(0, len(self.decoder_list), 2):
            x = self.decoder_list[i](x)
            skip_connection = skip_connections[i//2]
            if skip_connection.shape != x.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            x = torch.cat([skip_connection, x], dim=1)
            x = self.decoder_list[i+1](x)

        x = self.final_conv(x)

        return x
