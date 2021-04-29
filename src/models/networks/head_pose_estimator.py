import math


import torch
from torch import nn
import torchvision


class Hopenet(nn.Module):
    #TODO: make sense what is block.expansion and why it determine the input features
    def __init__(self):
        """The net is similar to the implementation of ResNet50 from Pytorch. 
        The differences from that of Pytorch include,
        + There are 3 FC layer at the end
        + The last avgpool is AvgPool2d, instead of AdaptiveAvgPool2d   
        
        Args:
            block (TYPE, optional): Description
            layers (list, optional): Description
            num_bins (int, optional): Description
        """
        super().__init__()
        block = torchvision.models.resnet.Bottleneck
        layers = [3, 4, 6, 3]
        num_bins = 66

        self.model = torchvision.models.resnet50()
        self.model.avgpool = nn.AvgPool2d(7)
        self.model.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.model.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.model.fc_roll = nn.Linear(512 * block.expansion, num_bins)


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw = self.model.fc_yaw(x)
        pre_pitch = self.model.fc_pitch(x)
        pre_roll = self.model.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll