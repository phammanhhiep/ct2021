import math


import torch
from torch import nn
import torchvision
import torch.nn.functional as F


from src.common import utils 


class HopeNet(nn.Module):
    #TODO: make sense what is block.expansion and why it determines the input features
    def __init__(self):
        """The net is similar to the implementation of ResNet50 from Pytorch. 
        The differences from that of Pytorch include,
        + There are 3 FC layer at the end
        + The last avgpool is AvgPool2d, instead of AdaptiveAvgPool2d   
        
        The net returns how likely the angles fall in the num_bin bins (though
        they are not probabilities). See "[2017] Fine-Grained Head Pose 
        Estimation Without Keypoints (Ruiz et al)" for details about the
        architecture.
        """
        super().__init__()
        block = torchvision.models.resnet.Bottleneck
        layers = [3, 4, 6, 3]
        num_bins = 66
        self.idx_tensor = torch.tensor(range(num_bins))

        self.model = torchvision.models.resnet50()
        del self.model.fc
        self.model.avgpool = nn.AvgPool2d(7)
        self.model.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.model.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.model.fc_roll = nn.Linear(512 * block.expansion, num_bins)


    def forward(self, x):
        """Summary
        
        Args:
            x (TYPE): a batch of images of size (B, C, H, W) 
        
        Returns:
            TYPE: a list of tensor of size (B, num_bins)
        """
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


    @torch.no_grad()
    def to_degree(self, x):
        """The formula follow the training procedure of the related research. 
        
        Args:
            x (TYPE): output of the network [pre_yaw, pre_pitch, pre_roll]
        """
        d = []
        batch_size = x[0].size()[0]
        for xi in x:
            p = F.softmax(xi, dim=1)
            di = torch.sum(p * self.idx_tensor, dim=1) * 3 - 99
            d.append(di)
        return torch.cat(d).reshape((len(x), batch_size))


    def load(self, label, save_dir):
        utils.load_net(self.model, label, save_dir)
