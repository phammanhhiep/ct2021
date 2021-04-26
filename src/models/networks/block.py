import torch
from torch import nn

from models.networks.normalization import AADeNorm


class AADResBlk(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        num_adds = opt["num_adds"]
        self.model = []

        for n in range(num_adds):
            self.model.append(nn.sequential(
                AADeNorm(opt["AADeNorm"][n]),
                nn.ReLU(),
                nn.Conv2d(
                    opt["conv"]["in_channels"][n],
                    opt["conv"]["out_channels"][n],
                    opt["conv"]["kernel_size"][n]
                    )
                ))

        if opt["mismatch_in_out_channels"]:
            self.input_transform_sq = nn.sequential(
                AADeNorm(opt["input_transform"]["AADeNorm"]),
                nn.ReLU(),
                nn.Conv2d(
                    opt["input_transform"]["conv"]["in_channels"],
                    opt["input_transform"]["conv"]["out_channels"],
                    opt["input_transform"]["conv"]["kernel_size"]
                    )                
                )


    def forward(self, x):
        """The method also takes into account when input and output channels are 
        different. 
        
        Args:
            x (TYPE): (activation output, identity, attribute) 
        
        Returns:
            TYPE: Description
        """
        h, idt, attr = x
        for sq in self.model:
            h_out = sq(x)
            x = (h_out, idt, attr)

        if self.opt["mismatch_in_out_channels"]: 
            h = self.input_transform_sq((h, idt, attr))
        return h + h_out