import torch
from torch import nn

from models.networks.normalization import AADNorm


class AADResBlk(nn.Module):
    def __init__(self, in_size, attr_size, idt_size, out_channels, opt):
        super().__init__()
        self.opt = opt
        in_channels = in_size[1]
        self.model = []
        self.diff_in_out_channels = out_channel != in_channels

        self.model.append(nn.sequential([
            AADNorm(in_size, attr_size, idt_size, opt["AADNorm"]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels,
                in_channels,
                opt["conv"]["kernel_size"]
                )
            ]))
        self.model.append(nn.sequential([
            AADNorm(in_size, attr_size, idt_size, opt["AADNorm"]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels,
                out_channels,
                opt["conv"]["kernel_size"]
                )
            ]))

        if self.diff_in_out_channels:
            self.input_transform = nn.sequential(
                AADNorm(in_size, attr_size, idt_size, opt["AADNorm"]),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels,
                    out_channel,
                    opt["conv"]["kernel_size"]
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

        if self.diff_in_out_channels: 
            h = self.input_transform((h, idt, attr))
        return h + h_out