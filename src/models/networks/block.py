import torch
from torch import nn

from models.networks.normalization import AADNorm


class AADResBlk(nn.Module):
    def __init__(self, in_channels, attr_channels, idt_channels, out_channels, 
        opt):
        """Summary
        
        Args:
            in_channels (TYPE): Description
            attr_channels (TYPE): Description
            idt_channels (TYPE): Description
            out_channels (TYPE): Description
            opt (TYPE): Description
        """
        super().__init__()
        self.model = []
        self.diff_in_out_channels = out_channels != in_channels
        conv_kernel_size = opt["conv"]["kernel_size"]
        conv_stride = opt["conv"]["stride"]
        conv_padding = opt["conv"]["padding"]

        self.model.append(nn.sequential([
            AADNorm(in_channels, attr_channels, idt_channels, opt["AADNorm"]),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, conv_kernel_size, conv_stride,
                conv_padding)
            ]))
        self.model.append(nn.sequential([
            AADNorm(in_channels, attr_channels, idt_channels, opt["AADNorm"]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels, out_channels, conv_kernel_size, conv_stride,
                conv_padding)
            ]))

        if self.diff_in_out_channels:
            self.input_transform = nn.sequential([
                AADNorm(in_channels, attr_channels, idt_channels, 
                    opt["AADNorm"]),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, conv_kernel_size, 
                    conv_stride, conv_padding)
            ])


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