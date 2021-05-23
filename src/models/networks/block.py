import torch
from torch import nn

from src.models.networks.normalization import AADNorm


class AADResBlk(nn.Module):
    def __init__(self, in_channels, attr_channels, idt_channels, out_channels):
        """The block takes into account when input and output channels 
        are different. The output, previous activation (input), and attributes 
        are supposed to have the same spatial dimensions.
        
        Args:
            in_channels (TYPE): Description
            attr_channels (TYPE): Description
            idt_channels (TYPE): Description
            out_channels (TYPE): Description
        """
        super().__init__()
        self.diff_in_out_channels = out_channels != in_channels
        conv_kernel_size = 3
        conv_stride = 1
        conv_padding = 1

        self.sq1 = nn.Sequential(
            AADNorm(in_channels, attr_channels, idt_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, conv_kernel_size, conv_stride,
                conv_padding)
            )
        self.sq2 = nn.Sequential(
            AADNorm(in_channels, attr_channels, idt_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, conv_kernel_size, conv_stride,
                conv_padding)
            )

        if self.diff_in_out_channels:
            self.input_transform = nn.Sequential(
                AADNorm(in_channels, attr_channels, idt_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, conv_kernel_size, 
                    conv_stride, conv_padding)
            )


    def forward(self, x):
        """
        Args:
            x (TYPE): (activation output, identity, attribute) 
        
        Returns:
            TYPE: Description
        """
        h, idt, attr = x
        h_out = self.sq1(x)
        h_out = self.sq2((h_out, idt, attr))

        if self.diff_in_out_channels: 
            h = self.input_transform((h, idt, attr))
        return h + h_out