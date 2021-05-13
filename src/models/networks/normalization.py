import torch
from torch import nn


class AADNorm(nn.Module):
    def __init__(self, in_channels, attr_channels, idt_channels):
        """
        Every convolution layers are expected to not change spatial dimention of 
        the corresponding input. 
        
        Args:
            in_channels (int): number of channels of the activation input from
            previous layer
            attr_channels (TYPE): Description
            idt_channels (TYPE): Description
            
        """
        super().__init__()
        conv_kernel_size = 3
        conv_stride = 1
        conv_padding = 1

        self.bn = nn.BatchNorm2d(in_channels)
        self.mask_conv = nn.Conv2d(
            in_channels, in_channels, conv_kernel_size, conv_stride,
            conv_padding)        
        self.sigmoid = nn.Sigmoid()

        self.gamma_conv = nn.Conv2d(
            attr_channels, in_channels, conv_kernel_size, conv_stride,
            conv_padding
            )
        self.beta_conv = nn.Conv2d(
            attr_channels, in_channels, conv_kernel_size, conv_stride,
            conv_padding
            )

        self.gamma_fc = nn.Linear(idt_channels, in_channels)
        self.beta_fc = nn.Linear(idt_channels, in_channels)


    #TODO: the current implementation save memory and computation cost by duplicate output of beta_fc and gamma_fc; consider to remove the restriction.
    def forward(self, x):
        """Spatial dimensions of each data point in attr and h are assumed to be
        the same.
        Note that after expand_as, output of beta_fc and gamma_fc are transformed
        so that the final outputs having the same values in individual channels. 
        Args:
            x (TYPE): (bath of previous activations, batch of identity, 
            batch of attributes). 
        
        Returns:
            tensor: same size as input
        """
        h, idt, attr = x
        idt = idt.permute(0, 2, 3, 1)

        h_hat = self.bn(h)
        mask = self.sigmoid(self.mask_conv(h_hat))

        attr_beta = self.beta_conv(attr)
        attr_gamma = self.gamma_conv(attr)
        denorm_attr =  h_hat * attr_gamma + attr_beta

        idt_beta = self.beta_fc(idt).permute(0,3,1,2).expand_as(h) 
        idt_gamma = self.gamma_fc(idt).permute(0,3,1,2).expand_as(h)
        denorm_idt = h_hat * idt_gamma + idt_beta
        
        return (1-mask) * denorm_attr + mask * denorm_idt