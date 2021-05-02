import torch
from torch import nn


class AADNorm(nn.Module):
    def __init__(self, in_channels, attr_channels, idt_channels, opt):
        """Summary
        
        Args:
            in_channels (int): number of channels of the activation input from
            previous layer
            attr_channels (TYPE): Description
            opt (TYPE): Description
        """
        super().__init()

        self.bn = nn.BatchNorm2d(in_channels)
        self.mask_conv = nn.Conv2d(
            in_channels,
            in_channels,
            opt["mask_conv"]["kernel_size"],            
            )        
        self.sigmoid = nn.Sigmoid()

        self.gamma_conv = nn.Conv2d(
            attr_channels,
            in_channels,
            opt["gamma_conv"]["kernel_size"],
            )
        self.beta_conv = nn.Conv2d(
            attr_channels,
            in_channels,
            opt["beta_conv"]["kernel_size"],
            )

        self.gamma_fc = nn.Linear(
            idt_channels,
            in_channels
            )
        self.beta_fc = nn.Linear(
            idt_channels,
            in_channels
            )


    #TODO: consider construct output of beta_fc and gamma_fc large enough to able to reshape the output to the size of the activation input, instead of duplicating values with unsqueeze and expand_as; the current choice reduces the computational requirement, but at the same time duplicates values for each channel when being added to the h_hat, and thus may reduce the expressive power of the normalization
    def forward(self, x):
        h, idt, attr = x
        h_hat = self.bn(h)
        mask = self.sigmoid(self.mask_conv(h_hat))

        attr_beta = self.beta_conv(attr)
        attr_gamma = self.gamma_conv(attr)
        denorm_attr =  h_hat * attr_gamma + attr_beta

        idt_beta = self.beta_fc(idt).unsqueeze(-1).unsqueeze(-1).expand_as(
            h_hat.size()) 
        idt_gamma = self.gamma_fc(idt).unsqueeze(-1).unsqueeze(-1).expand_as(
            h_hat.size())
        denorm_idt = h_hat * idt_gamma + idt_beta
        
        return (1-mask) * denorm_attr + mask * denorm_idt