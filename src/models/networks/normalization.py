import torch
from torch import nn


class AADNorm(nn.Module):
    def __init__(self, opt):
        super().__init()
        self.bn = nn.BatchNorm2d(opt["in_size"][1])
        self.mask_conv = nn.Conv2d(
            opt["in_size"][1],
            opt["in_size"][1],
            opt["mask_conv"]["kernel_size"],            
            )        
        self.sigmoid = nn.Sigmoid()

        self.gamma_conv = nn.Conv2d(
            opt["attr_size"][1],
            opt["in_size"][1],
            opt["gamma_conv"]["kernel_size"],
            )
        self.beta_conv = nn.Conv2d(
            opt["attr_size"][1],
            opt["in_size"][1],
            opt["beta_conv"]["kernel_size"],
            )

        self.gamma_fc = nn.Linear(
            opt["idt_size"],
            opt["in_size"]
            )
        self.beta_fc = nn.Linear(
            opt["idt_size"],
            opt["in_size"]
            )

    def forward(self, x):
        h, idt, attr = x
        h_hat = self.bn(h)
        mask = self.sigmoid(self.mask_conv(h_hat))

        attr_beta = self.beta_conv(attr)
        attr_gamma = self.gamma_conv(attr)
        denorm_attr =  h_hat * attr_gamma + attr_beta

        idt_beta = self.beta_fc(idt)
        idt_gamma = self.gamma_fc(idt)
        denorm_idt = h_hat * idt_gamma + idt_beta
        
        return (1-mask) * denorm_attr + mask * denorm_idt