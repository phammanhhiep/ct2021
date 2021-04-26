import torch
from torch import nn


class AADeNorm(nn.Module):
    def __init__(self, opt):
        super().__init()
        bn = nn.BatchNorm2d(opt["last_activation_size"])
        gamma_conv = nn.Conv2d(
            opt["last_target_attr_num_channels"],
            opt["AADeNorm"]["gamma_out_channels"],
            opt["AADeNorm"]["gamma_kernel_size"],
            )
        beta_conv = nn.Conv2d(
            opt["last_attr_num_channels"],
            opt["AADeNorm"]["gamma_out_channels"],
            opt["AADeNorm"]["gamma_kernel_size"],
            )
        gamma_fc = nn.Linear(
            opt["identity_num_features"],
            opt["AADeNorm"]["fc_num_activations"]
            )
        beta_fc = nn.Linear(
            opt["identity_num_features"],
            opt["AADeNorm"]["fc_num_activations"]
            )        
        mask_conv = nn.Conv2d(
            opt["last_activation_size"][1],
            opt["AADeNorm"]["mask_out_channels"],
            opt["AADeNorm"]["mask_kernel_size"],            
            )

    def forward(self, x):
        in_h, idt, attr = x
        h_hat = bn(h)
        mask = nn.functional.sigmoid(mask_conv(h_hat))

        attr_beta = beta_conv(attr)
        attr_gamma = gamma_conv(attr)
        denorm_attr =  h_hat * attr_gamma + attr_beta

        idt_beta = beta_fc(idt)
        idt_gamma = gamma_fc(idt)
        denorm_idt = idt_gamma * idt + idt_beta
        
        return (1-mask) * denorm_attr + mask * denorm_idt