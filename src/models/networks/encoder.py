import logging


import torch
from torch import nn
import torchvision


class AttrEncoder(nn.Module):

    """A U-net-like network to extract attributes from the target image.
    """

    def __init__(self):
        """Summary
        """
        super().__init__()
        in_channels = 3
        LeakyReLU_slope = 0.2

        encoder_out_channel_list = [32, 64, 128, 256, 512, 1024, 1024]
        conv_kernel_size = 4
        conv_stride = 2
        conv_padding = 1

        deccoder_in_channel_list = [1024, 2048, 1024, 512, 256, 128]
        decoder_out_channel_list = [1024, 512, 256, 128, 64, 32]
        conv_tr_kernel_size = 4
        conv_tr_stride = 2
        conv_tr_padding = 1

        self.ebn = 7
        self.dbn = 6
        self.decoder_features = []
        self.encoder_features = []
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()


        for n in range(self.ebn):
            out_channels = encoder_out_channel_list[n]
            sq = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, conv_kernel_size, 
                    conv_stride, conv_padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(LeakyReLU_slope)
                )
            in_channels = out_channels
            sq.register_forward_hook(self.encoder_forward_hook)
            self.encoder.append(sq)

        self.encoder = nn.Sequential(*self.encoder)

        for n in range(self.dbn):
            out_channels = decoder_out_channel_list[n]
            in_channels = deccoder_in_channel_list[n]
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 
                    conv_tr_kernel_size, conv_tr_stride, conv_tr_padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(LeakyReLU_slope)   
                ))

    def pre_forward(self):
        self.decoder_features = []


    def encoder_forward_hook(self, module, x, y):
        """Append the encoder layer output to a list.
        
        Args:
            x (TYPE): output of an encoder layer
        """
        self.encoder_features.append(y)


    def forward(self, x):
        self.pre_forward()
        x = self.encoder(x)
        self.decoder_features.append(x)

        for n in range(self.dbn):
            x = self.decoder[n](x)
            x = torch.cat([x, self.encoder_features[-2 - n]], dim=1)
            self.decoder_features.append(x)

        x = nn.functional.interpolate(x, scale_factor=(2,2), mode='bilinear', 
            align_corners=True)
        self.decoder_features.append(x)
        
        return x


    def get_decoder_features(self):
        return self.decoder_features


class IdtEncoder(nn.Module):

    #TODO: Provide an option to choose different model from torchvision
    def __init__(self):
        """The encoder is assumed to be a pretrained model, and thus should not
        be optimzed with the rest of the model.
        
        Args:
            opt (TYPE): Description
        """
        super().__init__()
        self.model = torchvision.models.resnet101(num_classes=256)
        self.model.requires_grad_(False)
        

    #TOD0: consider to downsample the input as in https://github.com/phammanhhiep/unoffical-faceshifter
    #TODO: consider to use the last layer before the FC layer is used as identity features, as described in the original paper FaceShifter 
    def forward(self, x):
        """
        Args:
            x (TYPE): a batch of tensors (representing images)
        
        Returns:
            tensor: of size (B, C, 1, 1)
        """
        h = self.model(x)
        return h.unsqueeze(-1).unsqueeze(-1)


    def load(self, pth, device="cpu"):
        self.model.load_state_dict(torch.load(pth, map_location=device))