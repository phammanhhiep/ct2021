import logging


import torch
from torch import nn
import torchvision


class AttrEncoder(nn.Module):

    """A U-net-like network to extract attributes from the target image.
    """

    def __init__(self, opt):
        """Summary
        
        Args:
            opt (TYPE): Description
            in_channels (int, optional): Image is fed directly to the network, 
            and thus the in_channel is assumed to be 3
        """
        super().__init__()
        in_channels = opt["in_channels"]
        eopt = opt["encoder"]
        dopt = opt["decoder"]
        LeakyReLU_slope = opt["LeakyReLU_slope"]

        encoder_out_channel_list = eopt["conv"]["out_channel_list"] 
        conv_kernel_size = eopt["conv"]["kernel_size"]
        conv_stride = eopt["conv"]["stride"]
        conv_padding = eopt["conv"]["padding"]

        deccoder_in_channel_list = dopt["conv_tr"]["in_channel_list"]
        decoder_out_channel_list = dopt["conv_tr"]["out_channel_list"]
        conv_tr_kernel_size = dopt["conv_tr"]["kernel_size"]
        conv_tr_stride = dopt["conv_tr"]["stride"]
        conv_tr_padding = dopt["conv_tr"]["padding"]

        self.ebn = eopt["num_blks"]
        self.dbn = dopt["num_blks"]
        self.decoder_features = []
        self.encoder_features = []
        self.encoder = []
        self.decoder = []


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
    def __init__(self, opt):
        self.model = torchvision.models.resnet101(
            num_classes=opt["num_classes"])

        if opt["pretrained_model"] is not None:
            self.model.load_state_dict(torch.load(opt["pretrained_model"], 
                map_location=opt["map_location"]))


    def forward(self, x):
        return self.model(x)