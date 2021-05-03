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
        self.ebn = self.eopt["num_blks"]
        self.dbn = self.dopt["num_blks"]
        self.decoder_features = []
        self.encoder_features = []
        self.encoder = []
        self.decoder = []


        for n in range(self.ebn):
            out_channels = self.eopt["conv"]["out_channels"][n]
            sq = nn.sequential(
                nn.Conv2d(
                    in_channels
                    out_channels,
                    self.eopt["conv"]["kernel_size"],
                    self.eopt["conv"]["stride"],
                    self.eopt["conv"]["padding"]                  
                    ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
                )
            sq.register_forward_hook(encoder_forward_hook)
            self.encoder.append(sq)
        self.encoder = nn.sequential(self.encoder)

        for n in range(self.dbn):
            in_channels = out_channels
            out_channels = self.dopt["conv_tr"]["out_channels"][n]
            self.decoder.append(nn.sequential(
                nn.ConvTranspose2d(
                    in_channels
                    out_channels,
                    self.dopt["conv_tr"]["kernel_size"],
                    self.dopt["conv_tr"]["stride"],
                    self.dopt["conv_tr"]["padding"]                     
                    ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()   
                ))

    def pre_forward(self):
        self.decoder_features = []


    def encoder_forward_hook(self, x, y):
        """Append the encoder layer output to a list.
        
        Args:
            x (TYPE): output of an encoder layer
        """
        self.encoder_features.append(y)


    def decoder_post_forward(self, x1, n=None):
        """Does two things
        If n is provided, get the corresponding encoder output and concatenate
        the decoder and encoder layer outputs.
        Append the decoder layer output to a list.
        
        Args:
            x1 (TYPE): output of the target decoder layer.
            n (None, optional): index of the target decoder layer.
        
        Returns:
            TYPE: Description
        """
        x = x1
        if n is not None:
            x = torch.cat([x1, self.encoder_features[-2 - n]], dim=1)
        self.decoder_features.append(x)
        return x


    def forward(self, x):
        self.pre_forward()
        x = self.encoder[n](x)

        for n in range(self.dbn):
            x = self.decoder[n](x)
            x = self.decoder_post_forward(x, n)

        x = nn.functional.interpolate(x, size=2, mode='bilinear', 
            align_corners=True)
        return self.decoder_post_forward(x)


    def get_decoder_feature_maps(self):
        return self.decoder_features


class IdtEncoder(nn.Module):

    #TODO: Provide an option to choose different model from torchvision
    def __init__(self, opt):
        self.model = torchvision.models.resnet101(
            num_classes=opt["num_classes"])
        self.model.load_state_dict(
            torch.load(opt["pretrained_model"], 
            map_location=opt["map_location"]))


    def forward(self, x):
        return self.model(x)