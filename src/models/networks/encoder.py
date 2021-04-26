import torch
from torch import nn
import torchvision


class AttrEncoder(nn.Module):

    """A U-net-like network to extract attribute from the target image.
    """

    def __init__(self, opt):
        super().__init__()
        self.eopt = opt["encoder_opt"]
        self.dopt = opt["decoder_opt"]
        self.ebn = eopt["num_blks"]
        self.dbn = dopt["num_blks"]

        self.decoder_features = []
        self.encoder_features = []
        self.encoder = []
        self.decoder = []
        for n in range(ebn):
            s = nn.sequential(
                nn.Conv2d(
                    eopt["conv"]["in_channels"][n],
                    eopt["conv"]["out_channels"][n],
                    eopt["conv"]["kernel_size"][n]                    
                    ),
                nn.BatchNorm2d(eopt["conv"]["out_channels"][n]),
                nn.LeakyReLU()
                )
            s.register_forward_hook(encoder_forward_hook) #TODO: verify if the hook works
            self.encoder.append(x)

        for n in range(dbn):
            self.decoder.append(nn.sequential(
                nn.ConvTranspose2d(
                    dopt["conv"]["in_channels"][n],
                    dopt["conv"]["out_channels"][n],
                    dopt["conv"]["kernel_size"][n]                    
                    ),
                nn.BatchNorm2d(dopt["conv"]["out_channels"][n]),
                nn.LeakyReLU()   
                ))

        self.last_layer = nn.Upsample(**opt["last_layer"]) #TODO: review the arguments


    def encoder_forward_hook(self, x):
        """Append the encoder layer output to a list.
        
        Args:
            x (TYPE): output of an encoder layer
        """
        self.encoder_features.append(x)


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


    #TODO: make sure input and out channel in decoder part are correct
    def forward(self, x):
        for n in range(self.ebn):
            x = self.encoder[n](x)

        for n in range(self.dbn):
            x = self.decoder[n](x)
            x = self.decoder_post_forward(x, n)
        y = self.last_layer(x)
        self.decoder_post_forward(y)
        return y


    def get_decoder_feature_maps(self):
        return self.decoder_features


#TODO: The ArcFace is not trained yet for face recognition (face identity), but just use the general pretrained object recognition model from pytorch. 
class ArcFace(nn.Module):
    def __init__(self, opt):
        self.resnet50 = torchvision.models.resnet50(pretrained=True)


    def forward(self, x):
        return self.resnet50(x)



