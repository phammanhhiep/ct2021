import torch
from torch import nn


from networks.generator import AEINet
from networks.discriminator import MultiScaleDiscriminator


class FaceShifterModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.create_g()
        self.create_d()


    def forward(self, x, mode=1):
        """Summary
        
        Args:
            x (TYPE): Description
            mode (int): 1=generator, 2=discriminator 
        """
        h = None
        if mode == 1:
            h = self.g(x)
        elif mode == 2:
            h = self.d(x)
        else:
            raise ValueError("Unknown mode: " + mode)
        return h


    def create_g(self):
        self.g = AEINet(self.opt["aei_generator"])
        #TODO: load pretrained model if not training


    def create_d(self):
        self.d = MultiScaleDiscriminator(self.opt["multi_scale_discriminator"])
        #TODO: load pretrained model if not training


    def get_g_params(self):
        return list(self.g.parameters())


    def get_d_params(self):
        return list(self.d.parameters())


    def save(self): 
        """Save the model
        """








