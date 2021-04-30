import torch
from torch import nn


from networks.generator import AEINet
from networks.discriminator import MultiScaleDiscriminator
from common import utils


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
        self.g = AEINet(self.opt["aei_net"])
        self.g_checkpoint_name = "{}_g" # modelid_g
    

    def create_d(self):
        self.d = MultiScaleDiscriminator(self.opt["multi_scale_discriminator"])
        self.d_checkpoint_name = "{}_d" # e.g. modelid_g


    def get_g_params(self):
        return list(self.g.parameters())


    def get_d_params(self):
        return list(self.d.parameters())


    def get_idt_encoder(self):
        return self.g.get_idt_encoder()


    def save(self, model_id, save_dir): 
        """Save the model
        
        Args:
            model_id (TYPE): Description
            save_dir (TYPE): Description
        """
        utils.save_net(self.g, self.g_checkpoint_name.format(model_id),
            save_dir)
        utils.save_net(self.d, self.d_checkpoint_name.format(model_id),
            save_dir)


    def load(self, model_id, load_dir):
        """Summary
        
        Args:
            model_id (TYPE): Description
            load_dir (TYPE): Description
        """
        utils.load_net(self.g, self.g_checkpoint_name.format(model_id),
            save_dir)
        utils.load_net(self.d, self.d_checkpoint_name.format(model_id),
            save_dir)
