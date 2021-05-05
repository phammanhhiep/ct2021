import torch
from torch import nn


from networks.generator import AEINet
from networks.discriminator import MultiScaleDiscriminator
from common import utils


class FaceShifterModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.create_g(opt["AEINet"])
        self.create_d(opt["MultiScaleDiscriminator"])


    #TODO: consider if the unbalanced data fed to discriminator affects its performance
    def forward(self, xs, xt, mode=1):
        """Both source and target images are fed to the discriminator, beside
        the synthesized images, i.e. the number of real images is doulbe that of
        synthesized image. 
        
        Args: 
            xs (TYPE): a batch of source images
            xt (TYPE): a batch of target images
            mode (int): 1=generator, 2=discriminator 
        """
        h = None
        if mode == 1:
            h = self.g(xs, xt)
        elif mode == 2:
            x_hat = self.g(xs, xt)
            x = torch.cat((xs, xt))
            h = self.d(x, x_hat)
        else:
            raise ValueError("Unknown mode: " + mode)
        return h


    def create_g(self, opt):
        self.g = AEINet(opt)
        self.g_checkpoint_name = "{}_g" # modelid_g
    

    def create_d(self, opt):
        self.d = MultiScaleDiscriminator(opt)
        self.d_checkpoint_name = "{}_d" # e.g. modelid_g


    def get_g_params(self):
        return list(self.g.parameters())


    def get_d_params(self):
        return list(self.d.parameters())


    #TODO: review the output
    def get_face_identity(self, x):
        idt_encoder = self.g.get_idt_encoder()
        return idt_encoder(x)


    def get_face_attribute(self, x):
        attr_encoder = self.g.get_attr_encoder()
        attr_encoder(x)
        return attr_encoder.get_decoder_feature_maps()


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
            load_dir)
        utils.load_net(self.d, self.d_checkpoint_name.format(model_id),
            load_dir)
