import torch
from torch import nn


from src.models.networks.generator import AEINet
from src.models.networks.discriminator import MultiScaleDiscriminator
from src.common import utils


class FaceShifterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.create_g()
        self.create_d()


    #TODO: consider if the unbalanced data fed to discriminator affects its performance
    #TODO: raise exception when mode is 3 but x_hat is not provided
    def forward(self, xs, xt, mode=1, x_hat=None):
        """Both source and target images are fed to the discriminator, beside
        the synthesized images, i.e. the number of real images is doulbe that of
        synthesized image. 
        
        Args:
            xs (TYPE): a batch of source images
            xt (TYPE): a batch of target images
            x_hat (None, optional): a batch of generated images 
            mode (int): 1="generate images", 2="discriminate both real and 
            generated images", 3="discriminate generated images" 
        
        Returns:
            TYPE: Description
        
        Raises:
            ValueError: Description
        """
        h = None
        if mode == 1:
            h = self.g(xs, xt)
        elif mode == 2:
            with torch.no_grad():
                x_hat = self.g(xs, xt)
                x_hat.requires_grad_()
            x = torch.cat((xs, xt))
            h1 = self.d(x)
            h2 = self.d(x_hat)
            h = [h1, h2]
        elif mode == 3:
            h = self.d(x_hat)
        else:
            raise ValueError("Unknown mode: " + mode)
        return h


    def create_g(self):
        self.g = AEINet()
        self.g_checkpoint_name = "{}_g" # e.g. modelid_g
    

    def create_d(self):
        self.d = MultiScaleDiscriminator()
        self.d_checkpoint_name = "{}_d" # e.g. modelid_d


    def get_g_params(self):
        return list(self.g.parameters())


    def get_d_params(self):
        return list(self.d.parameters())


    def get_face_identity(self, x):
        idt_encoder = self.g.get_idt_encoder()
        return idt_encoder(x)


    def get_face_attribute(self, x):
        attr_encoder = self.g.get_attr_encoder()
        attr_encoder(x)
        return attr_encoder.get_decoder_features()


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


    def load(self, model_id, load_dir, device="cpu", load_d=True):
        """Summary
        
        Args:
            model_id (TYPE): Description
            load_dir (TYPE): Description
        """
        utils.load_net(self.g, self.g_checkpoint_name.format(model_id),
            load_dir, device)
        if load_d:
            utils.load_net(self.d, self.d_checkpoint_name.format(model_id),
                load_dir, device)


    def load_pretrained_idt_encoder(self, pth, device):
        self.g.load_pretrained_idt_encoder(pth, device)