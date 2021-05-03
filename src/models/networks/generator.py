import torch
from torch import nn


from models.networks.block import AADResBlk
from models.networks.encoder import AttrEncoder, IdtEncoder


class AEINet(nn.Module):

    """Combine AADGenerator, Attribute Encoder, and Identity Encoder together.
    """
    def __init__(self, opt):
        super().__init__()
        self.attr_encoder = AttrEncoder(opt["AttrEncoder"])
        self.idt_encoder = IdtEncoder(opt["ArcFace"])
        self.generator = AADGenerator(opt["AADGenerator"])


    def forward(self, x):
        """Summary
        
        Args:
            x (TYPE): [source images, target images]
        Returns:
            TYPE: Description
        """
        xs, xt = x
        idt = self.idt_encoder(xs)
        _ = self.attr_encoder(xt)
        multi_level_attrs = self.attr_encoder.get_decoder_feature_maps()
        return self.generator(idt, multi_level_attrs)


    def get_idt_encoder(self):
        return self.idt_encoder


    def get_attr_encoder(self):
        return self.attr_encoder


class AADGenerator(nn.Module):

    """The implementaion is based on the ADD generator discussed in "FaceShifter:
    Towards High Fidelity And Occlusion Aware Face Swapping (Li et al)".
    """
    def __init__(self, opt):
        """Summary
        
        Args:
            attr_channel_list (TYPE): Description
            opt (TYPE): Description
        """
        super().__init__()
        self.model = []
        self.upsample_scale = opt["upsample_scale"]
        attr_channel_list = opt["attr_channel_list"]
        num_AADResBlk = opt["num_AADResBlk"]
        AADResBlk_out_channel_list = opt["AADResBlk_out_channel_list"]
        idt_channels = opt["idt_channels"]

        self.conv_tr = nn.ConvTranspose2d(
            idt_channels,
            opt["conv_tr"]["out_channels"],
            opt["conv_tr"]["kernel_size"],
            opt["conv_tr"]["stride"],
            opt["conv_tr"]["padding"]
            )

        in_size = opt["conv_tr"]["out_channels"]
        for n in range(num_AADResBlk):
            attr_channels = attr_channel_list[n]
            out_channels = AADResBlk_out_channel_list[n]
            self.model.append(AADResBlk(
                in_size, attr_channels, idt_channels, out_channels, 
                opt["AADResBlk"]))
            in_size = out_channels


    def forward(self, idt, attr_sq):
        """Summary
        
        Args:
            idt (TYPE): a batch of identity features of size (B, C, 1, 1)
            attr_sq (TYPE): a batch of squences of attributes of size
            (num_attr, B, C, H, W)
        
        Returns:
            TYPE: Description
        """
        h = self.conv_tr(idt)
        for blk, attr in zip(self.model, attr_sq):
            h = self.upsample(blk((h, idt, attr)))
        return h


    def upsample(self, x):
        return nn.functional.interpolate(x, 
            scale_factor=(1,1,self.upsample_scale, self.upsample),
            mode="bilinear", align_corners=True)


#TODO: implement HEARNet
class HEARNet(nn.Module):
    def __init__(self, opt):
        super().__init__()


    def forward(self, x): pass