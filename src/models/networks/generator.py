import torch
from torch import nn


from src.models.networks.block import AADResBlk
from src.models.networks.encoder import AttrEncoder, IdtEncoder


class AEINet(nn.Module):

    """Combine AADGenerator, Attribute Encoder, and Identity Encoder together.
    """
    def __init__(self, opt):
        super().__init__()
        self.attr_encoder = AttrEncoder(opt["AttrEncoder"])
        self.idt_encoder = IdtEncoder(opt["ArcFace"])
        self.generator = AADGenerator(opt["AADGenerator"])


    def forward(self, xs, xt):
        """Summary
        
        Args:
            xs (TYPE): a batch of source images
            xt (TYPE): a batch of target images
        Returns:
            TYPE: Description
        """
        idt = self.idt_encoder(xs)
        _ = self.attr_encoder(xt)
        multi_level_attrs = self.attr_encoder.get_decoder_features()
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
        self.num_AADResBlk = opt["num_AADResBlk"]
        attr_channel_list = opt["attr_channel_list"]
        AADResBlk_out_channel_list = opt["AADResBlk_out_channel_list"]
        idt_channels = opt["idt_channels"]
        conv_tr_out_channels = opt["conv_tr"]["out_channels"]
        conv_tr_kernel_size = 2
        conv_tr_stride = 1
        conv_tr_padding = 0

        self.conv_tr = nn.ConvTranspose2d(
            idt_channels,
            conv_tr_out_channels,
            conv_tr_kernel_size,
            conv_tr_stride,
            conv_tr_padding
            )

        in_channels = conv_tr_out_channels
        for n in range(self.num_AADResBlk):
            attr_channels = attr_channel_list[n]
            out_channels = AADResBlk_out_channel_list[n]
            self.model.append(AADResBlk(
                in_channels, attr_channels, idt_channels, out_channels, 
                opt["AADResBlk"]))
            in_channels = out_channels


    def forward(self, idt, attrs):
        """Summary
        
        Args:
            idt (TYPE): a batch of identity features of size (B, C, 1, 1)
            attrs (TYPE): a batch of squences of attributes of size
            (num_attr, B, C, H, W)
        
        Returns:
            TYPE: Description
        """
        h = self.conv_tr(idt)
        for i in range(self.num_AADResBlk):
            h = self.model[i]((h, idt, attrs[i]))
            if i < self.num_AADResBlk - 1:
                h = self.upsample(h)
        return h


    def upsample(self, x):
        return nn.functional.interpolate(x, 
            scale_factor=(self.upsample_scale, self.upsample_scale),
            mode="bilinear", align_corners=True)


#TODO: implement HEARNet
class HEARNet(nn.Module):
    def __init__(self, opt):
        super().__init__()


    def forward(self, x): pass