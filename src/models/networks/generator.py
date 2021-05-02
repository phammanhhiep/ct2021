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


    def forward(self, xs, xt):
        """Summary
        
        Args:
            xs (TYPE): source image
            xt (TYPE): target image
        Returns:
            TYPE: Description
        """
        idt = self.idt_encoder(xs)
        _ = self.attr_encoder(xt)
        multi_level_attrs = self.attr_encoder.get_decoder_feature_maps()
        return self.generator(idt, idt, multi_level_attrs)


    def get_idt_encoder(self):
        return self.idt_encoder


class AADGenerator(nn.Module):

    """The implementaion is based on the ADD generator discussed in "FaceShifter:
    Towards High Fidelity And Occlusion Aware Face Swapping (Li et al)".
    """
    def __init__(self, attr_channel_list, opt):
        """Summary
        
        Args:
            attr_channel_list (TYPE): Description
            opt (TYPE): Description
        """
        super().__init__()
        self.model = []
        num_AADResBlk = opt["num_AADResBlk"]
        AADResBlk_out_channels = opt["AADResBlk_out_channels"]
        idt_channels = opt["idt_channels"]

        self.model.append(nn.ConvTranspose2d(
            idt_channels,
            opt["conv_tr"]["out_channels"]
            opt["conv_tr"]["kernel_size"]
            ))

        in_size = opt["conv_tr"]["out_channels"]
        for n in range(num_AADResBlk):
            attr_channels = attr_channel_list[n]
            out_channels = AADResBlk_out_channels[n]
            self.model.append(AADResBlk(
                in_size, attr_channels, idt_channels, out_channels, opt["AADResBlk"]))
            in_size = out_channels


    def forward(self, x):
        h, idt, attr_sq = x
        for blk, attr in zip(self.model, attr_sq):
            h = blk((h, idt, attr))
        return h


#TODO: implement HEARNet
class HEARNet(nn.Module):
    def __init__(self, opt):
        super().__init__()


    def forward(self, x): pass