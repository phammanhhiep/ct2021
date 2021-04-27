import torch
from torch import nn


from models.networks.block import AADResBlk
from models.networks.encoder import AttrEncoder, ArcFace


class AEINet(nn.Module):

    """Combine AADGenerator, Attribute Encoder, and Identity Encoder together.
    """
    def __init__(self, opt):
        super().__init__()
        self.attr_encoder = AttrEncoder(opt["attr_encoder"])
        self.idt_encoder = ArcFace(opt["idt_encoder"])
        self.generator = AADGenerator(opt["generator"])


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
        attr = self.attr_encoder.get_decoder_feature_maps()
        return self.generator(idt, idt, attr)


class AADGenerator(nn.Module):

    """The implementaion is based on the ADD generator discussed in "FaceShifter:
    Towards High Fidelity And Occlusion Aware Face Swapping (Li et al)".
    """

    #TODO: provide arguments to the methods
    #TODO: provide options to specify the number of ADDResBlks 
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = []
        num_aadResBlk = opt["num_aadResBlk"]

        for n in range(num_aadResBlk):
            self.model.append(AADResBlk(opt["aadResBlk"][n]))

        self.last_layer = nn.Conv2d(
            opt["last_layer"]["in_channels"],
            opt["last_layer"]["out_channels"],
            opt["last_layer"]["kernel_size"]
            )


    def forward(self, x):
        h, idt, attr_sq = x
        for blk, attr in zip(self.model, attr_sq):
            h = blk((h, idt, attr))
        return self.last_layer(h)

