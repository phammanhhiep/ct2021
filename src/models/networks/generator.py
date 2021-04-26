import torch
from torch import nn


from models.networks.block import AADResBlk


class AEINet(nn.Module):

    """Combine AADGenerator, Attribute Encoder, and Identity Encoder together.
    """

    def __init__(self):
        super().__init__()


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

