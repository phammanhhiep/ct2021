import logging


import torch
from torch import nn


logger = logging.getLogger(__name__)


class AEINetLoss(nn.Module):
    def __init__(self, opt): pass


class GanLoss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, multi_y, t): pass



    def _hinge_loss(self, multi_y, t):
        """Summation of individual hinge losses.
        
        Args:
            multi_y (TYPE): prediction
            t (TYPE): 1=real, and 0=synthesis
        """


    def _feature_match_loss(self):
        """Summary
        """

class AttrLoss(nn.Module):

    """The loss is a multiple of the summation of L2 distances between 
    corresponding multi-level feature maps of the target image and the synthesis
    image, obtained from the attribute decoder. 
    """
    
    def __init__(self):
        super().__init__()


    def forward(self, x, y):
        """
        Args:
            x (TYPE): Description
            y (TYPE): Description
        """
        return 0.5 * nn.functional.mse_loss(x, y, reduction="sum")


class RecLoss(nn.Module):

    """The reconstruction loss is a multiple of L2 distance between the target
    image and the synthesis image.
    """

    def __init__(self):
        super().__init__()


    def forward(self, x, y):
        return 0.5 * nn.functional.mse_loss(x, y, reduction="sum")        


class IdtLoss(nn.Module):
    def __init__(self):
        super().__init__()


    #TODO: correct the dim parameter of cosine_similarity
    def forward(self, x, y):
        return 1 - nn.functional.cosine_similarity(x, y)