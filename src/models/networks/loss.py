import logging


import torch
from torch import nn


logger = logging.getLogger(__name__)


class AEINetLoss(nn.Module):
    def __init__(self, opt): pass


class MultiScaleGanLoss(nn.Module):
    def __init__(self):
        super().__init__()


    #TODO: review the implementation of the hinge loss in project https://trello.com/c/cA9SYd0x. There are some complications which are not handled in here. Also, the implementation may be
    #TODO: verify size of input y 
    def forward(self, y, t, compute_d_loss=True):
        """Reshape the input and compute loss for individual scales first, and
        then sum up the losses. 
        
        Args:
            y (TYPE): a batch of multi-scale discriminator outputs, each of
            which is a list of 3D tensors; size of y is expected to be
            (batch_size, multi_scale_size, discriminator_output) 
            t (TYPE): 1 for real image; -1 for fake image
            compute_d_loss (bool, optional): compute d loss or g loss
        
        Returns:
            TYPE: Description
        """ 
        y = torch.reshape(y, (1, 0, 2))
        loss = torch.zeros(y.size()[1])
        for yi in y:
            loss += self._hinge_loss(yi, t, compute_d_loss)
        return loss


    def _hinge_loss(self, y, t=1, compute_d_loss=True):
        """Summation of individual hinge losses.
        
        Args:
            y (TYPE): a batch of outputs
            t (bool, optional): 1=real, -1=fake 
            compute_d_loss (bool, optional): compute loss for d or g.
        """
        loss = 0
        if compute_d_loss:
            loss = -torch.mean(
                torch.min(self.zeros(y.size()), -1 + t * y), 
                dim=0)
        else:
            loss = -torh.mean(y, dim=0)
        return loss


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