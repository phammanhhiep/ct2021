import logging


import torch
from torch import nn


logger = logging.getLogger(__name__)


class AEINetLoss(nn.Module):

    """The loss function of the generator
    """

    def __init__(self, opt):
        super().__init__()
        self.adv_criterion = MultiScaleGanLoss(opt["adv_loss"])
        self.attr_criterion = AttrLoss(opt["attr_loss"])
        self.rec_criterion = RecLoss(opt["rec_loss"])
        self.idt_criterion = IdtLoss(opt["idt_loss"])
        self.attr_w, self.rec_w, self.idt_w = opt["loss_weights"]


    def forward(self, xt, y, xt_features, y_attr_features, xs_features,
        y_idt_features, d_output, reconstructed=False):
        adv_loss = self.adv_criterion(d_output, t=-1, compute_d_loss=False)
        attr_loss = self.attr_criterion(xt_features, y_attr_features)
        rec_loss = self.rec_criterion(xt, y, reconstructed=reconstructed)
        idt_loss = self.idt_criterion(xs_features, y_idt_features)
        return adv_loss + attr_loss * self.attr_w + rec_loss*self.rec_w \
            + idt_loss * self.idt_w


class MultiScaleGanLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()


    #TODO: review the implementation of the hinge loss in project https://trello.com/c/cA9SYd0x. There are some complications which are not handled in here.
    #TODO: verify size of input y 
    def forward(self, y, t=1, compute_d_loss=True):
        """Reshape the input and compute loss for individual scales first, and
        then sum up the losses. 
        
        Args:
            y (TYPE): a batch of multi-scale discriminator outputs, each of
            which is a list of 5D tensors; size of y is expected to be
            (batch_size, num_d, 1, H, W)
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
            y (TYPE): a batch of outputs from a discriminator of size 
            (B, 1, H, W)
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

    """The loss is a factor of the summation of L2 distances between 
    corresponding multi-level feature maps of the target image and the synthesized
    image, obtained from the attribute decoder. 
    """
    
    def __init__(self, opt):
        super().__init__()


    def forward(self, xfsq, yfsq):
        """
        Args:
            xfsq (TYPE): sequence of multi-level attributes of the target image
            yfsq (TYPE): sequence of multi-level attributes of the synthesized image
        
        Returns:
            TYPE: Description
        """
        loss = 0
        for xf, yf in zip(xfsq, yfsq):
            loss += nn.functional.mse_loss(xf, yf, reduction="sum")
        return 0.5 * loss


class RecLoss(nn.Module):

    """The reconstruction loss is a multiple of L2 distance between the target
    image and the synthesis image.
    """

    def __init__(self, opt):
        super().__init__()


    def forward(self, x, y, reconstructed=False):
        """Summary
        
        Args:
            x (TYPE): target (or source) image
            y (TYPE): sythesized image
            reconstructed (bool, optional): whether same source and target image
        
        Returns:
            TYPE: Description
        """
        loss = 0 if not reconstructed else \
            0.5 * nn.functional.mse_loss(x, y, reduction="sum")
        return loss     


class IdtLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()


    #TODO: verify the assumption about the size of inputs, and if the dim parameter of cosine_similarity is correct
    #TODO: verify summing up across spatial dimensions is appropriate to represent the loss
    def forward(self, xf, yf):
        """Summary
        
        Args:
            xf (TYPE): feature map of the last layer before fc layer in
            the identity encoder, of the source image. The shape is supposed to 
            be (B,C,W,H)
            yf (TYPE): the feature map of the synthesized image.
        
        Returns:
            TYPE: A tensor of size (B, 1)
        """
        return torch.sum(
            1 - nn.functional.cosine_similarity(xf, yf, dim=1), 
            dim=(1,2))