import logging


import torch
from torch import nn


logger = logging.getLogger(__name__)


class AEINetLoss(nn.Module):

    """The loss function of the generator AEINet
    """

    def __init__(self, opt):
        super().__init__()
        self.adv_criterion = MultiScaleGanLoss(opt["GANLoss"])
        self.attr_criterion = AttrLoss(opt["AttrLoss"])
        self.rec_criterion = RecLoss(opt["RecLoss"])
        self.idt_criterion = IdtLoss(opt["IdtLoss"])
        self.AttrLoss_w = opt["loss_weights"]["AttrLoss"]
        self.RecLoss_w = opt["loss_weights"]["RecLoss"]
        self.IdtLoss_w = opt["loss_weights"]["IdtLoss"]


    def forward(self, xt, y, xt_attr, y_attr, xs_idt,
        y_idt, d_output, reconstructed=False):
        """Summary
        
        Args:
            xt (TYPE): a batch of target images
            y (TYPE): a batch of generated images
            xt_attr (TYPE): a batch of multi-level attributes of the target images
            y_attr (TYPE): a batch of multi-level attributes of the generated 
            images
            xs_idt (TYPE): a batch of identities of the source images
            y_idt (TYPE): a batch of identities of the generated images
            d_output (TYPE): a batch of outputs of the discriminator
            reconstructed (bool, optional): whether source and target images are
            the same
        
        Returns:
            TYPE: Description
        """
        adv_loss = self.adv_criterion(d_output, t=-1, compute_d_loss=False)
        attr_loss = self.attr_criterion(xt_attr, y_attr)
        rec_loss = self.rec_criterion(xt, y, reconstructed=reconstructed)
        idt_loss = self.idt_criterion(xs_idt, y_idt)
        return adv_loss + attr_loss * self.AttrLoss_w + \
            rec_loss * self.RecLoss_w + idt_loss * self.IdtLoss_w


class MultiScaleGanLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()


    #TODO: review the implementation of the hinge loss in project https://trello.com/c/cA9SYd0x. There are some complications which are not handled in here.
    def forward(self, y, label=True, compute_d_loss=True):
        """Sum the losses of individual discriminators. 
        
        Args:
            y (TYPE): multi-scale discriminator outputs, each of
            which is a list of 4D tensors; the size of ech component of y is 
            (B, 1, H, W).
            label (TYPE): True for real image; False for fake image
            compute_d_loss (bool, optional): compute d loss or g loss
        
        Returns:
            float: Description
        """ 
        t = 1 if label else -1
        loss = 0
        for yi in y:
            loss += self._hinge_loss(yi, t, compute_d_loss)
        return loss


    def _hinge_loss(self, y, t=1, compute_d_loss=True):
        """
        It is required to compute the mean in spatial dimension first, because 
        patchGAN is supposed to be used.
        Args:
            y (TYPE): a batch of outputs from a discriminator of size 
            (B, 1, H, W)
            t (bool, optional): 1=real, -1=fake 
            compute_d_loss (bool, optional): compute loss for d or g
        Returns:
            float: Description
        """
        loss = 0
        if compute_d_loss:
            loss = -torch.mean(
                torch.min(
                    torch.zeros(y.size()), 
                    torch.mean(-1 + t * y, dim=(2,3))))
        else:
            loss = -torch.mean(torch.mean(y, dim=(2,3)))
        return loss


class AttrLoss(nn.Module):

    """The loss is a factor of the summation of L2 distances between 
    corresponding multi-level feature maps of the target image and the synthesized
    image, obtained from the attribute decoder. 
    """
    
    def __init__(self, opt):
        super().__init__()


    def forward(self, x_attrs, y_attrs):
        """
        Reshaping x_attrs and y_attrs is required because L2 is computed for
        each pair of individual attributes.

        Args:
            x_attrs (TYPE): a batch of multi-level attributes of the target 
            image, with size (B, num_attrs, C, H, W)
            y_attrs (TYPE): a batch of multi-level attributes of the synthesized 
            image, with size (B, num_attrs, C, H, W)
        
        Returns:
            TYPE: Description
        """
        loss = 0
        x_attrs = x_attrs.reshape(1,0,2,3,4)
        y_attrs = y_attrs.reshape(1,0,2,3,4)
        for x, y in zip(x_attrs, y_attrs):
            loss += nn.functional.mse_loss(x, y)
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
            x (TYPE): a batch of target (or source) image
            y (TYPE): a batch of corresponding sythesized image
            reconstructed (bool, optional): whether same source and target image
        
        Returns:
            TYPE: Description
        """
        return 0 if not reconstructed else \
            0.5 * nn.functional.mse_loss(x, y, reduction="sum")     


class IdtLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()


    def forward(self, x_idt, y_idt):
        """The loss is summation of individual loss for each pair of identity
        (x,y).
        
        Args:
            x_idt (TYPE): a batch of identities of real images obtained from an
            indentity encoder, with size (B, C, H, W)
            y_idt (TYPE): corresponding batch of identities of the synthesized
            images
        
        Returns:
            float: in range [0, 1]
        """
        return torch.sum(
            1 - nn.functional.cosine_similarity(x_idt, y_idt, dim=1))