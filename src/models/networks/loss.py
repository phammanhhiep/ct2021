import logging


import torch
from torch import nn


logger = logging.getLogger(__name__)


class AEINetLoss(nn.Module):

    """The loss function of the generator AEINet
    """

    def __init__(self, opt):
        super().__init__()
        self.adv_criterion = MultiScaleGanLoss()
        self.attr_criterion = AttrLoss()
        self.rec_criterion = RecLoss()
        self.idt_criterion = IdtLoss()
        self.AttrLoss_w = opt["weights"]["AttrLoss"]
        self.RecLoss_w = opt["weights"]["RecLoss"]
        self.IdtLoss_w = opt["weights"]["IdtLoss"]


    def forward(self, xt, y, xt_attr, y_attr, xs_idt, y_idt, d_output, 
        reconstructed):
        """Summary
        
        Args:
            xt (TYPE): a batch of target images
            y (TYPE): a batch of generated images
            xt_attr (TYPE): a list of batches of multi-level attributes of the 
            target images, each of which of size (B, C, H, W)
            y_attr (TYPE): a list of batches of multi-level attributes of the 
            synthesized images, each of which of size (B, C, H, W)
            xs_idt (TYPE): a batch of identities of the source images
            y_idt (TYPE): a batch of identities of the generated images
            d_output (TYPE): multi-scale discriminator outputs which is a list of 
            4D tensors; the size of ech component of y is (B, 1, H, W)
            reconstructed (TYPE): indicate whether source and target images are
            the same; its size is (B,1,1,1)
        
        Returns:
            TYPE: Description
        """
        adv_loss = self.adv_criterion(d_output, label=False, compute_d_loss=False)
        attr_loss = self.attr_criterion(xt_attr, y_attr)
        rec_loss = self.rec_criterion(xt, y, reconstructed)
        idt_loss = self.idt_criterion(xs_idt, y_idt)

        loss = adv_loss + attr_loss * self.AttrLoss_w + \
            rec_loss * self.RecLoss_w + idt_loss * self.IdtLoss_w

        return loss, adv_loss, attr_loss, rec_loss, idt_loss


class MultiScaleGanLoss(nn.Module):
    def __init__(self):
        super().__init__()


    #TODO: review the implementation of the hinge loss in project https://trello.com/c/cA9SYd0x. There are some complications which are not handled in here.
    def forward(self, y, label=True, compute_d_loss=True):
        """Sum the losses of individual discriminators. 
        
        Args:
            y (TYPE): multi-scale discriminator outputs which is a list of 
            4D tensors; the size of ech component of y is (B, 1, H, W)
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
                    torch.zeros(y.size()[:2]), 
                    torch.mean(-1 + t * y, dim=(2,3))))
        else:
            loss = -torch.mean(torch.mean(y, dim=(2,3)))
        return loss


class AttrLoss(nn.Module):

    """The loss is a factor of the summation of L2 distances between 
    corresponding multi-level feature maps of the target image and the synthesized
    image, obtained from the attribute decoder. 
    """
    
    def __init__(self):
        super().__init__()


    def forward(self, x_attrs, y_attrs):
        """
        Reshaping x_attrs and y_attrs is required because L2 is computed for
        each pair of individual attributes.

        Args:
            x_attrs (TYPE): a list of batches of multi-level attributes of the 
            target images, each of which of size (B, C, H, W)
            y_attrs (TYPE): a list of batches of multi-level attributes of the 
            synthesized images, each of which of size (B, C, H, W)
        
        Returns:
            TYPE: Description
        """
        loss = 0
        for x, y in zip(x_attrs, y_attrs):
            loss += nn.functional.mse_loss(x, y)
        return 0.5 * loss


class RecLoss(nn.Module):

    """The reconstruction loss is a multiple of L2 distance between the target
    image and the synthesis image.
    """

    def __init__(self):
        super().__init__()


    def forward(self, x, y, reconstructed):
        """Summary
        
        Args:
            x (TYPE): a batch of target (or source) image
            y (TYPE): a batch of corresponding sythesized image
            reconstructed (tensor): indicate whether source and target images are
            the same; its size is (B,1,1,1)
        
        Returns:
            float: Description
        """
        x = x * reconstructed
        y = y * reconstructed
        return 0.5 * nn.functional.mse_loss(x, y, reduction='mean')     


class IdtLoss(nn.Module):
    def __init__(self):
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
            float: Description
        """
        return torch.mean(
            1 - nn.functional.cosine_similarity(x_idt, y_idt, dim=1))