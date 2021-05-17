import os
import logging


import torch
from torch import nn


from src.common import utils
from src.models.faceShifter_model import FaceShifterModel
from src.models.networks.loss import AEINetLoss, MultiScaleGanLoss


logger = logging.getLogger(__name__)


class FaceShifterTrainer:
    def __init__(self, opt): 
        self.opt = opt
        self.trainer_opt = opt["FaceShifterTrainer"]
        self.initialize()


    def initialize(self):
        """
        Create model, optimizers, and criterions; and load saved parameters if 
        available.
        """
        self.last_epoch = 0
        self.last_d_loss = self.last_g_loss = 0
        self.checkpoint = None
        self.device = self.trainer_opt["device"]

        self.model = FaceShifterModel()
        self.model.load_pretrained_idt_encoder(
            self.opt["IdtEncoder"]["pretrained_model"], self.device)

        if self.opt["checkpoint"]["continue"]:
            checkpoint_id = self.opt["checkpoint"]["checkpoint_id"]
            logger.info("Load checkpoint {}".format(checkpoint_id))
            self.load_checkpoint(checkpoint_id, 
                self.opt["checkpoint"]["save_dir"])

        self.model = self.model.to(self.device)
        
        self.create_criterion()
        self.create_optimizer()  
        
        if self.checkpoint is not None:
            self.g_optimizer.load_state_dict(
                self.checkpoint["g_optim_state_dict"])
            self.d_optimizer.load_state_dict(
                self.checkpoint["d_optim_state_dict"])            


    def fit(self, dataloader):
        """Summary
        
        Args:
            dataloader (TYPE): Description
        """
        source_size = int(self.opt["dataset"]["batch_size"] / 2)
        save_dir = self.opt["checkpoint"]["save_dir"]
        max_epochs = self.trainer_opt["max_epochs"]

        for epoch in range(self.last_epoch, max_epochs):
            for bi, batch_data in enumerate(dataloader):
                logger.info("Fetch batch: epoch {} - batch {}".format(epoch, bi))
                xs, xt, reconstructed = batch_data
 
                xs, xt, reconstructed = xs.to(self.device), xt.to(self.device),\
                    reconstructed.to(self.device)
                
                if bi % self.trainer_opt["d_step_per_g"] == 0:
                    logger.info("Fit g: epoch {} - batch {}".format(epoch, bi))
                    self.fit_g(xs, xt, reconstructed)
                
                logger.info("Fit d: epoch {} - batch {}".format(epoch, bi))
                self.fit_d(xs, xt)
                
                if bi % self.opt["checkpoint"]["save_interval"] == 0 and bi > 0:
                    logger.info(
                        "Save checkpoint: epoch {} - batch {}".format(epoch, bi))
                    self.save_checkpoint(epoch, save_dir)


            if bi % self.opt["checkpoint"]["save_interval"] != 0:
                logger.info("Save checkpoint: epoch {}".format(epoch))
                self.save_checkpoint(epoch, save_dir)

            self.update_optimizer(epoch)


    #TODO: save loss values 
    def fit_g(self, xs, xt, reconstructed):
        """Summary
        
        Args:
            xs (TYPE): a batch of source images
            xt (TYPE): a batch of target images
            reconstructed (TYPE): Description
        
        """
        y = self.model(xs, xt, mode=1)
        xs_idt = self.model.get_face_identity(xs)
        y_idt = self.model.get_face_identity(y)
        xt_attr = self.model.get_face_attribute(xt)
        y_attr = self.model.get_face_attribute(y)
        d_output = self.model(None, None, mode=3, x_hat=y)

        loss = self.g_criterion(xt, y, xt_attr, y_attr, xs_idt, y_idt, d_output,
            reconstructed)
        self.last_g_loss = loss.item()
        logger.info("G loss: {}".format(loss.item()))

        self.g_optimizer.zero_grad()
        self.model.detach_d_parameters()
        loss.backward()
        self.model.detach_d_parameters(False)
        self.g_optimizer.step()


    #TODO: the loss is implemented in different from that of project SPADE. Consider if the difference could change performance of the trainer
    #TODO: save loss value
    def fit_d(self, xs, xt):
        """Fit the discriminator using both real and generated data
        
        Args:
            xs (TYPE): a batch of source images
            xt (TYPE): a batch of target images
        """
        real_pred, generated_pred = self.model(xs, xt, mode=2)
  
        real_loss = self.d_criterion(real_pred, True)
        generated_loss = self.d_criterion(generated_pred, False)
        loss = real_loss + generated_loss

        self.last_d_loss = loss.item()
        logger.info("D loss: {}".format(loss.item()))

        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()


    def create_optimizer(self):
        g_params = self.model.get_g_params()
        d_params = self.model.get_d_params()

        lr = self.trainer_opt["optim"]["lr"]
        betas = self.trainer_opt["optim"]["betas"]

        self.g_optimizer = torch.optim.Adam(g_params, lr=lr, betas=betas)
        self.d_optimizer = torch.optim.Adam(d_params, lr=lr, betas=betas)


    def create_criterion(self):
        self.g_criterion = AEINetLoss(self.opt["AEINetLoss"])
        self.d_criterion = MultiScaleGanLoss()


    def update_optimizer(self, epoch):
        """Update parameters of the optimizer.
        
        Args:
            epoch (TYPE): Description
        """
        #TODO: update optimizer


    def save_checkpoint(self, epoch, save_dir):
        """Summary
        
        Args:
            epoch (TYPE): Description
            save_dir (TYPE): Description
        """
        name = "faceShiter_checkpoint_{}.tar".format(epoch)
        checkpoint = {
            "epoch": epoch,
            "g_state_dict": self.model.get_g_state_dict(),
            "d_state_dict": self.model.get_d_state_dict(),
            "d_loss": self.last_d_loss,
            "g_loss": self.last_g_loss,
            "g_optim_state_dict": self.g_optimizer.state_dict(),
            "d_optim_state_dict": self.d_optimizer.state_dict()
        }

        utils.save_state_dict(checkpoint, name, save_dir)


    def load_checkpoint(self, checkpoint_id, load_dir):
        """Summary
        
        Args:
            checkpoint_id (TYPE): Description
            load_dir (TYPE): Description
        """
        name = "{}.tar".format(checkpoint_id)
        checkpoint = utils.load_state_dict(name, load_dir, self.device)

        self.last_epoch = checkpoint["epoch"]
        self.model.load_g_state_dict(checkpoint["g_state_dict"])
        self.model.load_d_state_dict(checkpoint["d_state_dict"])
        self.last_d_loss = checkpoint["d_loss"]
        self.last_g_loss = checkpoint["g_loss"]

        self.checkpoint = {
            "g_optim_state_dict": checkpoint["g_optim_state_dict"],
            "d_optim_state_dict": checkpoint["d_optim_state_dict"]
            }


    def save_model(self, save_dir):
        pass