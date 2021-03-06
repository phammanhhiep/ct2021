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
        """Summary
        
        Args:
            opt (dict): Description
        """
        self.opt = opt
        self.trainer_opt = self.opt["FaceShifterTrainer"]
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
        self.log_dir = self.opt["log"]["root_dir"]

        self.model = FaceShifterModel()
        self.model.load_pretrained_idt_encoder(
            self.opt["IdtEncoder"]["pretrained_model"], self.device)

        if self.opt["checkpoint"]["continue"]:
            checkpoint_id = self.opt["checkpoint"]["checkpoint_id"]
            logger.info("Load checkpoint {}".format(checkpoint_id))
            self.load_checkpoint(checkpoint_id, 
                self.opt["checkpoint"]["root_dir"])

        self.model = self.model.to(self.device)
        
        self.create_criterion()
        self.create_optimizer()  
        
        if self.checkpoint is not None:
            self.g_optimizer.load_state_dict(
                self.checkpoint["g_optim_state_dict"])
            self.d_optimizer.load_state_dict(
                self.checkpoint["d_optim_state_dict"])

        self.kill_signal_handler = utils.KillSignalHandler()           


    def fit(self, dataloader):
        """Summary
        
        Args:
            dataloader (TYPE): Description
        """
        source_size = int(self.opt["dataset"]["batch_size"] / 2)
        save_dir = self.opt["checkpoint"]["root_dir"]
        max_epochs = self.trainer_opt["max_epochs"]
        save_checkpoint_msg = "Save checkpoint: epoch {} - batch {}"
        save_interval = self.opt["checkpoint"]["save_interval"]

        for epoch in range(self.last_epoch, max_epochs):
            for bi, batch_data in enumerate(dataloader):
                if self.kill_signal_handler.received:
                    logger.info("Receive kill signal")
                    prev_bi = bi - 1
                    saved = (prev_bi % save_interval) == 0
                    if prev_bi == 0 or not saved:
                        logger.info(save_checkpoint_msg.format(epoch, prev_bi))
                        self.save_checkpoint(epoch, save_dir)
                    return 1

                logger.info("Fetch batch: epoch {} - batch {}".format(epoch, bi))
                xs, xt, reconstructed = batch_data
                xs = xs.to(self.device)
                xt = xt.to(self.device)
                reconstructed = reconstructed.to(self.device)
                
                if bi % self.trainer_opt["d_step_per_g"] == 0:
                    logger.info("Fit g: epoch {} - batch {}".format(epoch, bi))
                    self.fit_g(xs, xt, reconstructed)
                    logger.info("Complete Fit g")
                
                logger.info("Fit d: epoch {} - batch {}".format(epoch, bi))
                self.fit_d(xs, xt)
                logger.info("Complete Fit d")

                if bi % save_interval == 0 and bi > 0:
                    logger.info(save_checkpoint_msg.format(epoch, bi))
                    self.save_checkpoint(epoch, save_dir)


            if bi % save_interval != 0:
                logger.info(save_checkpoint_msg.format(epoch, bi))
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
        xs_idt = self.model.get_face_identity(xs, detach=True)
        y_idt = self.model.get_face_identity(y)
        xt_attr = self.model.get_face_attribute(xt)
        y_attr = self.model.get_face_attribute(y)
        d_output = self.model(None, None, mode=3, x_hat=y)

        loss, adv_loss, attr_loss, rec_loss, idt_loss = self.g_criterion(
            xt, y, xt_attr, y_attr, xs_idt, y_idt, d_output,reconstructed)
        
        logger.info("Save g loss")
        self.save_g_loss([adv_loss, attr_loss, rec_loss, idt_loss, loss])

        self.last_g_loss = loss.item()

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

        logger.info("Save d loss")
        self.save_d_loss([loss])

        self.last_d_loss = loss.item()

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
        name = "faceShifter_checkpoint_{}".format(epoch)
        checkpoint = {
            "epoch": epoch,
            "g_state_dict": self.model.get_g_state_dict(),
            "d_state_dict": self.model.get_d_state_dict(),
            "d_loss": self.last_d_loss,
            "g_loss": self.last_g_loss,
            "g_optim_state_dict": self.g_optimizer.state_dict(),
            "d_optim_state_dict": self.d_optimizer.state_dict()
        }

        utils.save_state_dict(checkpoint, name, save_dir, 
            remove_old=self.opt["checkpoint"]["remove_old"])
        self.opt["checkpoint"]["checkpoint_id"] = name


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


    def save_g_loss(self, loss):
        loss = [i.item() for i in loss]
        utils.save_loss(loss, "g", self.log_dir)


    def save_d_loss(self, loss):
        loss = [i.item() for i in loss]
        utils.save_loss(loss, "d", self.log_dir)