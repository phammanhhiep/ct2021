import logging
from datetime import datetime


import torch
from torch import nn


from src.models.faceShifter_model import FaceShifterModel
from src.models.networks.loss import AEINetLoss, MultiScaleGanLoss


logger = logging.getLogger(__name__)


class FaceShifterTrainer:
    def __init__(self, opt): 
        self.opt = opt
        self.last_epoch = 0
        self.trainer_opt = opt["FaceShifterTrainer"]
        self.model = FaceShifterModel()
        self.init_model()
        self.create_criterion()
        self.create_optimizer()


    #TODO: provide option to initialize model when training from scratch (though models provided by torchvision and pytoch are initialized automaitcally
    def init_model(self):
        """Initialize model or load the trained paramaters in the given epoch
        """
        self.model.load_pretrained_idt_encoder(
            self.opt["IdtEncoder"]["pretrained_model"])

        if self.opt["checkpoint"]["continue"]["status"]:
            self.load_checkpoint(
                self.opt["checkpoint"]["continue"]["name"], 
                self.opt["checkpoint"]["save_dir"])
            self.set_last_epoch()


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
                
                if bi % self.trainer_opt["d_step_per_g"] == 0:
                    self.fit_g(xs, xt, reconstructed)
                    logger.info("Fit g: epoch {} - batch {}".format(epoch, bi))
                
                self.fit_d(xs, xt)
                logger.info("Fit d: epoch {} - batch {}".format(epoch, bi))
            
            self.save_checkpoint(epoch, save_dir)
            logger.info("Save checkpoint: epoch {} - batch {}".format(epoch, bi))
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
        self.g_optimizer.zero_grad()
        loss.backward()
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


    def set_last_epoch(self):
        """Take into account the case in which a model is continued to be 
        trained. last_epoch should be positive only if in the case.
        
        Returns:
            TYPE: Description
        """
        self.last_epoch = int(
            self.opt["checkpoint"]["continue"]["name"].split("_")[1]) + 1


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
        date_str = datetime.today().strftime('%Y%m%d')
        name = "{}_{}".format(date_str, epoch)
        self.model.save(name, save_dir)


    def load_checkpoint(self, model_id, load_dir):
        """Summary
        
        Args:
            model_id (TYPE): Description
            load_dir (TYPE): Description
        """
        self.model.load(model_id, load_dir)