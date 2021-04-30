import logging


import torch
from torch import nn


from models.faceShifter_model import FaceShifterModel
from networks.loss import AEINetLoss, GanLoss


logger = logging.getLogger(__name__)


class FaceShifterTrainer:
    def __init__(self, opt): 
        self.opt = opt
        self.model = FaceShifterModel(opt)
        self.create_criterion()
        self.create_optimizer()


    #TODO: review the training interval of the generator and discriminator
    def fit(self, data, batch_index):
        if batch_index % self.opt["D_step_per_G"] == 0:
            self.fit_g(data)
        self.fit_d(data)


    def fit_g(self, data):
        x, label = data
        y = self.model(data, 1)
        loss = self.g_criterion(y, label)
        self.g_optimizer.zero_grad()
        loss.backward()
        self.g_optimizer.step()


    def fit_d(self, data):
        x, label = data
        y = self.model(data, 1)
        loss = self.d_criterion(y, label)
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()        


    def create_optimizer(self):
        g_params = self.model.get_g_params()
        d_params = self.model.get_d_params()

        self.g_optimizer = torch.optim.Adam(g_params, 
            lr=self.opt["optim"]["lr"],
            beta=self.opt["optim"]["beta"])
        self.d_optimizer = torch.optim.Adam(d_params, 
            lr=self.opt["optim"]["lr"],
            beta=self.opt["optim"]["beta"])


    def create_criterion(self):
        self.g_criterion = AEINetLoss(opt["aei_net_loss"])
        self.d_criterion = GanLoss(opt["gan_loss"])


    def update_optimizer(self, epoch):
        """Update parameters of the optimizer.
        
        Args:
            epoch (TYPE): Description
        """

    def save_checkpoint(self, model_id, save_dir):
        """Summary
        
        Args:
            model_id (TYPE): Description
            save_dir (TYPE): Description
        """
        self.model.save(model_id, save_dir)


    def load_checkpoint(self, model_id, load_dir)
        self.model.load(model_id, load_dir)
