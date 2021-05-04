import logging


import torch
from torch import nn


from models.faceShifter_model import FaceShifterModel
from networks.loss import AEINetLoss, GanLoss


logger = logging.getLogger(__name__)


class FaceShifterTrainer:
    def __init__(self, opt): 
        self.opt = opt
        self.trainer_opt = opt["FaceShifterTrainer"]
        self.model = FaceShifterModel(opt)
        self.init_model()
        self.create_criterion()
        self.create_optimizer()


    #TODO: provide option to initialize model when training from scratch (though models provided by torchvision and pytoch are initialized automaitcally
    def init_model(self):
        """Initialize model or load the trained paramaters in the given epoch
        """
        if self.trainer_opt["continue"]["status"]:
            self.model.load(
                self.trainer_opt["continue"]["epoch"], 
                self.opt["checkpoint"]["save_dir"])


    #TODO: review the input of discriminator and generator
    def fit(self, data, batch_index):
        """Summary
        
        Args:
            data (TYPE): Description
            batch_index (TYPE): Description
        """
        if batch_index % self.trainer_opt["d_step_per_g"] == 0:
            self.fit_g(data)
        self.fit_d(data)


    def fit_g(self, data):
        """Summary
        
        Args:
            data (TYPE): [source images, target images]
        """
        xs, xt = data
        y = self.model(data, 1)
        xs_idt = self.model.get_face_identity(xs)
        y_idt = self.model.get_face_identity(y)
        xt_attr = self.model.get_face_attribute(xt)
        y_attr = self.model.get_face_attribute(y)

        loss = self.g_criterion(xt, y, xt_attr, y_attr, xs_idt, y_idt, 
            self.d_output)
        self.g_optimizer.zero_grad()
        loss.backward()
        self.g_optimizer.step()


    #TODO: the loss is implemented in different from that of project SPADE. Consider if the difference could change performance of the trainer
    def fit_d(self, data):
        """Fit the discriminator using both real and generated data
        
        Args:
            data (TYPE): a batch of either real or synthesized images
        """
        real_pred, generated_pred = self.model(data, 2)
        real_loss = self.d_criterion(real_pred, True)
        generated_loss = self.d_criterion(generated_pred, False)
        loss = real_loss + generated_loss
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()        


    def create_optimizer(self):
        g_params = self.model.get_g_params()
        d_params = self.model.get_d_params()

        self.g_optimizer = torch.optim.Adam(g_params, 
            lr=self.opt["trainer"]["optim"]["lr"],
            beta=self.opt["trainer"]["optim"]["beta"])
        self.d_optimizer = torch.optim.Adam(d_params, 
            lr=self.opt["trainer"]["optim"]["lr"],
            beta=self.opt["trainer"]["optim"]["beta"])


    def create_criterion(self):
        self.g_criterion = AEINetLoss(self.opt)
        self.d_criterion = GanLoss(self.opt)


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


    def load_checkpoint(self, model_id, load_dir):
        """Summary
        
        Args:
            model_id (TYPE): Description
            load_dir (TYPE): Description
        """
        self.model.load(model_id, load_dir)
