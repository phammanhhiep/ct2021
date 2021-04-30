import logging


from torch.utils import data


from common.utils import create_root_logger
from common.train_options import TrainOptions
from trainers.faceShifter_trainer import FaceShifterTrainer
from data.dataset import Dataset


#TODO: remove the need to hard-code a trainer, but select a trainer through options. Consider using trainer from pytorch_lightning
def train(logger):
    opt = TrainOptions()

    dataset = Dataset(opt["dataset"]["root_dir"], 
        opt["dataset"]["data_list_file"])
    
    dataloader = data.DataLoader(dataset, 
        batch_size=opt["batch_size"],
        shuffle=True,
        num_workers=opt["dataset"]["num_worker"],
        )

    trainer = FaceShifterTrainer(opt["trainer"]) #TODO: Review the argument

    for epoch in range(opt["epochs"]):
        for j, batch_data in enumerate(dataloader):
            trainer.fit(batch_data, j)
        trainer.update_optimizer(epoch)

        if epoch % opt["log"]["checkpoint_interval"]
            trainer.save_checkpoint(epoch, opt["log"]["checkpoint_dir"])

 
if __name__ == "__main__":
    train(create_root_logger())