import logging


from torch.utils import data
import torch


from src.options.options import TrainOptions
from src.trainers.faceShifter_trainer import FaceShifterTrainer
from src.data.dataset import Dataset


#TODO: explicit pass source and target image to trainer
#TODO: remove the need to hard-code using if-else statement and specify datasets and trainers. Review SPADE project a hint. 
#TODO: Consider using trainer from pytorch_lightning
#TODO: Provide option to save model within each epoch
def train(opt, logger):
    trainer_name = opt["trainer"]["name"]
    dataset_name = opt["dataset"]["train"]
    
    train_dataset = Dataset(opt[dataset_name]["root_dir"], 
        opt[dataset_name]["train"])
    
    dataloader = data.DataLoader(
        train_dataset, 
        batch_size=opt["dataset"]["batch_size"],
        shuffle=True,
        num_workers=opt["dataset"]["num_worker"],
        )

    if trainer_name == "FaceShifterTrainer":
        trainer = FaceShifterTrainer(opt)

    trainer.fit(dataloader)


if __name__ == "__main__":
    from src.common import utils
    opt = TrainOptions()
    logger = utils.create_root_logger(level=opt.get_opt()["log"]["level"], 
        file_name=opt.get_opt()["log"]["file_name"])
    try:
        train(opt, logger)
    except Exception as e:
        logger.error(str(e))