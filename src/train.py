import logging


from torch.utils import data


from options.options import Options
from trainers.faceShifter_trainer import FaceShifterTrainer
from data.dataset import FFHQ, CelebAHQ


#TODO: remove the need to hard-code using if-else statement and specify datasets and trainers. Review SPADE project a hint. 
#TODO: Consider using trainer from pytorch_lightning
def train():
    opt = Options()
    dataset_name = opt["data"]["train"]
    trainer_name = opt["trainer"]["name"] 

    if dataset_name == "FFHQ":
        train_datasets = FFHQ(opt[dataset_name]["train"])
    elif dataset_name == "CelebAHQ":
        train_datasets = CelebAHQ(opt[dataset_name]["train"])
    
    dataloader = data.DataLoader(
        train_datasets, 
        batch_size=opt["data"]["batch_size"],
        shuffle=True,
        num_workers=opt["data"]["num_worker"],
        )

    if trainer_name == "FaceShifterTrainer"
        trainer = FaceShifterTrainer(opt[trainer_name])

    for epoch in range(opt["trainer"]["epochs"]):
        for j, batch_data in enumerate(dataloader):
            trainer.fit(batch_data, j)
        trainer.update_optimizer(epoch)

        if epoch % opt["checkpoint"]["save_interval"]
            trainer.save_checkpoint(epoch, opt["checkpoint"]["save_dir"])

 
if __name__ == "__main__":
    create_root_logger()
    train()