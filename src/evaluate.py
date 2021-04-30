import logging


from torch.utils import data
from torch import nn


from common.utils import create_root_logger
from common.test_options import TestOptions
from data.dataset import Dataset
from 
from src.models.networks.faceShifter_model import FaceShifterModel
from src.models.networks.head_pose_estimator import HopeNet


#TODO: The original research use CosFace (https://trello.com/c/jSfB5Dpz) to extract identity vector, but the current implementation use ArcFace instead
#TODO: compute fe_measure
#TODO: save evaluation result
def test(opt):
    logger = create_root_logger()
    dataset = Dataset(opt["dataset"]["root_dir"], 
        opt["dataset"]["data_list_file"])
    
    dataloader = data.DataLoader(dataset, 
        batch_size=opt["batch_size"],
        shuffle=True,
        num_workers=opt["dataset"]["num_worker"],
        )

    model = FaceShifterModel(opt["faceShifter_model"])
    model.load(opt["model_id"], opt["log"]["checkpoint_dir"])
    model.eval()

    idt_encoder = model.get_idt_encoder()
    generator = model.get_generator()

    hp_estimator = HopeNet()
    hp_estimator.load(opt["head_pose_estimator_id"], 
        opt["log"]["checkpoint_dir"])
    hp_estimator.eval()

    idt_measure = 0
    hp_measure = 0
    fe_measure = 0
    for j, batch_data in enumerate(dataloader):
        generated_imgs = generator(batch_data)
        source_images = batch_data[:, opt["source_img_idx"]]
        idt_measure += idt_retrieval(source_images, generated_imgs)
        hp_measure += head_pose_error(source_images, generated_imgs, 
            hp_estimator)

    logger.info((idt_measure, hp_measure, fe_measure))


def idt_retrieval(x, y, idt_encoder, dataset):
    """Summary
    
    Args:
        x (TYPE): real image
        y (TYPE): synthesized image
        idt_encoder (TYPE): Description
        dataset (TYPE): set of real image, excluding x
    """
    true_dist = torch.sum(nn.functional.cosine_similarity(
        idt_encoder(x), idt_encoder(y), dim=0))
    min_dist = true_dist
    for z in dataset:
        dist = torch.sum(nn.functional.cosine_similarity(
        idt_encoder(z), idt_encoder(y), dim=0))
        dist = min(dist, min_dist)
    return torch.sum(true_dist <= min_dist)


def head_pose_error(x, y, estimator):
    """Summary
    
    Args:
        x (TYPE): a batch of real images
        y (TYPE): a batch of corresponding synthesized images
        estimator (TYPE): head pose estimator, assumed to be a net
    
    Returns:
        TYPE: output of size (B, value)
    """

    x_pred = estimator(x)
    y_pred = estimator(y)
    x_d = estimator.to_degree(x_pred)
    y_d = estimator.to_degree(y_pred)
    return torch.sum(torch.nn.functional.pairwise_distance(x_d, y_d))


#TODO: implement later because no pretrained model to extract facial expression found
def facial_expression_error(x, y, estimator):
    """Summary
    
    Args:
        x (TYPE):  a batch of real images
        y (TYPE): a batch of corresponding synthesized images
        estimator (TYPE): facial expression estimator, assumed to be a net
    """
    return 0
