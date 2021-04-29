import logging


from torch.utils import data
from torch import nn


from common.utils import create_root_logger
from common.test_options import TestOptions
from data.dataset import Dataset
from src.models.networks.encoder import IdtEncoder
from src.models.networks.head_pose_estimator import HopeNet


def test(): pass


#TODO: The original research use CosFace (https://trello.com/c/jSfB5Dpz) to extract identity vector, but the current implementation use ArcFace instead 
def idt_retrieval(x, y, idt_encoder, dataset):
    """Summary
    
    Args:
        x (TYPE): real face
        y (TYPE): synthesized face
        idt_encoder (TYPE): Description
        dataset (TYPE): set of real faces from which we look for the 
        corresponding real face for the synthesized face
    """
    true_dist = torch.sum(nn.functional.cosine_similarity(
        idt_encoder(x), idt_encoder(y), dim=0))
    min_dist = true_dist
    for z in dataset:
        dist = torch.sum(nn.functional.cosine_similarity(
        idt_encoder(x), idt_encoder(y), dim=0))
        dist = min(dist, min_dist)
    return true_dist <= min_dist


def head_pose_error(x, y, head_pose_estimator):
    """Summary
    
    Args:
        x (TYPE): a batch of real images
        y (TYPE): a batch of corresponding synthesized images
        head_pose_estimator (TYPE): Description
    
    Returns:
        TYPE: output of size (B, value)
    """

    x_pred = head_pose_estimator(x)
    y_pred = head_pose_estimator(y)
    x_d = head_pose_estimator.to_degree(x_pred)
    y_d = head_pose_estimator.to_degree(y_pred)
    return torch.nn.functional.pairwise_distance(x_d, y_d)


#TODO: implement later because no pretrained model to extract facial expression found
def facial_expression_error(): pass