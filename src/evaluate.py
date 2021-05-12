import os, logging
from datetime import datetime
import csv

from torch.utils import data
from torch import nn
import torch
import torchvision


from src.common.utils import create_root_logger
from src.options.options import EvalOptions
from data.dataset import Dataset
from src.models.faceShifter_model import FaceShifterModel
from src.models.networks.head_pose_estimator import HopeNet


#TODO: The original research use CosFace (https://trello.com/c/jSfB5Dpz) to extract identity vector, but the current implementation use ArcFace instead
#TODO: compute fe_measure
#TODO: save evaluation result
def evaluate(logger):
    """Summary
    
    Args:
        opt (TYPE): Description
        logger (TYPE): Description
    """
    opt = EvalOptions(); opt = opt.get_opt()

    model = FaceShifterModel()
    model.load(opt["model"]["name"], opt["model"]["load_dir"], load_d=False)
    model.eval()

    hp_estimator = HopeNet()
    hp_estimator.load(opt["head_pose_estimator"]["name"], 
        opt["head_pose_estimator"]["load_dir"])
    hp_estimator.eval()

    facial_expression_estimator = None #TODO: provide facial_expression_estimator

    data_list = opt["data"] 

    date_str = datetime.today().strftime('%Y%m%d')
    img_name = date_str + "_{}_{}.png"
    stat_pth = opt["statistics"].format(date_str)

    idt_measure = 0
    hp_measure = 0
    fe_measure = 0
    
    result = []
    idt_dist = []

    dataset = Dataset(data_list, return_name=True)
    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True)
    generated_count = len(dataset) / 2

    logger.debug("Evaluate dataset: {}".format(data_list))

    for j, batch_data in enumerate(dataloader):
        xs, xt = batch_data[0][:1], batch_data[0][1:]
        x_hat = model(xs, xt)
        
        idt_retrieval(xs, xt, x_hat, model, idt_dist)
        hp_measure += head_pose_error(xt, x_hat, hp_estimator)
        fe_measure += facial_expression_error(xt, x_hat, 
            facial_expression_estimator)

        result.append((x_hat, batch_data[1]))
        logger.debug("Generate image: #{}".format(j))

    idt_measure = len(idt_dist) / generated_count
    hp_measure = hp_measure / generated_count
    fe_measure = fe_measure / generated_count


    save_generated_images(result, img_name, opt["generated_image"]["save_dir"])
    save_evaluation_results(
        (idt_measure, hp_measure.item(), fe_measure.item()), stat_pth)

    logger.debug("Save generated images and evaluation results")


def idt_retrieval(xs, xt, y, model, idt_dist):
    """
    Compute idt distance between generated image and true images, and determine
    if that distance is smaller than between that generated image and the rest of
    real images. To avoid loading all real images at once, the implementation 
    stores the true distance and the identity of every generated images in 
    idt_dist to compare when real images are available. If there exist a real 
    image, whose distance with the generated image is less than the true distance, 
    the generated image is no longer consider in the next iteration. 

    Args:
        xs (TYPE): source image (identity image)
        xt (TYPE): target image
        y (TYPE): the corresponding synthesized image
        model (TYPE): Description x
        idt_dist (TYPE): a list of [idt, idt distance]
    
    Returns:
        TYPE: Description
    """

    compute_idt_dist = \
        lambda x, y: torch.sum(nn.functional.cosine_similarity(x, y, dim=1))

    xs_idt = model.get_face_identity(xs)
    xt_idt = model.get_face_identity(xt)
    y_idt = model.get_face_identity(y)
    true_dist = compute_idt_dist(y_idt, xs_idt)
 
    count = len(idt_dist)

    for i in range(count):
        z_idt, z_dist = idt_dist[i]
        if compute_idt_dist(z_idt, xs_idt) < z_dist or \
        compute_idt_dist(z_idt, xt_idt) < z_dist:
            idt_dist.pop(i)

    if compute_idt_dist(y_idt, xt_idt) >= true_dist:
        idt_dist.append([y_idt, true_dist])


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
    return torch.tensor(0)


def save_generated_images(imgs, img_name, save_dir):
    """a list of images
    
    Args:
        imgs (TYPE): a list of tensors, each of which representing an image
        img_name (TYPE): Description
        save_dir (TYPE): Description
    """

    n = len(imgs)
    for i in range(n):
        img = imgs[i]
        torchvision.utils.save_image(img[0], 
            os.path.join(save_dir, img_name.format("_".join(img[1]), i)))


def save_evaluation_results(stat, pth):
    """Summary
    
    Args:
        stat (TYPE): Description
        pth (TYPE): file to save result
    """

    with open(pth, mode="a") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(stat)


if __name__ == "__main__":
    logger = create_root_logger(level=logging.DEBUG)
    evaluate(logger)