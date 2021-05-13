import os, logging
from datetime import datetime
import csv

from torch.utils import data
from torch import nn
import torch
import torchvision


from src.options.options import EvalOptions
from data.dataset import Dataset
from src.models.faceShifter_model import FaceShifterModel
from src.models.networks.head_pose_estimator import HopeNet


#TODO: The original research use CosFace (https://trello.com/c/jSfB5Dpz) to extract identity vector, but the current implementation use ArcFace instead
#TODO: compute fe_measure
#TODO: save evaluation result
def evaluate(opt, logger):
    """Summary
    
    Args:
        opt (TYPE): Description
        logger (TYPE): Description
    """
    model = FaceShifterModel()
    model.load(opt["model"]["name"], opt["model"]["load_dir"], load_d=False)
    model.eval()

    hp_estimator = HopeNet()
    hp_estimator.load(opt["head_pose_estimator"]["name"], 
        opt["head_pose_estimator"]["load_dir"])
    hp_estimator.eval()

    facial_expression_estimator = None #TODO: provide facial_expression_estimator

    data_list = opt["dataset"]["evaluation"]
    data_root_dir = opt["dataset"]["root_dir"]
    batch_size = opt["dataset"]["batch_size"]
    num_worker = opt["dataset"]["num_worker"]

    date_str = datetime.today().strftime('%Y%m%d')
    img_name = date_str + "_{}_{}.png"
    stat_pth = opt["statistics"].format(date_str)

    idt_measure = 0
    hp_measure = 0
    fe_measure = 0
    
    result = []
    idt_dist = []

    dataset = Dataset(data_root_dir, data_list, return_name=True)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, 
        num_workers=num_worker, shuffle=False)
    generated_count = len(dataset)

    for j, batch_data in enumerate(dataloader):
        logger.info("Generate batch of images: #{}".format(j))
        xs, xt, reconstructed, xs_names, xt_names = batch_data
        x_hat = model(xs, xt)
        
        idt_retrieval(xs, xt, x_hat, model, idt_dist)
        hp_measure += head_pose_error(xt, x_hat, hp_estimator)
        fe_measure += facial_expression_error(xt, x_hat, 
            facial_expression_estimator)

        logger.info("Save {} generated images".format(x_hat.size()[0]))
        save_generated_images((x_hat, xs_names, xt_names), img_name, j,
            opt["generated_image"]["save_dir"])

    idt_measure = len(idt_dist) / generated_count
    hp_measure = hp_measure / generated_count
    fe_measure = fe_measure / generated_count


    logger.info("Save stat results")
    save_evaluation_results(
        (idt_measure, hp_measure.item(), fe_measure.item()), stat_pth)


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


def save_generated_images(generated_result, img_name, batch_idx, save_dir):
    """a list of images
    
    Args:
        generated_result (TYPE): a list of tensors representing individual image,
        and related source and target image names
        img_name (TYPE): name template to create names for generated images
        save_dir (TYPE): Description
    """
    images, xs_names, xt_names = generated_result
    for img, xs, xt in zip(images, xs_names, xt_names):
        torchvision.utils.save_image(img, os.path.join(
                save_dir, img_name.format("_".join((xs,xt)), batch_idx)))


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
    from src.common import utils
    opt = EvalOptions(); opt = opt.get_opt()
    logger = utils.create_root_logger(level=opt["log"]["level"], 
        file_name=opt["log"]["file_name"])
    try:
        evaluate(opt, logger)
    except Exception as e:
        logger.error(str(e))