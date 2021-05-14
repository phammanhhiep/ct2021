from sys import stdout
import os
import logging
from datetime import datetime

import torch


def create_root_logger(level=logging.DEBUG, file_name=None):
    FORMAT = logging.Formatter(
        "%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    logger = logging.getLogger()
    if file_name is not None:
        timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')
        handler = logging.FileHandler("log/{}_{}.log".format(file_name, timestamp))
    else:
        handler = logging.StreamHandler(stdout)
    
    handler.setFormatter(FORMAT)
    logger.addHandler(handler)
    logger.setLevel(level)

    return logger


def save_net(net, label, save_dir):
    name = "{}.pth".format(label)
    save_path = os.path.join(save_dir, name)
    torch.save(net.state_dict(), save_path)


#TODO: the implementation assume model is load to a CPU
def load_net(net, label, load_dir):
    name = "{}.pth".format(label)
    load_path = os.path.join(load_dir, name)
    net.load_state_dict(torch.load(load_path, map_location="cpu"))