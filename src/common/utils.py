from sys import stdout
import os
import logging
from datetime import datetime
import argparse
import signal
import time


import torch


logger = logging.getLogger(__name__)


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
def load_net(net, label, load_dir, device="cpu"):
    name = "{}.pth".format(label)
    load_path = os.path.join(load_dir, name)
    net.load_state_dict(torch.load(load_path, map_location=device))


def save_state_dict(state, name, save_dir, remove_old=True):
    """Check if there exists file with the same name first, rename the old file,
    save the file, and optionally remove the old file
    
    Args:
        state (TYPE): Description
        name (TYPE): Description
        save_dir (TYPE): Description
    """
    save_path = os.path.join(save_dir, name + ".tar")
    old_path = save_path + ".old"
    if os.path.exists(save_path):
        os.rename(save_path, old_path)

    torch.save(state, save_path)

    while not os.path.exists(save_path):
        time.sleep(2) # wait until the state is saved to the disk

    if remove_old and os.path.exists(old_path):
        os.remove(old_path)


def load_state_dict(name, load_dir, device="cpu"):
    """Summary
    
    Args:
        name (TYPE): Description
        load_dir (TYPE): Description
        device (str, optional): Description
    
    Returns:
        TYPE: Description
    """
    pth = os.path.join(load_dir, name)
    state = None
    try:
        state = torch.load(pth, map_location=device)
    except FileNotFoundError as e:
        logger.error(str(e))
    else:
        return state


def extract_model_from_checkpoint(pth, save_dir, device="cpu"):
    """Extract the state dict of the generator from checkpoint, and save to disk
    
    Args:
        pth (TYPE): Description
    """
    checkpoint = torch.load(pth, map_location=device)
    model_name = os.path.basename(pth).split(".")[0] + ".pth"
    model = checkpoint["g_state_dict"]
    torch.save(model, os.path.join(save_dir, model_name))


class KillSignalHandler:
    received = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit)
        signal.signal(signal.SIGTERM, self.exit)


    def exit(self, x, y):
        self.received = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--extract_model', type=str, 
        help="Relative path to checkpoint")

    args = parser.parse_args()

    if args.extract_model:
        checkpoint = args.extract_model
        extract_model_from_checkpoint(checkpoint, "experiments/models/")