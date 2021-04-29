from sys import stdout
import logging


def create_root_logger(level=logging.INFO):
    FORMAT = logging.Formatter(
        "%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    logger = logging.getLogger()
    handler = logging.StreamHandler(stdout)
    handler.setFormatter(FORMAT)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def save_net(net, label, save_dir):
    name = "{}.pth".format(epoch, label)
    save_path = os.path.join(save_dir, name)
    torch.save(net.state_dict(), save_path)


#TODO: the implementation assume model is load to a CPU
def load_net(net, label, load_dir):
    name = "{}.pth".format(epoch, label)
    load_path = os.path.join(load_dir, name)
    net.load_state_dict(torch.load(load_path), map_location=torch.device("cpu"))
    return net