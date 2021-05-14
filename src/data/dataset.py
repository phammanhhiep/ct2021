import os
import random
import logging
from PIL import Image
import csv


import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T


class Dataset(data.Dataset):
    def __init__(self, root_dir, data_list, transforms=None, return_name=False):
        """Each instance represent a dataset.   
        
        Args:
            root_dir (TYPE): Description
            data_list (TYPE): the file that contains a list of file in the dataset
            transforms (None, optional): a (pytorch) composite of transformations
            return_name (bool, optional): whether to return file name together
            with the corresponding data point
        """
        self.return_name = return_name
        self.root_dir = root_dir

        with open(data_list, newline="") as fd:
            reader = csv.reader(fd, delimiter=",")
            self.data_names = list(reader)

        self.data_names = np.random.permutation(self.data_names)

        if transforms is None:
            self.transforms = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor()
                ])


    def __len__(self):
        return len(self.data_names)


    def __getitem__(self, index):
        """Return a data point from a sampled file
        
        Args:
            index (TYPE): Description
        
        Returns:
            list: [source, target, reconstructed, [source name, target name]]
        """
        source_name, target_name, reconstructed = self.data_names[index]
        reconstructed = torch.tensor([[[int(reconstructed)]]])
        sample_source = Image.open(os.path.join(self.root_dir, source_name))
        sample_target = Image.open(os.path.join(self.root_dir, target_name))
        sample_source = self.transforms(sample_source)
        sample_target = self.transforms(sample_target)

        sample = [sample_source, sample_target, reconstructed]

        if self.return_name:
            sn = os.path.basename(source_name).split(".")[0]
            if reconstructed[0][0][0] == 1:
                tn = sn
            else:
                tn = os.path.basename(target_name).split(".")[0]
            sample += [sn, tn]
        return sample