import os
import random
import logging
from PIL import Image


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

        with open(data_list, "r") as fd:
            self.data_names = fd.readlines()

        self.data_names = [p.replace("\n", "") for p in self.data_names]
        self.data_names = np.random.permutation(self.data_names)

        if transforms is None:
            self.transforms = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor()
                ])


    def __len__(self):
        return len(self.data_names)


    def __getitem__(self, index):
        """Return data from a sampled file
        
        Args:
            index (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        sample_name = self.data_names[index]
        sample = Image.open(os.path.join(self.root_dir, sample_name))
        sample = self.transforms(sample)
        if self.return_name:
            sample = [sample, os.path.basename(sample_name).split(".")[0]]
        return sample