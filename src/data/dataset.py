import os
import random
import logging
from PIL import Image


import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class Dataset(data.Dataset):
    def __init__(self, data_list, transforms=None):
        """Each instance represent a dataset.   
        
        Args:
            data_lists (TYPE): path to the file that contains a list of file in
            the dataset
            transform (TYPE): a composite of transformations that are applied to
            every data point in the dataset
        """
        with open(data_list, "r") as fd:
            self.data_paths = fd.readlines()
        self.data_paths = np.random.permutation(self.data_paths)
        if transforms is None:
            self.transforms = T.Compose([
                T.Resize(256),
                T.ToTensor()
                ])


    def __len__(self):
        count = 0
        for d in self.data_paths:
            count += len(d)
        return count


    def __getitem__(self, index):
        """Return data from a sampled file
        
        Args:
            index (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        sample_path = self.data_paths[index]
        sample = Image.open(sample_path)
        sample = self.transform(sample)
        return sample