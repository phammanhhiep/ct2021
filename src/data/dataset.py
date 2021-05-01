import os
import random
import logging


import numpy as np
from skimage import Image, io
from skimage.color import gray2rgb
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class Dataset(data.Dataset):
    def __init__(self, data_list, transform):
        """It represents one or more collections of data. The implementation
        assumes only one dataset is passed at one time.   
        
        Args:
            data_lists (TYPE): path to the file that contains a list of names of
            file in the dataset
            transform (TYPE): Description
        """
        self.transform = transform
        with open(data_list, "r") as fd:
            self.data_paths = fd.readlines()
        self.data_paths = np.random.permutation(self.data_paths)


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
        sample = io.imread(sample_path)
        sample = self.transform(sample)
        return sample