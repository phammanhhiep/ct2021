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
    def __init__(self, root_dir, data_list_path, transform):
        """Represent a collection of data. 
        
        Args:
            root_dir (TYPE): directory that store data files.
            data_list_path (TYPE): the path to file that contains a list of 
            names of images in the dataset
            phase (str, optional): train or test.
        """
        self.root_dir = root_dir

        with open(data_list_path, "r") as fd:
            data_names = fd.readlines()
        self.data_paths = [os.path.join(root, name) for name in data_names]
        self.data_paths = np.random.permutation(self.data_paths)
        self.transform = transform


    def __len__(self):
        return len(self.data_paths)


    def __getitem__(self, index):
        """Return data from a sampled file
        
        Args:
            index (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        sample_path = self.img_files[index]
        sample = io.imread(sample_path)
        sample = self.transform(sample)
        return sample