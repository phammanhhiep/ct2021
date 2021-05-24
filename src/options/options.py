import sys, os
import argparse
import yaml
from datetime import datetime


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument("--option_file", type=str, 
            help="Relative path to the option file")
        self.parser.add_argument("--data_list", type=str, default=None)
        self.parser.add_argument("--data_root_dir", type=str, default=None) 


    def gather_arguments(self):
        self.args = self.parser.parse_args()
        self.option_file = self.args.option_file


    def gather_opt(self):
        with open(self.option_file, "r") as fd:
            self.opt = yaml.load(fd, Loader=yaml.FullLoader)

        if self.args.data_list and self.args.data_root_dir:
            dataset_name = "NoNameDataset"
            self.opt[dataset_name] = {
                "data_list": self.args.data_list,
                "root_dir": self.args.data_root_dir
            }
            self.opt["dataset"]["name"] = dataset_name


    def get_opt(self):
        return self.opt    


class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__(self)
        self.parser.add_argument("--checkpoint", type=str, 
            help="The name of a checkpoint to be loaded", default=None)
        self.gather_arguments()
        self.gather_opt2()


    def gather_opt2(self):
        self.gather_opt()
        if self.args.checkpoint:
            self.opt["checkpoint"]["continue"] = True
            self.opt["checkpoint"]["checkpoint_id"] = self.args.checkpoint


class EvalOptions(BaseOptions):
    def __init__(self):
        super().__init__(self)
        self.parser.add_argument("--model", type=str, 
            help="The name of a model to be evaluated", default=None)
        self.parser.add_argument("--model_root_dir", type=str, 
            help="Directory that contain the model", default=None)

        self.gather_arguments()
        self.gather_opt2()


    def gather_opt2(self):
        self.gather_opt()
        if self.args.model:
            self.opt["model"]["name"] = self.args.model
        if self.args.model_root_dir:
            self.opt["model"]["root_dir"] = self.args.root_dir