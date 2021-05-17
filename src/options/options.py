import sys, os
import argparse
import yaml
from datetime import datetime


#TODO: implement the class so that it work like a dictionary
class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument("--option_file", type=str, 
            help="Relative path to the option file")
        self.parser.add_argument("--checkpoint", type=str, 
            help="The name of a checkpoint to be loaded", default=None)
        
        args = self.gather_arguments()
        self.opt = self.gather_opt(args.option_file)
        if args.checkpoint is not None:
            self.opt["checkpoint"]["checkpoint_id"] = args.checkpoint


    def gather_arguments(self):
        args = self.parser.parse_args()
        return args


    def gather_opt(self, opt_file):
        with open(opt_file, "r") as fd:
            opt = yaml.load(fd, Loader=yaml.FullLoader)
        return opt


    def get_opt(self):
        return self.opt


class EvalOptions(TrainOptions):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--option_file', type=str, 
            help="Relative path to the option file")
        
        args = self.gather_arguments()
        self.opt = self.gather_opt(args.option_file)