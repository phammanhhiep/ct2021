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
        self.option_file = args.option_file
        self.gather_opt()


    def gather_arguments(self):
        args = self.parser.parse_args()
        return args


    def gather_opt(self):
        with open(self.option_file, "r") as fd:
            self.opt = yaml.load(fd, Loader=yaml.FullLoader)


    def get_opt(self):
        return self.opt


    def save(self):
        """Save the options to disk
        """
        with open(self.option_file, 'w') as fd:
            _ = yaml.dump(self.opt, fd)        


class EvalOptions(TrainOptions):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--option_file', type=str, 
            help="Relative path to the option file")
        
        args = self.gather_arguments()
        self.option_file = args.option_file
        self.gather_opt()