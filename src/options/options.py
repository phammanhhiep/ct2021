import sys, os
import argparse
import yaml
from datetime import datetime


#TODO: implement the class so that it work like a dictionary
class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--option_file', type=str, 
            help="Relative path to the option file")
        self.parser.add_argument('--experiment', type=str, 
            help="Name of experiment; default name include 2 parts, YYMMDD and \
            experiment count", 
            default=datetime.today().strftime("%Y%m%d") + "_0")
        
        args = self.gather_arguments()
        self.opt = self.gather_opt(args.option_file)
        self.opt["checkpoint"]["experiment"] = args.experiment


    def gather_arguments(self):
        args = self.parser.parse_args()
        return args


    def gather_opt(self, opt_file):
        with open(opt_file, "r") as fd:
            opt = yaml.load(fd, Loader=yaml.FullLoader)
        return opt


    def get_opt(self):
        return self.opt