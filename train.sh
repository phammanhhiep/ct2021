#!/bin/bash

export PYTHONPATH="$HOME/projects/ct2021"
eval "$(conda shell.bash hook)"
conda activate cv
python -W "ignore" src/train.py --option_file "train_options.yaml" 