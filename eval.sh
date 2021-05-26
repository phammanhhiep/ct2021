#!/bin/bash

export PYTHONPATH="$HOME/projects/ct2021"
eval "$(conda shell.bash hook)"
conda activate cv
python src/eval.py --option_file "eval_options.yaml"