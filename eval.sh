#!/bin/bash

options="eval_options.yaml"

if [ ! -z "$1" ]; then
    options="$1"
fi

export PYTHONPATH="$HOME/projects/ct2021"
eval "$(conda shell.bash hook)"
conda activate cv
python src/eval.py --option_file "$options"