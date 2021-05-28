#!/bin/bash

OPTIONS_FILE="eval_options.yaml"
CONDA_ENV="ct2021"

usage() {
    echo "Not implement yet"
}


while getopts 'e:o:h' c; do
    case $c in
        e) CONDA_ENV="$OPTARG" ;;
        o) OPTIONS_FILE="$OPTARG" ;;
        h) usage ;;
    esac
done


export PYTHONPATH="$HOME/projects/ct2021"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
python src/eval.py --option_file "$OPTIONS_FILE"


