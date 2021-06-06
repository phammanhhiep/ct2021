#!/bin/bash

DEFAULT_CONDA_ENV="ct2021"
DEFAULT_APP_DIR="$HOME/projects/ct2021"

usage() {
    echo "Not implement yet"
}

while getopts 'd:e:o:h' c; do
    case $c in
        d) app_dir="$OPTARG" ;;
        e) myenv="$OPTARG" ;;
        o) option_file="$OPTARG" ;;
        h) usage ;;
    esac
done

if [ -z "$myenv" ]; then
    myenv="$DEFAULT_CONDA_ENV"
fi

if [ -z "$app_dir" ]; then
    app_dir="$DEFAULT_APP_DIR"
fi

if [ -z "$option_file" ]; then
    option_file="$app_dir/artifacts/options/train_options.yaml"
fi


export PYTHONPATH="$app_dir"
eval "$(conda shell.bash hook)"
if [ "$myenv" != "base" ]; then
    conda activate "$myenv"
fi
cd "$app_dir"
python -W ignore "src/train.py" --option_file "$option_file"