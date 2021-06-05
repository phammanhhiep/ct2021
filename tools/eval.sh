#!/bin/bash

DEFAULT_ARTIFACTS="$HOME/projects/ct2021/artifacts"
DEFAULT_OPTIONS_FILE="$DEFAULT_ARTIFACTS/options/eval_options.yaml"
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

if [ -z "$option_file" ]; then
    option_file="$DEFAULT_OPTIONS_FILE"
fi

if [ -z "$app_dir" ]; then
    app_dir="$DEFAULT_APP_DIR"
fi

export PYTHONPATH="$app_dir"
eval "$(conda shell.bash hook)"
conda activate "$myenv"
python -W ignore "$app_dir/src/eval.py" --option_file "$option_file"