#!/bin/bash

DEFAULT_REQ="requirements.cpu.txt"
DEFAULT_CONDA_ENV="ct2021"

while getopts 'c:r:h' c; do
    case $c in
        c) myenv="$OPTARG" ;;
        r) requirements="$OPTARG" ;;
        h) usage ;;
    esac
done

if [ -z "$requirements" ]; then
    requirements="$DEFAULT_REQ"
fi

if [ -z "$myenv" ]; then
    myenv="$DEFAULT_CONDA_ENV"
fi

export CONDA_ALWAYS_YES="true"
conda config --append channels conda-forge
conda config --append channels pytorch
conda create -n "$myenv" --file "$requirements"
unset CONDA_ALWAYS_YES