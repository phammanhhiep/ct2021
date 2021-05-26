#!/bin/bash

env_name="cv"
req_file="requirements.txt"

eval "$(conda shell.bash hook)"


export CONDA_ALWAYS_YES="true"
conda config --append channels conda-forge
conda config --append channels pytorch
conda create --name "$env_name" --file "$req_file"
unset CONDA_ALWAYS_YES