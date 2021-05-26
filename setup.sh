#!/bin/bash

env_name="cv"
req_file="requirements.txt"

eval "$(conda shell.bash hook)"

conda config --append channels conda-forge
conda config --append channels pytorch
conda create --name "$env_name" --file "$req_file"