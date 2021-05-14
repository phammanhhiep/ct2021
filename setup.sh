#!/bin/bash

env_name="cv"
req_file="requirements.txt"

eval "$(conda shell.bash hook)"
conda create --name "$env_name" --file "$req_file"