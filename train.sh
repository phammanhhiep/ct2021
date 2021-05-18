#!/bin/bash

get_lastest_checkpoint() {
    # Assume in the project directory
    epoch=0
    for name in $(ls "checkpoint/" | grep "checkpoint" | grep -v "old"); do
        new_epoch=$(echo $name | cut -d "." -f 1 | cut -d "_" -f 3)
        if (( $epoch < $new_epoch )); then
            epoch=$new_epoch
        fi
    done
    echo "faceShifter_checkpoint_$epoch"
}


export PYTHONPATH="$HOME/projects/ct2021"
eval "$(conda shell.bash hook)"
conda activate cv
cd $HOME/projects/ct2021/
python src/train.py --option_file "train_options.yaml" --checkpoint $(get_lastest_checkpoint)