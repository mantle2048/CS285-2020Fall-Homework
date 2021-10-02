#! /usr/bin/env bash

# Run for Hw3

hw_dir="cs285/scripts/run_hw3_dqn.py"


if [[ $1 == '3.0' ]]; then

    python $hw_dir\
        --env_name 'LunarLander-v3' \
        --exp_name q1_lander

fi

if [[ $1 == '3.1' ]]; then

    python $hw_dir\
        --env_name 'MsPacman-v0' \
        --exp_name q1_MsPacman

fi
