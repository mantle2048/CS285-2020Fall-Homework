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
        --exp_name q1

fi


if [[ $1 == '3.2' ]]; then

    seeds=(0 1 2 3 4)
    for seed in ${seeds[@]}; do
        python $hw_dir \
            --env_name 'LunarLander-v3' \
            --exp_name 'q2_dqn_'$seed \
            --seed $seed
    done

    for seed in ${seeds[@]}; do
        python $hw_dir\
            --env_name 'LunarLander-v3' \
            --exp_name 'q2_doubledqn_'$seed \
            --double_q \
            --seed $seed
    done

fi


if [[ $1 == '3.3' ]]; then

    batch_sizes=(16 32 64 128)
    for batch_size in ${batch_sizes[@]}; do
        python $hw_dir\
            --env_name 'LunarLander-v3' \
            --exp_name 'q3_dqn_b_'$batch_size
    done

fi
