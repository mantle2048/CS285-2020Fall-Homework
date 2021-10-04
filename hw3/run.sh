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


ac_dir="cs285/scripts/run_hw3_actor_critic.py"
if [[ $1 == '3.4' ]]; then
    ntus=(1 100 1 10)
    ngsptus=(1 1 100 10)
    length=${#ntus[@]}
    for ((i=0; i<$length; i++)); do
        python $ac_dir  \
            --env_name 'CartPole-v0' \
            --n_iter 100  \
            --batch_size 1000 \
            --exp_name 'q4_ac_'${ntus[$i]}'_'${ngsptus[$i]} \
            --scalar_log_freq 1 \
            -ntu ${ntus[$i]} \
            -ngsptu ${ngsptus[$i]}
    done
fi


if [[ $1 == '3.5' ]]; then
    python $ac_dir  \
        --env_name 'InvertedPendulum-v2' \
        --n_iter 100  \
        --ep_len 1000 \
        --discount 0.95 \
        --n_layers 2 \
        --size 64 \
        --batch_size 5000 \
        --learning_rate 0.01 \
        --exp_name 'q5_ac_10_10' \
        --scalar_log_freq 1 \
        -ntu 10 \
        -ngsptu 10

    python $ac_dir  \
        --env_name 'HalfCheetah-v2' \
        --n_iter 150  \
        --ep_len 150 \
        --discount 0.90 \
        --scalar_log_freq 1 \
        --n_layers 2 \
        --size 32 \
        --batch_size 30000 \
        --learning_rate 0.02 \
        --eval_batch_size 1500 \
        --exp_name 'q5_ac_10_10' \
        -ntu 10 \
        -ngsptu 10
fi
