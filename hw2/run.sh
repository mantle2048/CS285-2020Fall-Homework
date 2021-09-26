#! /usr/bin/env bash

# Run for Hw2

hw_dir="cs285/scripts/run_hw2.py"
# env_names=("HalfCheetah" "Ant" "Hopper" "Walker2d" "Humanoid")

if [[ $1 = "2.1" ]]; then

    env_names=("CartPole-v0")
    for env_name in ${env_names[@]}; do
        python $hw_dir \
            --env_name $env_name \
            --exp_name "q1_sb_no-rtg_dsa" \
            --n_iter 100 \
            --batch_size 1000 \
            --dont_standardize_advantages \
            --video_log_freq -1
    done


    for env_name in ${env_names[@]}; do
        python $hw_dir \
            --env_name $env_name \
            --exp_name "q1_sb_rtg_dsa" \
            --n_iter 100 \
            --batch_size 1000 \
            --dont_standardize_advantages \
            --reward_to_go \
            --video_log_freq -1
    done


    for env_name in ${env_names[@]}; do
        python $hw_dir \
            --env_name $env_name \
            --exp_name "q1_sb_rtg_na" \
            --n_iter 100 \
            --batch_size 1000 \
            --reward_to_go \
            --video_log_freq -1
    done


    for env_name in ${env_names[@]}; do
        python $hw_dir \
            --env_name $env_name \
            --exp_name "q1_lb_no-rtg_dsa" \
            --n_iter 100 \
            --batch_size 5000 \
            --dont_standardize_advantages \
            --video_log_freq -1
    done


    for env_name in ${env_names[@]}; do
        python $hw_dir \
            --env_name $env_name \
            --exp_name "q1_lb_rtg_dsa" \
            --n_iter 100 \
            --batch_size 5000 \
            --dont_standardize_advantages \
            --reward_to_go \
            --video_log_freq -1
    done


    for env_name in ${env_names[@]}; do
        python $hw_dir \
            --env_name $env_name \
            --exp_name "q1_lb_rtg_na" \
            --n_iter 100 \
            --batch_size 5000 \
            --reward_to_go \
            --video_log_freq -1
    done

fi
