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

if [[ $1 = "2.2" ]]; then
    batch_sizes=(100 500 1000)
    lrs=(5e-4 1e-3 5e-3)
    for  batch_size in ${batch_sizes[@]}; do
        for  lr in ${lrs[@]}; do
            python $hw_dir \
                --env_name 'InvertedPendulum-v2' \
                --ep_len 1000 \
                --discount 0.9 \
                --n_iter 100 \
                --n_layers 2 \
                --size 64 \
                --batch_size $batch_size \
                --learning_rate $lr \
                --reward_to_go \
                --exp_name 'q2_b_'$batch_size'_r_'$lr \
                --eval_batch_size 400 \
                --video_log_freq -1
        done
    done
fi

if [[ $1 == "2.3" ]]; then
    seeds=(0 1 2 3 4)
    for seed in ${seeds[@]}; do
        python $hw_dir \
            --env_name 'LunarLanderContinuous-v2' \
            --ep_len 1000 \
            --discount 0.99 \
            --n_iter 100 \
            --n_layers 2 \
            --size 64 \
            --batch_size 40000 \
            --learning_rate 5e-3 \
            --reward_to_go \
            --seed $seed \
            --nn_baseline \
            --exp_name 'q3_b-40000_r-0.005' \
            --video_log_freq -1
    done
fi

if [[ $1 == "2.4.1" ]]; then
    batch_sizes=(10000 30000 50000)
    lrs=(0.005 0.01 0.02)
    for batch_size in ${batch_sizes[@]}; do
        for lr in ${lrs[@]}; do
            python $hw_dir \
                --env 'HalfCheetah-v2' \
                --ep_len 150 \
                --discount 0.95 \
                --n_iter 100 \
                --n_layers 2 \
                --size 32 \
                --batch_size $batch_size \
                --learning_rate $lr \
                --reward_to_go \
                --nn_baseline \
                --exp_name 'q4_search_b_'$batch_size'_lr_'$lr'_rtg_nnbaseline' \
                --video_log_freq -1
        done
    done
fi

if [[ $1 == "2.4.2" ]]; then

    python $hw_dir \
        --env_name "HalfCheetah-v2" \
        --ep_len 150 \
        --discount 0.95 \
        --n_iter 100  \
        --n_layers 2  \
        --size 32  \
        --batch_size 30000  \
        --learning_rate 0.02 \
        --exp_name "q4_b-30000_lr-0.02"

    python $hw_dir \
        --env_name "HalfCheetah-v2" \
        --ep_len 150 \
        --discount 0.95 \
        --n_iter 100  \
        --n_layers 2  \
        --size 32  \
        --reward_to_go \
        --batch_size 30000  \
        --learning_rate 0.02 \
        --exp_name "q4_b-30000_lr-0.02_rtg"

    python $hw_dir \
        --env_name "HalfCheetah-v2" \
        --ep_len 150 \
        --discount 0.95 \
        --n_iter 100  \
        --n_layers 2  \
        --size 32  \
        --nn_baseline \
        --batch_size 30000  \
        --learning_rate 0.02 \
        --exp_name "q4_b-30000_lr-0.02_nnbaseline"

    python $hw_dir \
        --env_name "HalfCheetah-v2" \
        --ep_len 150 \
        --discount 0.95 \
        --n_iter 100  \
        --n_layers 2  \
        --size 32  \
        --nn_baseline \
        --reward_to_go \
        --batch_size 30000  \
        --learning_rate 0.02 \
        --exp_name "q4_b-30000_lr-0.02_rtg_nnbaseline"
fi

if [[ $1 = "2.5" ]]; then
    lambdas=(0.99)
    for lambda in ${lambdas[@]}; do
        python $hw_dir \
            --env_name 'Hopper-v2' \
            --ep_len 1000 \
            --discount 0.99 \
            --n_iter 300 \
            --n_layers 2  \
            --size 32 \
            --batch_size 2000 \
            --learning_rate 0.001 \
            --reward_to_go \
            --nn_baseline \
            --action_noise_std 0.5 \
            --gae_lambda $lambda \
            --exp_name 'q5_b_2000_r_0.001_lambda_'$lambda
    done

fi

if [[ $1 == "2.6" ]]; then
    seeds=(0 1 2 3 4)
    for seed in ${seeds[@]}; do
        python $hw_dir \
            --env_name 'LunarLanderContinuous-v2' \
            --ep_len 1000 \
            --discount 0.99 \
            --n_iter 100 \
            --n_layers 2 \
            --size 64 \
            --batch_size 40000 \
            --learning_rate 5e-3 \
            --reward_to_go \
            --seed $seed \
            --nn_baseline \
            --exp_name 'q6_b_40000_r_0.005' \
            --num_envs  10 \
            --video_log_freq -1
    done
fi

if [[ $1 == "2.7" ]]; then

    multi_steps=(1 2 3 4)
    for ms in ${multi_steps[@]}; do
        python $hw_dir \
            --env_name "HalfCheetah-v2" \
            --ep_len 150 \
            --discount 0.95 \
            --n_iter 100  \
            --n_layers 2  \
            --size 32  \
            --nn_baseline \
            --reward_to_go \
            --batch_size 30000  \
            --num_envs 15 \
            --multi_step $ms \
            --learning_rate 0.02 \
            --seed 2 \
            --exp_name "q7_b_30000_lr_0.02_ms_"$ms"_rtg_nnbaseline"
    done
fi
