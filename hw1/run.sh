#! /usr/bin/env bash

# Run for Hw1

hw_dir="cs285/scripts/run_hw1.py"
expert_dir="cs285/policies/experts/"
expert_data="cs285/expert_data/expert_data_"


if [[ $1 = "1.2" ]]; then

    env_names=("HalfCheetah" "Ant" "Hopper" "Walker2d" "Humanoid")
    for env_name in ${env_names[@]}; do
        python $hw_dir \
            --expert_policy_file $expert_dir$env_name".pkl" \
            --env_name $env_name"-v2" \
            --exp_name "bc-"$env_name"_train-step_1000_n-iter_1" \
            --n_iter 1 \
            --eval_batch_size 10000 \
            --expert_data $expert_data$env_name"-v2.pkl" \
            --video_log_freq -1
    done

fi

if [[ $1 = "1.3" ]]; then

    env_name="Ant"
    for train_step in {0..5000..500}; do
        python $hw_dir \
            --expert_policy_file $expert_dir$env_name".pkl" \
            --env_name $env_name"-v2" \
            --exp_name "bc-"$env_name"_train-step_"$train_step"_n-iter_1" \
            --n_iter 1 \
            --eval_batch_size 10000 \
            --num_agent_train_steps_per_iter $train_step \
            --expert_data $expert_data$env_name"-v2.pkl" \
            --video_log_freq -1
    done

fi


if [[ $1 = "2" ]]; then

    env_name="Humanoid"
    n_iter="50"
    train_step=5000
    # env_names=("HalfCheetah" "Ant" "Hopper" "Walker2d" "Humanoid")
    python $hw_dir \
        --expert_policy_file $expert_dir$env_name".pkl" \
        --env_name $env_name"-v2" \
        --exp_name "dagger-"$env_name"_train-step_"$train_step"_n-iter_"$n_iter \
        --n_iter $n_iter \
        --do_dagger \
        --eval_batch_size 10000 \
        --learning_rate 5e-3 \
        --num_agent_train_steps_per_iter $train_step \
        --expert_data $expert_data$env_name"-v2.pkl" \
        --video_log_freq -1

fi
