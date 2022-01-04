#!/usr/bin/env bash
hw_dir="cs285/scripts/run_hw5_expl.py"


if [[ $1 == '5.1' ]];then

    count=0
    envs=('PointmassEasy-v0' 'PointmassMedium-v0')
    seeds=(0 1 2)

    for env in ${envs[@]}; do
        count=$(($count+1))
        for seed in ${seeds[@]}; do

            python $hw_dir \
                --env_name $env \
                --use_rnd \
                --unsupervised_exploration \
                --exp_name 'q1_env'$count'_rnd' \
                --seed $seed

            python $hw_dir \
                --env_name $env \
                --unsupervised_exploration \
                --exp_name 'q1_env'$count'_random' \
                --seed $seed
        done

    done
fi

if [[ $1 == '5.2' ]];then

    # envs=('PointmassMedium-v0')
    envs=('PointmassHard-v0')
    seeds=(1 2)
        for seed in ${seeds[@]}; do

            python $hw_dir \
                --env_name 'PointmassMedium-v0' \
                --unsupervised_exploration \
                --use_dyn \
                --exp_name 'q1_med_alg' \
                --seed $seed

            python $hw_dir \
                --env_name 'PointmassHard-v0' \
                --unsupervised_exploration \
                --use_dyn \
                --exp_name 'q1_hard_alg' \
                --seed $seed

            python $hw_dir \
                --env_name 'PointmassMedium-v0' \
                --unsupervised_exploration \
                --use_rnd \
                --exp_name 'q1_med_rnd' \
                --seed $seed

            python $hw_dir \
                --env_name 'PointmassHard-v0' \
                --unsupervised_exploration \
                --use_rnd \
                --exp_name 'q1_hard_rnd' \
                --seed $seed

    done
fi
