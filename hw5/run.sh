#!/usr/bin/env bash
hw_dir="cs285/scripts/run_hw5_expl.py"


if [[ $1 == '1.1' ]];then

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

if [[ $1 == '1.2' ]];then

    envs=('PointmassMedium-v0' 'PointmassHard-v0')
    seeds=(0 1 2)
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

if [[ $1 == '2.1' ]];then

    seeds=(0 1 2 3 4)

    for seed in ${seeds[@]};do

        python $hw_dir \
            --env_name 'PointmassMedium-v0' \
            --exp_name 'q2_med_dqn' \
            --use_rnd \
            --unsupervised_exploration \
            --offline_exploitation \
            --cql_alpha 0 \
            --exploit_rew_shift 1 \
            --exploit_rew_scale 100 \
            --seed $seed

        python $hw_dir \
            --env_name 'PointmassMedium-v0' \
            --exp_name 'q2_med_cql' \
            --use_rnd \
            --unsupervised_exploration \
            --offline_exploitation \
            --cql_alpha 0.1 \
            --exploit_rew_shift 1 \
            --exploit_rew_scale 100 \
            --seed $seed
    done

fi

if [[ $1 == '2.2' ]]; then

    seeds=(0 1 2)

    num_exploration_steps=(5000 10000 15000)

    for num_exploration_step in ${num_exploration_steps[@]}; do
        for seed in ${seeds[@]}; do
            python $hw_dir \
                --env_name 'PointmassMedium-v0' \
                --use_rnd \
                --num_exploration_steps $num_exploration_step \
                --offline_exploitation \
                --unsupervised_exploration  \
                --cql_alpha 0.1 \
                --exploit_rew_shift 1 \
                --exploit_rew_scale 100 \
                --exp_name 'q2_med_cql_numsteps_'$num_exploration_step \
                --seed $seed


            python $hw_dir \
                --env_name 'PointmassMedium-v0' \
                --use_rnd \
                --num_exploration_steps $num_exploration_step \
                --offline_exploitation \
                --unsupervised_exploration \
                --cql_alpha 0.0 \
                --exploit_rew_shift 1 \
                --exploit_rew_scale 100 \
                --exp_name 'q2_med_dqn_numsteps_'$num_exploration_step \
                --seed $seed
        done
    done

fi


if [[ $1 == '2.3' ]]; then

    cql_alphas=(0.02 0.5)
    seeds=(0 1 2)

    for cql_alpha in ${cql_alphas[@]}; do
        for seed in ${seeds[@]}; do
            python $hw_dir \
                --env_name 'PointmassMedium-v0' \
                --use_rnd \
                --unsupervised_exploration \
                --offline_exploitation \
                --cql_alpha $cql_alpha \
                --exp_name 'q2_alpha_'$cql_alpha \
                --seed $seed
        done
    done

fi

if [[ $1 == '3' ]]; then

    seeds=(0 1 2)

    for seed in ${seeds[@]}; do

        python $hw_dir \
            --env_name 'PointmassMedium-v0' \
            --use_rnd \
            --num_exploration_steps 20000 \
            --cql_alpha 0.0 \
            --exp_name 'q3_med_dqn' \
            --seed $seed

        python $hw_dir \
            --env_name 'PointmassMedium-v0' \
            --use_rnd \
            --num_exploration_steps 20000 \
            --cql_alpha 1.0 \
            --exp_name 'q3_med_cql' \
            --seed $seed

        python $hw_dir \
            --env_name 'PointmassHard-v0' \
            --use_rnd \
            --num_exploration_steps 20000 \
            --cql_alpha 0.0 \
            --exp_name 'q3_hard_dqn' \
            --seed $seed

        python $hw_dir \
            --env_name 'PointmassHard-v0' \
            --use_rnd \
            --num_exploration_steps 20000 \
            --cql_alpha 1.0 \
            --exp_name 'q3_hard_cql' \
            --seed $seed

    done

fi
