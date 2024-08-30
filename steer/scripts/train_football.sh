#!/bin/sh
env="football"
scenario="academy_3_vs_1_with_keeper"
# academy_pass_and_shoot_with_keeper
# academy_3_vs_1_with_keeper
# academy_counterattack_easy
n_agent=3
num_env_steps=5000000
algo="steer"
exp="steer"
seed=1

CUDA_VISIBLE_DEVICES=1 python train/train_football.py --seed ${seed} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--scenario ${scenario} --n_agent ${n_agent} --lr 7e-4 --entropy_coef 0.01 --max_grad_norm 0.5 --eval_episodes 32 --n_training_threads 16 \
--n_rollout_threads 10 --num_mini_batch 1 --episode_length 400 --eval_interval 25 --num_env_steps 4000000 --ppo_epoch 15 \
--clip_param 0.2 --use_eval --use_value_active_masks --use_policy_active_masks --n_block 1 --add_state_token
