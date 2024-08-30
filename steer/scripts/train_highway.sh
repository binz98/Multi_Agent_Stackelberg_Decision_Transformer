#!/bin/sh
env="highway"
scenario_name="merge-multi-agent-v0"
reward="regionalR"
obs="Kinematics"
traffic_density=1
algo="steer"
seed=0
exp="steer"

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_highway.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario_name} \
--seed ${seed} --n_block 1 --n_training_threads 16 --n_rollout_threads 20 --num_mini_batch 1 --episode_length 200 --num_env_steps 5000000 \
--lr 5e-4 --ppo_epoch 5 --clip_param 0.05 --save_interval 100000 --use_value_active_masks --use_eval --add_state_token \
--encode_state --use_state_agent  --log_interval 5 --eval_interval 10 --traffic_density ${traffic_density} --reward_type ${reward} --obs_type ${obs}
