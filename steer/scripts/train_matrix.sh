#!/bin/sh
env="nstep_matrix" # nstep_matrix
scenario="cooperation" # penalty0, penalty-100, different_order, cooperation, coordination
action_dim=2
algo="steer"
exp="steer_final"
running_max=5
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${running_max}"
for number in `seq ${running_max}`; #matrix 4 25 10000, nstep 10 100 200000
do
    echo "the ${number}-th running:"
    CUDA_VISIBLE_DEVICES=1 python train/train_matrix.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --seed ${number} \
    --scenario_name ${scenario}  --action_dim ${action_dim} --lr 5e-4 --critic_lr 5e-4  --n_training_threads 8 --n_rollout_threads 10 \
    --num_mini_batch 1 --episode_length 100 --num_env_steps 200000 --ppo_epoch 5 --n_block 1 --add_state_token --entropy_coef 0.05 \
    --running_id ${number} --use_eval --use_value_active_masks --eval_interval 1 --log_interval 1
done