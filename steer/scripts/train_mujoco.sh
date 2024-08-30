#!/bin/sh
env="mujoco"
scenario="HalfCheetah-v2"
agent_conf="6x1"
agent_obsk=1
faulty_node=-1
eval_faulty_node="-1"
algo="steer"
n_block=2
seed_max=1
exp="steer"
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for number in `seq ${seed_max}`;
do
    echo "the ${number}-th running:"
    CUDA_VISIBLE_DEVICES=1 python train/train_mujoco.py --seed ${number} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} \
    --agent_conf ${agent_conf} --agent_obsk ${agent_obsk} --faulty_node ${faulty_node} --eval_faulty_node ${eval_faulty_node} --n_block ${n_block} --critic_lr 5e-5 --lr 5e-5 \
    --entropy_coef 0.001 --max_grad_norm 0.5 --eval_episodes 5 --n_training_threads 16 --n_rollout_threads 40 --num_mini_batch 40 --episode_length 100 \
    --eval_interval 25 --num_env_steps 10000000 --ppo_epoch 10 --clip_param 0.05 --use_eval --add_center_xy --use_state_agent --use_value_active_masks \
    --use_policy_active_masks
done
