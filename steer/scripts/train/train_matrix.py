#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
sys.path.append("../../")
from config import get_config
from envs.mpe.MPE_env import MPEEnv
from envs.matrix_game.env_wrapper import SubprocVecEnv, DummyVecEnv
from runner.shared.matrix_runner import MGRunner as Runner

"""Train script for MPEs."""

def make_train_env(all_args):
    return SubprocVecEnv(all_args)

def make_eval_env(all_args):
    return DummyVecEnv(all_args)


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='penalty', help="Which scenario to run on")
    parser.add_argument("--action_dim", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    # main(sys.argv[1:])
    if len(sys.argv[1:]) == 0:
        argv = ['--env_name', 'nstep_matrix', '--algorithm_name', 'steer_state_v3_gru', '--lr', '5e-4', '--critic_lr', '5e-3', \
            '--experiment_name', 'debug', '--scenario_name', 'cooperation', '--action_dim', '2', \
            '--seed', '1', '--n_training_threads', '8', '--n_rollout_threads','10', \
            '--num_mini_batch', '1', '--episode_length', '100', '--num_env_steps', '200000','--ppo_epoch', '5', \
            '--use_eval', '--use_value_active_masks', '--running_id', '1', \
            '--eval_interval', '1', '--log_interval', '1'] # , '--share_policy'
    else:
        argv = sys.argv[1:]
        print(argv)
    main(argv)
# argv = ['--env_name', 'matrix', '--algorithm_name', 'steer_state_v9', '--lr', '5e-4', '--critic_lr', '5e-3', \
#             '--experiment_name', 'debug', '--scenario_name', '10step_matrix', \
#             '--seed', '1', '--n_training_threads', '8', '--n_rollout_threads','4', \
#             '--num_mini_batch', '1', '--episode_length', '10', '--num_env_steps', '10000','--ppo_epoch', '5', \
#             '--use_eval', '--use_value_active_masks', '--running_id', '1', \
#             '--eval_interval', '1', '--log_interval', '1'] # , '--share_policy'