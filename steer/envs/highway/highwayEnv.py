import sys
sys.path.append("/home/zb/Project/STEER/steer/envs/highway")

import numpy as np
import highway_env
import gym

import warnings
warnings.filterwarnings("ignore")

class Highway_Env(object):
    """
    # 环境中的智能体
    """
    def __init__(self, args, i):
        self.env = gym.make(args.scenario_name)
        
        self.env.seed = i
        self.env.unwrapped.seed = i
        self.env.config['simulation_frequency'] = 15
        self.env.config['duration'] = 20
        self.env.config['policy_frequency'] = 5
        self.env.config['COLLISION_REWARD'] = 200
        self.env.config['HIGH_SPEED_REWARD'] = 1
        self.env.config['HEADWAY_COST'] = 4
        self.env.config['HEADWAY_TIME'] = 1.2
        self.env.config['MERGING_LANE_COST'] = 4
        self.env.config['traffic_density'] = args.traffic_density
        self.env.config['action_masking'] = False
        self.env.config["observation"]["type"] = "MultiAgentObservation"
        self.env.config["observation"]["observation_config"]['type']=args.obs_type

        # ["observation_config"]
        self.traffic_density = args.traffic_density
        self.reward_type = args.reward_type  # "global_R"       # "regionalR"
        self.env_state, _ = self.env.reset(num_CAV=self.traffic_density+1)

        self.agent_num = len(self.env.controlled_vehicles)
        self.obs_dim = self.env_state.shape[1]            # self.env.n_s 
        self.action_dim = self.env.n_a
        # self.obs_dim = self.env.observation_space
        
    def reset(self):
        state, _ = self.env.reset(num_CAV=self.traffic_density+1)
        return state

    def step(self, actions):
        next_state, global_reward, done, info = self.env.step(actions.flatten().tolist())

        if self.reward_type == "regionalR":
            reward = info["regional_rewards"]
        elif self.reward_type == "global_R":
            reward = [global_reward] * self.agent_num

        done = np.array([done for _ in range(self.agent_num)])

        if done[0] == True:
            self.env.reset(num_CAV=self.traffic_density+1)

        return [next_state, reward, done, info]

if __name__ == '__main__':
    # print(Highway_Env.get_game_list())
    game = Highway_Env('merge-v0')
    game.reset()
    print(game.step(np.eye(2,5)))