import numpy as np
import sys
# sys.path.append("/home/zb/Project/Multi-Agent-Transformer-main/mat/envs/")
from gym.spaces import Box, Discrete
from abc import ABC, abstractmethod

class BaseGame(ABC):
    """
    Base Interface for Environment/Game.
    """

    @abstractmethod
    def step(self, actions):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self):
        pass

class EnvironmentNotFound(Exception):
    """Raised when the environments name
    isn't specified"""
    pass

class WrongNumberOfAgent(Exception):
    """Raised when the number of agent doesn't
    match the environment specification"""
    pass

class WrongNumberOfAction(Exception):
    """Raised when the number of action doesn't
    match the environment specification"""
    pass

class WrongNumberOfState(Exception):
    """Raised when the number of state doesn't
    match the environment specification"""
    pass


class RewardTypeNotFound(Exception):
    """Raised when the type of the reward isn't found
    (For PBeautyGame)"""
    pass

class WrongActionInputLength(Exception):
    """Raised when the the length of the
    input doesn't match the number of agents"""



class NStepMatrixGame(BaseGame):
    def __init__(self, args, steps=10, good_branches=2, state_type='idx'): # batch_size=None, 

        self.args = args
        # Define the agents
        self.agent_num = 2
        self.action_dim = self.args.action_dim
        self.scenario_name = self.args.scenario_name
        self.good_branches = good_branches

        self.episode_limit = steps
        self.state_type = state_type

        # print('self.state_type', self.state_type)
        # Define the internal state
        self.steps = 0

        self.pay_off = np.zeros(tuple([good_branches] + [self.episode_limit] + [self.agent_num] + [self.action_dim] * self.agent_num))
        if self.scenario_name == "coordination"and self.action_dim==3:
            for i in range(self.good_branches):
                self.pay_off[i][0][0] = [[1, 0, 0],[0, 0, 0],[0, 0, 1]]
                self.pay_off[i][0][1] = [[1, 0, 0],[0, 0, 0],[0, 0, 1]]
                self.pay_off[i][-1][0] = [[0, 0, 5],[30, 10, 0],[20, 0, 0]]
                self.pay_off[i][-1][1] = [[0, 0, 10],[0, 5, 0],[15, 0, 0]]
            for i in range(self.episode_limit-2):
                self.pay_off[0][i+1][0] = [[1, 0, 0],[0, 0, 0],[0, 0, 0]]
                self.pay_off[0][i+1][1] = [[1, 0, 0],[0, 0, 0],[0, 0, 0]]
                self.pay_off[1][i+1][0] = [[0, 0, 0],[0, 0, 0],[0, 0, 1]]
                self.pay_off[1][i+1][1] = [[0, 0, 0],[0, 0, 0],[0, 0, 1]]
            # self.pay_off[0][-1][0] = [[0, 0, 5],[30, 10, 0],[20, 0, 0]]
            # self.pay_off[0][-1][1] = [[0, 0, 10],[0, 5, 0],[15, 0, 0]]
            # self.pay_off[1][-1][0] = [[1, 1, 1],[1, 1, 1],[1, 1, 1]]
            # self.pay_off[1][-1][1] = [[1, 1, 1],[1, 1, 1],[1, 1, 1]]            

        elif self.scenario_name == "cooperation" and self.action_dim==2:
            for i in range(self.good_branches):
                self.pay_off[i][0][0] = [[1, 0],[0, 1]]
                self.pay_off[i][0][1] = [[1, 0],[0, 1]]
            for i in range(self.episode_limit-2):
                self.pay_off[0][i+1][0] = [[1, 0],[0, 0]]
                self.pay_off[0][i+1][1] = [[1, 0],[0, 0]]
                self.pay_off[1][i+1][0] = [[0, 0],[0, 1]]
                self.pay_off[1][i+1][1] = [[0, 0],[0, 1]]
            self.pay_off[0][-1][0] = [[1, 1],[1,4]]
            self.pay_off[0][-1][1] = [[1, 1],[1,4]]
            self.pay_off[1][-1][0] = [[1, 1],[1,1]]
            self.pay_off[1][-1][1] = [[1, 1],[1,1]]
        else:
            print("Can not support the " + self.scenario_name + "with" + self.action_dim + "actions environment.")
            raise NotImplementedError


        self.branches = 4
        self.branch = 0
        self.state_num = self.branches * (self.episode_limit + 1)

        

        self.n_actions = 3

        self.obs_dim = self.get_obs_size()

    def reset(self):
        """ Returns initial observations and states"""
        self.steps = 0
        self.branch = 0
        # return self.get_obs(), self.get_state()
        return self.get_obs()

    def step(self, actions):
        """ Returns reward, terminated, info """
        if len(actions) != self.agent_num:
            raise WrongActionInputLength(f"Expected number of actions is {self.agent_num}")
        actions = np.array(actions).reshape((self.agent_num,))
        current_branch = 0
        if (actions[0], actions[1]) == (0,0):
            current_branch = 0
        elif (actions[0], actions[1]) == (2,2):
            current_branch = 1
        else:
            current_branch = 2

        reward_n  = np.zeros((self.agent_num,))
        if self.steps == 0:
            self.branch = current_branch
            reward_n[0] = self.pay_off[0][self.steps][0][actions[0]][actions[1]]
            reward_n[1] = self.pay_off[0][self.steps][1][actions[0]][actions[1]]
        else:
            reward_n[0] = self.pay_off[self.branch][self.steps][0][actions[0]][actions[1]]
            reward_n[1] = self.pay_off[self.branch][self.steps][1][actions[0]][actions[1]]

        info = {}

        info["good_payoff"] = 0
        info["branch"] = self.branch
            

        self.steps += 1

        if self.steps < self.episode_limit and reward_n[0] > 0 and reward_n[1] > 0 and self.branch != 2:
            terminated = False
        else:
            terminated = True
        done_n = np.array([terminated for _ in range(self.agent_num)])

        # print(self.steps, reward, actions)

        info["episode_limit"] = False

        # How often the joint-actions are taken
        # info["action_00"] = 0
        # info["action_01"] = 0
        # info["action_10"] = 0
        # info["action_11"] = 0
        # if (actions[0], actions[1]) == (0, 0):
        #     info["action_00"] = 1
        # if (actions[0], actions[1]) == (0, 1):
        #     info["action_01"] = 1
        # if (actions[0], actions[1]) == (1, 0):
        #     info["action_10"] = 1
        # if (actions[0], actions[1]) == (1, 1):
        #     info["action_11"] = 1
        if terminated == True:
            state_n = self.reset()
        else:
            state_n = self.get_obs()
        info_n = []
        for agent_id in range(reward_n.shape[0]):
            info = {'individual_reward': reward_n[agent_id]}
            info_n.append(info)

        # return reward, terminated, info
        return [state_n, reward_n, done_n, info_n]

    def get_obs(self):
        """ Returns all agent observations in a list """
        if self.episode_limit == 1:
            one_hot_step = np.zeros(self.episode_limit + 1)
            one_hot_step[self.steps] = 1
        else:
            one_hot_step = np.zeros(self.episode_limit + 1 + self.branches)
            one_hot_step[self.steps] = 1
            one_hot_step[self.episode_limit + 1 + self.branch] = 1
        return np.array([np.copy(one_hot_step) for _ in range(self.agent_num)])
    
    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self.get_obs_agent(0))

    def get_state(self):
        if self.state_type == 'obs':
            return self.get_obs_agent(0)
        else:
            s = self.steps * self.branches + self.branch
            # print(self.state_num,  s )
            return np.array([s for _ in range(self.agent_num)])

    def get_state_size(self):
        """ Returns the shape of the state"""
        if self.state_type == 'obs':
            return self.get_obs_size()
        else:
            return 2

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.agent_num):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def get_stats(self):
        return None

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self):
        raise NotImplementedError