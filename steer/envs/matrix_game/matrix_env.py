"""
# @Time    : 2022/6/28 
# @Author  : Zhangbin
# @File    : env.py
"""
import sys
# sys.path.append("/home/zb/Project/Multi-Agent-Transformer-main/mat/envs/")
import numpy as np
from .base_game import BaseGame
from gym.spaces import Box, Discrete

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

class MatrixGame(BaseGame):
    def __init__(self, args, payoff=None, repeated=False, max_step=25, memory=0, discrete_action=True, tuple_obs=False):
        self.game_name = args.scenario_name
        self.discrete_action = discrete_action
        self.tuple_obs = tuple_obs
        self.num_state = 1

        game_list = self.get_game_list()

        self.agent_num = game_list[self.game_name]['agent_num']
        self.action_dim = game_list[self.game_name]['action_num']

        if not self.game_name in game_list:
            raise EnvironmentNotFound(f"The game {self.game_name} doesn't exists")

        expt_num_agent = game_list[self.game_name]['agent_num']
        expt_num_action = game_list[self.game_name]['action_num']

        if expt_num_agent != self.agent_num:
            raise WrongNumberOfAgent(f"The number of agent \
                required for {self.game_name} is {expt_num_agent}")

        if expt_num_action != self.action_dim:
            raise WrongNumberOfAction(f"The number of action \
                required for {self.game_name} is {expt_num_action}")


        self.action_spaces = tuple(Box(low=-1., high=1., shape=(1,)) for _ in range(self.agent_num))
        self.observation_spaces = tuple(Discrete(1) for _ in range(self.agent_num))

        if self.discrete_action:
            self.action_spaces = tuple(Discrete(self.action_dim) for _ in range(self.agent_num))
            if memory == 0:
                self.observation_spaces = tuple(Discrete(1) for _ in range(self.agent_num))
                self.obs_dim = 1
            elif memory == 1:
                self.observation_spaces = tuple(Discrete(5) for _ in range(self.agent_num))
                self.obs_dim = 5
        else:
            self.action_range = [-1., 1.]
            self.action_spaces = tuple(Box(low=-1., high=1., shape=(1,)) for _ in range(self.agent_num))
            if memory == 0:
                self.observation_spaces = tuple(Discrete(1) for _ in range(self.agent_num))
            elif memory == 1:
                self.observation_spaces =  tuple(Box(low=-1., high=1., shape=(12,)) for _ in range(self.agent_num))

        self.env_specs = [self.observation_spaces, self.action_spaces]

        self.t = 0
        self.repeated = repeated
        self.max_step = max_step
        self.memory = memory
        self.previous_action = 0
        self.previous_actions = []
        self.ep_rewards = np.zeros(self.agent_num)

        if payoff is not None:
            payoff = np.array(payoff)
            assert payoff.shape == tuple([self.agent_num] + [self.action_dim] * self.agent_num)
            self.payoff = payoff
        if payoff is None:
            self.payoff = np.zeros(tuple([self.agent_num] + [self.action_dim] * self.agent_num))

        if self.game_name == 'coordination_0_0':
            self.payoff[0]=[[1,-1],
                           [-1,-1]]
            self.payoff[1]=[[1,-1],
                           [-1,-1]]
        elif self.game_name == 'coordination_same_action_with_preference':
            self.payoff[0]=[[40, 0],
                           [80, 20]]
            self.payoff[1]=[[40, 0],
                           [0, 20]]
        elif self.game_name == 'zero_sum_nash_0_1':
            # payoff tabular of zero-sum game scenario. nash equilibrium: (Agenat1's action=0,Agent2's action=1)
            self.payoff[0]=[[5,2],
                            [-1,6]]
            self.payoff[1]=[[-5,-2],
                            [1,-6]]
        elif self.game_name == 'matching_pennies':
            # payoff tabular of zero-sumgame scenario. matching pennies
            self.payoff[0]=[[1,-1],
                           [-1,1]]
            self.payoff[1]=[[-1,1],
                           [1,-1]]
        elif self.game_name == 'matching_pennies_3':
            self.payoff[0]=[
                            [ [1,-1],
                              [-1,1] ],
                            [ [1, -1],
                             [-1, 1]]
                            ]
            self.payoff[1]=[
                            [ [1,-1],
                              [1,-1] ],
                            [[-1, 1],
                             [-1, 1]]
                            ]
            self.payoff[2] = [
                            [[-1, -1],
                             [1, 1]],
                            [[1, 1],
                             [-1, -1]]
                            ]
        elif self.game_name =='prison_lola':
            self.payoff[0]=[[-1,-3],
                           [0,-2]]
            self.payoff[1]=[[-1,0],
                           [-3,-2]]
        elif self.game_name =='prison':
            self.payoff[0]=[[3, 1],
                           [4, 2]]
            self.payoff[1]=[[3, 4],
                           [1, 2]]
        elif self.game_name =='stag_hunt':
            self.payoff[0]=[[4, 1],
                           [3, 2]]
            self.payoff[1]=[[4, 3],
                           [1, 2]]
        elif self.game_name =='chicken': # snowdrift
            self.payoff[0]=[[3, 2],
                           [4, 1]]
            self.payoff[1]=[[3, 4],
                           [2, 1]]
        elif self.game_name =='harmony':
            self.payoff[0] = [[4, 3],
                             [2, 1]]
            self.payoff[1] = [[4, 2],
                             [3, 1]]
        elif self.game_name == 'wolf_05_05':
            self.payoff[0] = [[0, 3],
                             [1, 2]]
            self.payoff[1] = [[3, 2],
                             [0, 1]]
            # \alpha, \beta = 0, 0.9, nash is 0.5 0.5
            # Q tables given, matian best response, learn a nash e.
        elif self.game_name == 'climbing':
            self.payoff[0] = [[20, 0, 0],
                              [30, 10, 0],
                              [0, 0, 5]]
            self.payoff[1] = [[15, 0, 0],
                              [0, 5, 0],
                              [0, 0, 10]]
        elif self.game_name == 'penalty':
            self.payoff[0] = [[10, 8, 0],[8, 8, 0],[0, 0, 20]]
            self.payoff[1] = [[10, 8, 0],[8, 8, 0],[0, 0, 20]]
        elif self.game_name == 'rock_paper_scissors':
            self.payoff[0] = [[0, -1, 1],
                              [1, 0, -1],
                              [-1, 1, 0]
                              ]
            self.payoff[1] = [[0, 1, -1],
                              [-1, 0, 1],
                              [1, -1, 0]
                              ]
        elif self.game_name == 'QPLEX_matrix_env':
            self.payoff[0] = [[8, -12, -12],
                              [-12, 6, 0],
                              [-12, 0, 6]
                              ]
            self.payoff[1] = [[8, -12, -12],
                              [-12, 6, 0],
                              [-12, 0, 6]
                              ]
        elif self.game_name == 'different_order':
            self.payoff[0] = [[0, -10, -8],[-5, -5, -15],[5, -10, -10]]
            self.payoff[1] = [[5, -5, 4],[-10, 0, -5],[0, -5, 5]]
        elif self.game_name == 'different_order_reverse':
            self.payoff[0] = [[5, -10, 0],[-5, 0, -5],[4, -5, 5]]
            self.payoff[1] = [[0, -5, 5],[-10, -5, -10],[-8, -15, -10]]
        elif self.game_name == 'penalty-100':
            self.payoff[0] = [[-100, 0, 10],[0, 2, 0],[8, 0, -100]]
            self.payoff[1] = [[-100, 0, 10],[0, 2, 0],[8, 0, -100]]
        elif self.game_name == 'penalty-50':
            self.payoff[0] = [[-50, 0, 10],[0, 2, 0],[8, 0, -50]]
            self.payoff[1] = [[-50, 0, 10],[0, 2, 0],[8, 0, -50]]
        elif self.game_name == 'penalty-25':
            self.payoff[0] = [[-25, 0, 10],[0, 2, 0],[8, 0, -25]]
            self.payoff[1] = [[-25, 0, 10],[0, 2, 0],[8, 0, -25]]
        elif self.game_name == 'penalty0':
            self.payoff[0] = [[0, 0, 10],[0, 2, 0],[8, 0, 0]]
            self.payoff[1] = [[0, 0, 10],[0, 2, 0],[8, 0, 0]]
        elif self.game_name == 'penalty-75':
            self.payoff[0] = [[-75, 0, 10],[0, 2, 0],[8, 0, -75]]
            self.payoff[1] = [[-75, 0, 10],[0, 2, 0],[8, 0, -75]]
        elif self.game_name == 'penalty-1000':
            self.payoff[0] = [[-1000, 0, 10],[0, 2, 0],[8, 0, -1000]]
            self.payoff[1] = [[-1000, 0, 10],[0, 2, 0],[8, 0, -1000]]

        self.rewards = np.zeros((self.agent_num,))

    @staticmethod
    def get_game_list():
        return {
            'coordination_0_0': {'agent_num': 2, 'action_num': 2},
            'coordination_same_action_with_preference': {'agent_num': 2, 'action_num': 2},
            'zero_sum_nash_0_1': {'agent_num': 2, 'action_num': 2},
            'matching_pennies': {'agent_num': 2, 'action_num': 2},
            'matching_pennies_3': {'agent_num': 3, 'action_num': 2},
            'prison_lola': {'agent_num': 2, 'action_num': 2},
            'prison': {'agent_num': 2, 'action_num': 2},
            'stag_hunt': {'agent_num': 2, 'action_num': 2},
            'chicken': {'agent_num': 2, 'action_num': 2},
            'harmony': {'agent_num': 2, 'action_num': 2},
            'wolf_05_05': {'agent_num': 2, 'action_num': 2},
            'climbing': {'agent_num': 2, 'action_num': 3},
            'penalty-100': {'agent_num': 2, 'action_num': 3},
            'rock_paper_scissors': {'agent_num': 2, 'action_num': 3},
            'QPLEX_matrix_env': {'agent_num': 2, 'action_num': 3},
            'different_order':{'agent_num': 2, 'action_num': 3},
            'different_order_reverse':{'agent_num': 2, 'action_num': 3},
            'penalty0': {'agent_num': 2, 'action_num': 3},
            'penalty-25': {'agent_num': 2, 'action_num': 3},
            'penalty-50': {'agent_num': 2, 'action_num': 3},
            'penalty-75': {'agent_num': 2, 'action_num': 3},
            'penalty-1000': {'agent_num': 2, 'action_num': 3}
        }


    def V(self, alpha, beta, payoff):
        u = payoff[(0, 0)] - payoff[(0, 1)] - payoff[(1, 0)] + payoff[(1, 1)]
        return alpha * beta * u + alpha * (payoff[(0, 1)] - payoff[(1, 1)]) + beta * (
                payoff[(1, 0)] - payoff[(1, 1)]) + payoff[(1, 1)]

    def get_rewards(self, actions):
        reward_n = np.zeros((self.agent_num,))
        if self.discrete_action:
            for i in range(self.agent_num):
                assert actions[i] in range(self.action_dim)
                reward_n[i] = self.payoff[i][tuple(actions)]
        else:
            actions = (actions + 1.) / 2.
            for i in range(self.agent_num):
                reward_n[i] = self.V(actions[0], actions[1], np.array(self.payoff[i]))
            # print(np.array(self.payoff[0]))
            # print('actions', actions)
            # print('reward', reward_n)
        return reward_n

    def step(self, actions):
        if len(actions) != self.agent_num:
            raise WrongActionInputLength(f"Expected number of actions is {self.agent_num}")

        actions = np.array(actions).reshape((self.agent_num,))
        reward_n = self.get_rewards(actions)
        self.rewards = reward_n
        info_n = []
        done_n = np.array([True for _ in range(self.agent_num)])
        if self.repeated:
            done_n = np.array([False for _ in range(self.agent_num)])
        self.t += 1
        if self.t >= self.max_step:
            done_n = np.array([True for _ in range(self.agent_num)])

        state = [0] * (self.action_dim * self.agent_num * (self.memory) + 1)
        # state_n = [tuple(state) for _ in range(self.agent_num)]
        if self.memory > 0 and self.t > 0:
            # print('actions', actions)
            if self.discrete_action:
                state[actions[1] + 2 * actions[0] + 1] = 1
            else:
                state = actions

        # tuple for tublar case, which need a hashabe obersrvation
        if self.tuple_obs:
            state_n = [tuple(state) for _ in range(self.agent_num)]
        else:
            state_n = np.array([state for _ in range(self.agent_num)])

        # for i in range(self.agent_num):
        #     state_n[i] = tuple(state_n[i][:])

        self.previous_actions.append(tuple(actions))
        self.ep_rewards += np.array(reward_n)
        for agent_id in range(reward_n.shape[0]):
            info = {'individual_reward': reward_n[agent_id]}
            info_n.append(info)
        # print(state_n, reward_n, done_n, info)
        return [state_n, reward_n, done_n, info_n]

    def reset(self):
        # print('reward,', self.ep_rewards / self.t)
        self.ep_rewards = np.zeros(2)
        self.t = 0
        self.previous_action = 0
        self.previous_actions = []
        state = [0] * (self.action_dim * self.agent_num * (self.memory)  + 1)
        # state_n = [tuple(state) for _ in range(self.agent_num)]
        if self.memory > 0:
            state = [0., 0.]
        if self.tuple_obs:
            # print(self.agent_num)
            state_n = [tuple(state) for _ in range(self.agent_num)]
        else:
            state_n = np.array([state for _ in range(self.agent_num)])
        # print(state_n)

        return state_n

    def render(self, mode='human', close=False):
        if mode == 'human':
            print(self.__str__())

    def terminate(self):
        pass

    def get_joint_reward(self):
        return self.rewards

    def __str__(self):
        content = 'Game Name {}, Number of Agent {}, Number of Action \n'.format(self.game_name, self.agent_num, self.action_dim)
        content += 'Payoff Matrixs:\n\n'
        for i in range(self.agent_num):
            content += 'Agent {}, Payoff:\n {} \n\n'.format(i+1, str(self.payoff[i]))
        return content


if __name__ == '__main__':
    print(MatrixGame.get_game_list())
    game = MatrixGame('matching_pennies_3')
    print(game)
    print(game.step([0,0,0]))
# class Env(object):
#     """
#     # 环境中的智能体
#     """
#     def __init__(self, game_name, payoff=None):
        
#         self.max_step=25
#         self.game_name = game_name

#         if payoff is None:
#             self.payoff = np.zeros(tuple([self.agent_num] + [self.action_dim] * self.agent_num))
        
#         if self.game_name == 'coordination_0_0':
#             self.payoff[0]=[[1,-1],
#                            [-1,-1]]
#             self.payoff[1]=[[1,-1],
#                            [-1,-1]]
#         elif self.game_name == 'coordination_same_action_with_preference':
#             self.payoff[0]=[[40, 0],
#                            [80, 20]]
#             self.payoff[1]=[[40, 0],
#                            [0, 20]]
#         elif self.game_name == 'zero_sum_nash_0_1':
#             # payoff tabular of zero-sum game scenario. nash equilibrium: (Agenat1's action=0,Agent2's action=1)
#             self.payoff[0]=[[5,2],
#                             [-1,6]]
#             self.payoff[1]=[[-5,-2],
#                             [1,-6]]
#         elif self.game_name == 'matching_pennies':
#             # payoff tabular of zero-sumgame scenario. matching pennies
#             self.payoff[0]=[[1,-1],
#                            [-1,1]]
#             self.payoff[1]=[[-1,1],
#                            [1,-1]]
#         elif self.game_name == 'matching_pennies_3':
#             self.payoff[0]=[
#                             [ [1,-1],
#                               [-1,1] ],
#                             [ [1, -1],
#                              [-1, 1]]
#                             ]
#             self.payoff[1]=[
#                             [ [1,-1],
#                               [1,-1] ],
#                             [[-1, 1],
#                              [-1, 1]]
#                             ]
#             self.payoff[2] = [
#                             [[-1, -1],
#                              [1, 1]],
#                             [[1, 1],
#                              [-1, -1]]
#                             ]
#         elif self.game_name =='prison_lola':
#             self.payoff[0]=[[-1,-3],
#                            [0,-2]]
#             self.payoff[1]=[[-1,0],
#                            [-3,-2]]
#         elif self.game_name =='prison':
#             self.payoff[0]=[[3, 1],
#                            [4, 2]]
#             self.payoff[1]=[[3, 4],
#                            [1, 2]]
#         elif self.game_name =='stag_hunt':
#             self.payoff[0]=[[4, 1],
#                            [3, 2]]
#             self.payoff[1]=[[4, 3],
#                            [1, 2]]
#         elif self.game_name =='chicken': # snowdrift
#             self.payoff[0]=[[3, 2],
#                            [4, 1]]
#             self.payoff[1]=[[3, 4],
#                            [2, 1]]
#         elif self.game_name =='harmony':
#             self.payoff[0] = [[4, 3],
#                              [2, 1]]
#             self.payoff[1] = [[4, 2],
#                              [3, 1]]
#         elif self.game_name == 'wolf_05_05':
#             self.payoff[0] = [[0, 3],
#                              [1, 2]]
#             self.payoff[1] = [[3, 2],
#                              [0, 1]]
#             # \alpha, \beta = 0, 0.9, nash is 0.5 0.5
#             # Q tables given, matian best response, learn a nash e.
#         elif self.game_name == 'climbing':
#             self.payoff[0] = [[20, 0, 0],
#                               [30, 10, 0],
#                               [0, 0, 5]]
#             self.payoff[1] = [[15, 0, 0],
#                               [0, 5, 0],
#                               [0, 0, 10]]
#         elif self.game_name == 'penalty':
#             self.payoff[0] = [[15, 10, 0],
#                               [10, 10, 0],
#                               [0, 0, 30]]
#             self.payoff[1] = [[15, 10, 0],
#                               [10, 10, 0],
#                               [0, 0, 30]]
#         elif self.game_name == 'rock_paper_scissors':
#             self.payoff[0] = [[0, -1, 1],
#                               [1, 0, -1],
#                               [-1, 1, 0]
#                               ]
#             self.payoff[1] = [[0, 1, -1],
#                               [-1, 0, 1],
#                               [1, -1, 0]
#                               ]

#         self.agent_num = len(self.payoff)  
#         self.obs_dim = 1
#         self.action_dim = len(self.payoff[0]) 
#         self.rewards = np.zeros((self.agent_num,))

#     def reset(self):
#         """
#         # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
#         """
#         sub_agent_obs = []
#         for i in range(self.agent_num):
#             sub_obs = np.random.random(size=(14, ))
#             sub_agent_obs.append(sub_obs)
#         return sub_agent_obs

#     def step(self, actions):
#         """
#         # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
#         # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作纬度为5，所里每个元素shape = (5, )
#         """
#         sub_agent_obs = []
#         sub_agent_reward = []
#         sub_agent_done = []
#         sub_agent_info = []
#         for i in range(self.agent_num):
#             sub_agent_obs.append(np.random.random(size=(14,)))
#             sub_agent_reward.append([np.random.rand()])
#             sub_agent_done.append(False)
#             sub_agent_info.append({})

#         return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
#     def get_obs(self, actions):
    
#         state = [0] * (self.action_dim * self.agent_num)
#         obs = np.array([state for _ in range(self.agent_num)])
    
#     def get_rewards(self, actions):
#         reward_n = np.zeros((self.agent_num,))
#         if self.discrete_action:
#             for i in range(self.agent_num):
#                 assert actions[i] in range(self.action_num)
#                 reward_n[i] = self.payoff[i][tuple(actions)]
#         else:
#             actions = (actions + 1.) / 2.
#             for i in range(self.agent_num):
#                 reward_n[i] = self.V(actions[0], actions[1], np.array(self.payoff[i]))
#             # print(np.array(self.payoff[0]))
#             # print('actions', actions)
#             # print('reward', reward_n)
#         return reward_n