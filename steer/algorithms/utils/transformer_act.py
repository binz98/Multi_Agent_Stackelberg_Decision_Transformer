import torch
from torch.distributions import Categorical, Normal
from torch.nn import functional as F


def discrete_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                available_actions=None, deterministic=False):
    one_hot_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):
        logit, v_loc = decoder(one_hot_action, obs_rep)
        logit = logit[:, i, :]
        if available_actions is not None:
            logit[available_actions[:, i, :] == 0] = -1e10

        distri = Categorical(logits=logit)
        action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action.unsqueeze(-1)
        output_action_log[:, i, :] = action_log.unsqueeze(-1)
        one_hot_action[: , i, :] = F.one_hot(action, num_classes=action_dim)
    return v_loc, output_action, output_action_log


def discrete_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv,
                          available_actions=None):
    one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim).to(**tpdv)  # (batch, n_agent, action_dim)
    logit, v_loc = decoder(one_hot_action, obs_rep)
    if available_actions is not None:
        logit[available_actions == 0] = -1e10

    distri = Categorical(logits=logit)
    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
    entropy = distri.entropy().unsqueeze(-1)
    return v_loc, action_log, entropy


def continuous_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                  deterministic=False):
    
    output_action = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.float32).to(**tpdv)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):
        act_mean, v_loc = decoder(output_action, obs_rep)
        act_mean = act_mean[:, i, :]
        action_std = torch.sigmoid(decoder.log_std) * 0.5

        # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
        # distri = Normal(act_mean, log_std.exp())
        distri = Normal(act_mean, action_std)
        action = act_mean if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action
        output_action_log[:, i, :] = action_log

        # print("act_mean: ", act_mean)
        # print("action: ", action)

    return v_loc, output_action, output_action_log


def continuous_parallel_act(decoder, obs_emb, obs, action, batch_size, n_agent, action_dim, tpdv):

    act_mean, v_loc = decoder(action, obs_emb)
    action_std = torch.sigmoid(decoder.log_std) * 0.5
    distri = Normal(act_mean, action_std)

    # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
    # distri = Normal(act_mean, log_std.exp())

    action_log = distri.log_prob(action)    #计算value在定义的正态分布中对应的概率的对数,计算当前策略下的动作分布,然后计算旧策略采样下的动作在本分布下的概率对数
    entropy = distri.entropy()
    return v_loc, action_log, entropy
