import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from algorithms.utils.util import check, init
from torch.distributions import Categorical, Normal


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


def continuous_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv):

    act_mean, v_loc = decoder(action, obs_rep)
    action_std = torch.sigmoid(decoder.log_std) * 0.5
    distri = Normal(act_mean, action_std)

    # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
    # distri = Normal(act_mean, log_std.exp())

    action_log = distri.log_prob(action)    #计算value在定义的正态分布中对应的概率的对数,计算当前策略下的动作分布,然后计算旧策略采样下的动作在本分布下的概率对数
    entropy = distri.entropy()
    return v_loc, action_log, entropy

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, seq_size, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        attn_pdrop = 0.1
        resid_pdrop = 0.1
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(seq_size + 1, seq_size + 1))
                             .view(1, 1, seq_size + 1, seq_size + 1))

        self.att_bp = None

    def forward(self, x):
        B, L, D = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs) L:num_agents
        q = self.query(x).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(x).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        # att = self.attn_drop(att)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        # y = self.resid_drop(self.proj(y))
        y = self.proj(y)
        return y


class Block(nn.Module):
    def __init__(self, n_embed, n_head, seq_size, masked):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.attn = SelfAttention(n_embed, n_head, seq_size, masked)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embed, 1*n_embed), activate=True),
            nn.GELU(),
            init_(nn.Linear(1*n_embed, n_embed))
            # nn.Dropout(resid_pdrop)
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x
    
class LearnableAbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size):
        super().__init__()
        self.is_absolute = True
        self.embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.register_buffer('position_ids', torch.arange(max_position_embeddings))

    def forward(self, x):
        """
        return (b l d) / (b h l d)
        """
        position_ids = self.position_ids[:x.size(-2)]

        if x.dim() == 3:
            return x + self.embeddings(position_ids)[None, :, :]

class ActorCritic(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, n_embed, n_head, n_agent, action_dim, n_block, action_type='Discrete', device="CPU"):
        super(ActorCritic, self).__init__()

        self.device = device
        self.n_agent = n_agent
        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

        if action_type == 'Discrete':
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embed, bias=False), activate=True),
                                                    nn.GELU())
        else:
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embed), activate=True), nn.GELU())

        self.seq_ln = nn.LayerNorm(n_embed)
        self.obs_proj = nn.Sequential(
            init_(nn.Linear(n_embed, n_embed), activate=True), nn.GELU()
            # nn.LayerNorm(n_embed)
        )

        self.obs_blocks = nn.Sequential(*[Block(n_embed, n_head, n_agent+1, masked=False) for _ in range(n_block)])
        # self.ac_blocks = nn.Sequential(*[Block(n_embed, n_head, n_agent+1, masked=True) for _ in range(n_block)])
        self.ac_blocks = nn.Sequential(*[Block(n_embed, n_head, n_agent+1, masked=True) for _ in range(n_block)])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(n_embed),
            init_(nn.Linear(n_embed, n_embed), activate=True), nn.GELU(),
            init_(nn.Linear(1*n_embed, n_embed))
        )
        self.action_head = nn.Sequential(init_(nn.Linear(n_embed, n_embed), activate=True), nn.GELU(), nn.LayerNorm(n_embed),
                                      init_(nn.Linear(n_embed, action_dim)))
        self.value_head = nn.Sequential(init_(nn.Linear(n_embed, n_embed), activate=True), nn.GELU(), nn.LayerNorm(n_embed),
                                  init_(nn.Linear(n_embed, 1)))
        self.pos_embed = LearnableAbsolutePositionEmbedding(1+n_agent, n_embed)

    def zero_std(self):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(self.device)
            self.log_std.data = log_std


    def forward(self, action, obs_emb):
        state_obs_rep = self.obs_blocks(obs_emb)
        state_rep = state_obs_rep[:,0:1,:]
        obs_rep = state_obs_rep[:,1:,:]
        action_emb = self.action_encoder(action)
        action_rep = action_emb

        seq = self.pos_embed(torch.cat([state_rep, action_rep], dim=1))
        # seq = torch.cat([state_rep, action_rep], dim=1) # v3
        x = self.ac_blocks(seq)
        x[:,:-1,:] += obs_rep
        x = x + self.out_proj(x)
        x = self.seq_ln(x)
        logit = self.action_head(x)[:, :-1, :]
        v_loc = self.value_head(x)[:, :-1, :]
        return logit, v_loc

class Steer(nn.Module):

    def __init__(self, state_dim, obs_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False, add_state_token=True):
        super(Steer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device
        self.add_state_token = add_state_token


        self.state_dim = state_dim
        self.encode_state = encode_state
        # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))

        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
        
        self.ln = nn.LayerNorm(n_embd)
        
        if self.add_state_token:
            self.class_token_encoding = nn.Parameter(torch.zeros(1, 1, n_embd))
            nn.init.trunc_normal_(self.class_token_encoding, mean=0.0, std=0.02)
        self.decoder = ActorCritic(n_embd, n_head, n_agent, action_dim, n_block, action_type, self.device)
        
        self.to(device)

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    def forward(self, state, obs, action, available_actions=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(state)[0]

        obs_embeddings = self.obs_encoder(obs)
        if self.add_state_token:
            obs_rep = torch.cat([self.class_token_encoding.expand(obs_embeddings.shape[0], -1, -1), obs_embeddings], dim=1) #batch_size, n_agent+1, embd
        else:
            state_embeddings = self.state_encoder(state[:,0:1,:])
            obs_rep = torch.cat([state_embeddings,obs_embeddings], dim=1) #batch_size, n_agent+1, embd
        
        if self.action_type == 'Discrete':
            action = action.long()
            v_loc, action_log, entropy = discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions)
        else:
            v_loc, action_log, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                          self.n_agent, self.action_dim, self.tpdv)

        return action_log, v_loc, entropy

    def get_actions(self, state, obs, available_actions=None, deterministic=False):

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]

        obs_embeddings = self.obs_encoder(obs)
        if self.add_state_token:
            obs_rep = torch.cat([self.class_token_encoding.expand(obs_embeddings.shape[0], -1, -1), obs_embeddings], dim=1) #batch_size, n_agent+1, embd
        else:
            state_embeddings = self.state_encoder(state[:,0:1,:])
            obs_rep = torch.cat([state_embeddings,obs_embeddings], dim=1) #batch_size, n_agent+1, embd

        if self.action_type == "Discrete":
            v_loc, output_action, output_action_log = discrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                           self.n_agent, self.action_dim, self.tpdv,
                                                                           available_actions, deterministic)
        else:
            v_loc, output_action, output_action_log = continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                             self.n_agent, self.action_dim, self.tpdv,
                                                                             deterministic)

        return output_action, output_action_log, v_loc

    def get_values(self, state, obs, actions, available_actions=None):
        # state unused

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        # state_embeddings = self.state_encoder(state[:,0:1,:])
        obs_embeddings = self.obs_encoder(obs)
        # obs_embeddings = self.state_encoder(state)
        if self.add_state_token:
            obs_rep = torch.cat([self.class_token_encoding.expand(obs_embeddings.shape[0], -1, -1), obs_embeddings], dim=1) #batch_size, n_agent+1, embd
        else:
            state_embeddings = self.state_encoder(state[:,0:1,:])
            obs_rep = torch.cat([state_embeddings,obs_embeddings], dim=1) #batch_size, n_agent+1, embd

        if self.action_type == 'Discrete':
            actions = actions.long()
            v_tot, _, _ = discrete_parallel_act(self.decoder, obs_rep, obs, actions, 1,
                                                self.n_agent, self.action_dim, self.tpdv, available_actions)
        else:
            v_tot, action_log, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, actions, 1,
                                                          self.n_agent, self.action_dim, self.tpdv)
        return v_tot