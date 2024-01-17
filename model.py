import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DecisionMLP(nn.Module):
    def __init__(self, env_name, env, goal_dim=2, h_dim=1024):
        super().__init__()

        env_name = env_name
        state_dim = env.observation_space['observation'].shape[0]
        act_dim = env.action_space.shape[0]

        self.mlp = nn.Sequential(
                nn.Linear(state_dim + goal_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, act_dim),
                nn.Tanh()
            )         

    def forward(self, states, goals):
        h = torch.cat((states, goals), dim=-1)
        action_preds = self.mlp(h)
        return action_preds
    
class MaskedCausalAttention(nn.Module):
    '''
    Thanks https://github.com/nikhilbarhate99/min-decision-transformer/tree/master
    '''
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.dropout = drop_p
        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

    def forward(self, x):
        B,T,C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        attention = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out

class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        x = x + self.attention(self.ln1(x)) # residual
        x = x + self.mlp(self.ln2(x)) # residual
        return x

class DecisionTransformer(nn.Module):
    def __init__(self, env_name, env, n_blocks, h_dim, context_len,
                 n_heads, drop_p, goal_dim=2, max_timestep=4096):
        super().__init__()

        self.env_name = env_name
        self.state_dim = env.observation_space['observation'].shape[0]
        self.act_dim = env.action_space.shape[0]
        self.goal_dim = goal_dim
        self.n_heads = n_heads
        self.h_dim = h_dim

        ### transformer blocks
        input_seq_len = 3 * context_len 
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)        
        self.embed_goal = torch.nn.Linear(goal_dim, h_dim)
        self.embed_state = torch.nn.Linear(self.state_dim, h_dim)
        self.embed_action = torch.nn.Linear(self.act_dim, h_dim)

        ### prediction heads
        self.final_ln = nn.LayerNorm(h_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, self.act_dim)] + ([nn.Tanh()]))
        )

    def forward(self, states, actions, goals):
        B, T, _ = states.shape

        timesteps = torch.arange(0, T, dtype=torch.long, device=states.device) 
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings       #B, T, h_dim
        action_embeddings = self.embed_action(actions) + time_embeddings    #B, T, h_dim
        goal_embeddings = self.embed_goal(goals) + time_embeddings          #B, T, h_dim

        h = torch.stack(
            (goal_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)
        
        # transformer and prediction
        h = self.transformer(h)

        h = self.final_ln(h)

        # get h reshaped such that its size = (B , 3 , T , h_dim) and
        # h[:, 0, t] is conditioned on the input sequence g_0, s_0, a_0 ... g_t
        # h[:, 1, t] is conditioned on the input sequence g_0, s_0, a_0 ... g_t, s_t
        # h[:, 2, t] is conditioned on the input sequence g_0, s_0, a_0 ... g_t, s_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus 
        # the 3 input variables at that timestep (g_t, s_t, a_t) in sequence.
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)              # B, 3, T, h_dim
        action_preds = self.predict_action(h[:,1])
        return action_preds