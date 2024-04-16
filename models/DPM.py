import torch
from torch import nn

class DisentangledAtten(nn.Module):
    def __init__(self, win_len, heads=1):
        super(DisentangledAtten, self).__init__()

        self.win_len = win_len
        self.heads = heads
        self.head_dim = self.win_len // heads

        assert (
            self.head_dim * heads == win_len
        ), "Sliding window size needs to be divisible by heads"



        self.W_common_Q = nn.Linear(win_len, win_len, bias=True)
        self.W_common_K = nn.Linear(win_len, win_len, bias=True)
        self.W_common_V = nn.Linear(win_len, win_len, bias=True)

        self.W_personal_Q = nn.Linear(win_len, win_len, bias=True)
        self.W_personal_K = nn.Linear(win_len, win_len, bias=True)
        self.W_personal_V = nn.Linear(win_len, win_len, bias=True)

        self.fc_out = nn.Linear(win_len, win_len, bias=True)

    def forward(self, query, key, value, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        queries = self.W_common_Q(query) + self.W_personal_Q(query) 
        keys = self.W_common_K(key) + self.W_personal_K(key) 
        values = self.W_common_V(value) + self.W_personal_V(value) 


        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        attention = torch.softmax(energy / (self.win_len ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out