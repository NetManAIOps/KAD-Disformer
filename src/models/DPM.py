from typing import Tuple

import torch
from torch import nn


class DisentangledAtten(nn.Module):
    def __init__(self, heads=4, d_model: int = 64):
        super(DisentangledAtten, self).__init__()

        self.heads = heads
        self.head_dim = d_model // heads
        self.d_model = d_model

        assert (
            self.head_dim * heads == d_model
        ), "d_model size needs to be divisible by heads"

        self.W_common_Q = nn.Linear(d_model, d_model, bias=True)
        self.W_common_K = nn.Linear(d_model, d_model, bias=True)
        self.W_common_V = nn.Linear(d_model, d_model, bias=True)

        self.W_personal_Q = nn.Linear(d_model, d_model, bias=True)
        self.W_personal_K = nn.Linear(d_model, d_model, bias=True)
        self.W_personal_V = nn.Linear(d_model, d_model, bias=True)

        self.common_q = None
        self.common_k = None
        self.common_v = None
        self.personal_q = None
        self.personal_k = None
        self.personal_v = None

    def get_common_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.common_q = self.W_common_Q(x)
        self.common_k = self.W_common_K(x)
        self.common_v = self.W_common_V(x)
        return self.common_q, self.common_k, self.common_v

    def get_personal_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.personal_q = self.W_personal_Q(x)
        self.personal_k = self.W_personal_K(x)
        self.personal_v = self.W_personal_V(x)
        return self.personal_q, self.personal_k, self.personal_v

    def forward(self, query, key, value) -> torch.Tensor:
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        queries = self.common_q + self.personal_q
        keys = self.common_k + self.personal_k
        values = self.common_v + self.personal_v

        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        attention = torch.softmax(energy / (self.d_model ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.d_model
        )

        return out

class DecoderAttention(nn.Module):
    def __init__(self, heads=4, d_model: int = 64):
        super(DecoderAttention, self).__init__()

        self.heads = heads
        self.head_dim = d_model // heads
        self.d_model = d_model

        assert (
            self.head_dim * heads == d_model
        ), "d_model size needs to be divisible by heads"

        self.W_Q = nn.Linear(d_model, d_model, bias=True)
        self.W_K = nn.Linear(d_model, d_model, bias=True)
        self.W_V = nn.Linear(d_model, d_model, bias=True)

    def get_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.W_Q(x)
        key = self.W_K(x)
        value = self.W_V(x)
        return query, key, value

    def forward(self, query, key, value) -> torch.Tensor:
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        query = query.reshape(N, query_len, self.heads, self.head_dim)
        key = key.reshape(N, key_len, self.heads, self.head_dim)
        value = value.reshape(N, value_len, self.heads, self.head_dim)
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])

        attention = torch.softmax(energy / (self.d_model ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, value]).reshape(
            N, query_len, self.d_model
        )

        return out


if __name__ == "__main__":
    atten = DisentangledAtten()
    x = torch.randn(10, 100, 64)

    c_q, c_k, c_v = atten.get_common_qkv(x)
    p_q, p_k, p_v = atten.get_personal_qkv(x)

    out = atten(c_q + p_q, c_k + p_k, c_v + p_v)
    print(f"out shape: {out.shape}")

