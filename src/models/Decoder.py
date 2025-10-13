import torch
from torch import nn

from .DPM import DecoderAttention


class DecoderTransformerBlock(nn.Module):
    def __init__(self, heads=4, d_model: int = 64, dropout=0):
        super(DecoderTransformerBlock, self).__init__()

        self.heads = heads
        self.d_model = d_model

        self.attention = DecoderAttention(
            heads=heads,
            d_model=d_model,
        )


        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.fd = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor, query_outside = None) -> torch.Tensor:
        _query, key, value = self.attention.get_qkv(x)

        if query_outside is not None:
            query = query_outside
        else:
            query = _query

        attention = self.attention(query, key, value)
        x = self.dropout(self.norm1(attention + x))
        forward = self.fd(x)
        out = self.dropout(self.norm2(forward + x))

        return out


class KAD_DisformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers=1,
        heads=4,
        d_model=64,
        dropout=0,
        W_common_Q=None,
        W_personal_Q=None,
    ):
        super(KAD_DisformerDecoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                DecoderTransformerBlock(
                    heads=heads,
                    d_model=d_model,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.W_common_Q = W_common_Q
        self.W_personal_Q = W_personal_Q

    def forward(self, x):
        out = None
        for layer in self.layers:
            if self.W_personal_Q is None or self.W_common_Q is None:
                out = layer(x)
            else:
                out = layer(x, self.W_personal_Q(x) + self.W_common_Q(x))

        return out
