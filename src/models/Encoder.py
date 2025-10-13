import torch
from torch import nn

from .Adapter import EncoderAdapter
from .DPM import DisentangledAtten


class EncoderTransformerBlock(nn.Module):
    def __init__(self, heads=4, d_model: int = 64, dropout=0, forward_expansion=2):
        super(EncoderTransformerBlock, self).__init__()

        self.heads = heads
        self.d_model = d_model

        self.attention = DisentangledAtten(
            heads=heads,
            d_model=d_model,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.encoder_adaptor = EncoderAdapter(d_model, d_model, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_q, c_k, c_v = self.attention.get_common_qkv(x)
        p_q, p_k, p_v = self.attention.get_personal_qkv(x)
        query, key, value = c_q + p_q, c_k + p_k, c_v + p_v

        attention = self.attention(query, key, value)
        x = self.dropout(self.norm1(attention + x))
        forward = self.encoder_adaptor(x)
        out = self.dropout(self.norm2(forward + x))

        return out


class KAD_DisformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers=1,
        heads=4,
        d_model=64,
        dropout=0,
        forward_expansion=1,
    ):
        super(KAD_DisformerEncoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                EncoderTransformerBlock(
                    heads=heads,
                    d_model=d_model,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = None
        for layer in self.layers:
            out = layer(x)

        return out


if __name__ == '__main__':
    model = KAD_DisformerEncoder(
        num_layers=1,
        heads=4,
        d_model=64,
        dropout=0,
        forward_expansion=1,
    )

    input = torch.randn(2, 3, 64, 64)
    output = model(input)