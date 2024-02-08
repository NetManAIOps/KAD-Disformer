from torch import nn
from .DPM import DisentangledAtten
from .Adapter import EncoderAdapter

class EncoderTransformerBlock(nn.Module):
    def __init__(self, win_len, heads, dropout, forward_expansion=1):
        super(EncoderTransformerBlock, self).__init__()
        self.attention = DisentangledAtten(win_len, heads)
        self.norm1 = nn.LayerNorm(win_len)
        self.norm2 = nn.LayerNorm(win_len)

        self.feed_forward = nn.Sequential(
            nn.Linear(win_len, forward_expansion * win_len),
            nn.ReLU(),
            nn.Linear(forward_expansion * win_len, win_len),
        )

        self.encoder_adaptor = EncoderAdapter(win_len, win_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attention = self.attention(query, key, value)
        x = self.dropout(self.norm1(attention + query))

        x = self.encoder_adaptor(x)

        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class KAD_DisformerEncoder(nn.Module):
    def __init__(
        self,
        win_len,
        num_layers,
        heads,
        device='cpu',
        dropout=0,
        forward_expansion=1,
    ):

        super(KAD_DisformerEncoder, self).__init__()
        self.win_len = win_len
        self.device = device

        self.layers = nn.ModuleList(
            [
                EncoderTransformerBlock(
                    win_len,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N = x.shape[0]
        out = self.dropout(x)

        for layer in self.layers:
            out = layer(out, out, out)

        return out


