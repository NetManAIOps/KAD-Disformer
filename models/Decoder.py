from torch import nn
from .DPM import DisentangledAtten
from .Encoder import EncoderTransformerBlock as TransformerBlock

class DecoderTransformerBlock(nn.Module):
    def __init__(self, win_len, heads, dropout=True, forward_expansion=1):
        super(DecoderTransformerBlock, self).__init__()
        self.norm = nn.LayerNorm(win_len)
        self.attention = DisentangledAtten(win_len, heads=heads)
        self.transformer_block = TransformerBlock(
            win_len, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, query):
        attention = self.attention(x, x, x)
        key = self.dropout(self.norm(attention + x))
        value = key
        out = self.transformer_block(query, key, value)
        return out



class KAD_DisformerDecoder(nn.Module):
    def __init__(
        self,
        win_len,
        num_layers,
        heads,
        device='cpu',
        dropout=0,
        forward_expansion=1,
    ):
        super(KAD_DisformerDecoder, self).__init__()
        self.device = device

        self.layers = nn.ModuleList(
            [
                DecoderTransformerBlock(win_len, heads, 
                                        dropout=dropout,
                                        forward_expansion=forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(win_len, win_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context_query):
        N = x.shape[0]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, context_query)

        out = self.fc_out(x)

        return out