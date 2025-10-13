import torch
from torch import nn

from .Decoder import KAD_DisformerDecoder
from .Encoder import KAD_DisformerEncoder
from .Adapter import SeriesAdapter

class KAD_DisformerBlock(nn.Module):
    def __init__(
        self,
        patch_size: int = 20,
        heads=4,
        d_model: int = 64,
        dropout=0,
        forward_expansion=2,
        encoder_num_layers: int = 1,
        decoder_num_layers: int = 1,
        is_context=True,
        W_common_Q=None,
        W_personal_Q=None,
    ):
        super(KAD_DisformerBlock, self).__init__()

        self.patch_size = patch_size
        self.heads = heads
        self.d_model = d_model
        self.dropout = dropout
        self.forward_expansion = forward_expansion
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.head_dim = d_model // heads

        assert (
            self.head_dim * heads == d_model
        ), "d_model size needs to be divisible by heads"

        self.series_adapter = SeriesAdapter(input_dim=self.patch_size)
        self.embed = nn.Linear(patch_size, d_model)

        self.encoder = KAD_DisformerEncoder(
            heads=heads,
            d_model=d_model,
            dropout=dropout,
            forward_expansion=forward_expansion,
            num_layers=encoder_num_layers,
        )

        self.W_common_Q = nn.Linear(d_model, d_model, bias=True)
        self.W_common_K = nn.Linear(d_model, d_model, bias=True)
        self.W_common_V = nn.Linear(d_model, d_model, bias=True)

        self.W_personal_Q = nn.Linear(d_model, d_model, bias=True)
        self.W_personal_K = nn.Linear(d_model, d_model, bias=True)
        self.W_personal_V = nn.Linear(d_model, d_model, bias=True)

        self.decoder = KAD_DisformerDecoder(
            heads=heads,
            d_model=d_model,
            dropout=dropout,
            num_layers=decoder_num_layers,
            W_common_Q=W_common_Q if not is_context else None,
            W_personal_Q=W_personal_Q if not is_context else None,
        )

        self.output_layer = nn.Linear(d_model, patch_size)

    def get_W_Q(self):
        return self.W_common_Q, self.W_personal_Q

    def forward(self, seq):
        x = self.series_adapter(seq)
        x = self.embed(x)
        x = self.encoder(x)

        query = self.W_common_Q(x) + self.W_personal_Q(x)
        key = self.W_common_K(x) + self.W_personal_K(x)
        value = self.W_common_V(x) + self.W_personal_V(x)

        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        query = query.reshape(N, query_len, self.heads, self.head_dim)
        key = key.reshape(N, key_len, self.heads, self.head_dim)
        value = value.reshape(N, value_len, self.heads, self.head_dim)
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])

        attention = torch.softmax(energy / (self.d_model ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        encoder_out = torch.einsum("nhql,nlhd->nqhd", [attention, value]).reshape(
            N, query_len, self.d_model
        )

        out = self.decoder(encoder_out)
        out = self.output_layer(out)

        return out


class KAD_Disformer(nn.Module):
    def __init__(
        self,
        patch_size: int = 20,
        heads=4,
        d_model: int = 64,
        dropout=0,
        forward_expansion=2,
        encoder_num_layers: int = 1,
        decoder_num_layers: int = 1,
    ):
        super(KAD_Disformer, self).__init__()

        self.context_block = KAD_DisformerBlock(
            patch_size=patch_size,
            heads=heads,
            d_model=d_model,
            dropout=dropout,
            forward_expansion=forward_expansion,
            encoder_num_layers=encoder_num_layers,
            decoder_num_layers=decoder_num_layers,
            is_context=True,
        )
        self.W_common_Q, self.W_personal_Q = self.context_block.get_W_Q()

        self.history_block = KAD_DisformerBlock(
            patch_size=patch_size,
            heads=heads,
            d_model=d_model,
            dropout=dropout,
            forward_expansion=forward_expansion,
            encoder_num_layers=encoder_num_layers,
            decoder_num_layers=decoder_num_layers,
            is_context=False,
            W_common_Q=self.W_common_Q,
            W_personal_Q=self.W_personal_Q,
        )

    def forward(self, seq_context, seq_history):
        x_1 = self.context_block(seq_context)
        x_2 = self.history_block(seq_history)

        return (x_1 + x_2) / 2
