from torch import nn
from .Encoder import KAD_DisformerEncoder
from .Decoder import KAD_DisformerDecoder

class KAD_Disformer(nn.Module):
    def __init__(
        self,
        win_len=100,
        num_layers=3,
        heads=8,
        dropout=0,
        device="cpu",
        forward_expansion=1,
    ):

        super(KAD_Disformer, self).__init__()

        self.encoder = KAD_DisformerEncoder(
            win_len,
            num_layers,
            heads,
            device,
            dropout,
            forward_expansion,
        )

        self.decoder = KAD_DisformerDecoder(
            win_len,
            num_layers,
            heads,
            device,
            dropout,
            forward_expansion,
        )
        self.device = device


    def forward(self, seq):
        x = self.encoder(seq)
        out = self.decoder(x, x)
        return out