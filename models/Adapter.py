from torch import nn

class EncoderAdapter(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(EncoderAdapter, self).__init__()

        self.adapter = nn.Linear(in_dim, out_dim)


    def forward(self, x):
        return self.adapter(x)



class SeriesAdapter(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(SeriesAdapter).__init__()

        self.adapter = nn.Linear(in_dim, out_dim)


    def forward(self, x):
        return self.adapter(x)