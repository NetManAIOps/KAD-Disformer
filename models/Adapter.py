from torch import nn

class EncoderAdapter(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(EncoderAdapter, self).__init__()

        self.adapter = nn.Linear(in_dim, out_dim)


    def forward(self, x):
        return self.adapter(x)



class SeriesDecomposition(nn.Module):
    def __init__(self, input_dim):
        super(SeriesDecomposition, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(input_dim)
        
    def forward(self, x):
        trend_part = self.avg_pool(x)
        seasonal_part = x - trend_part
        
        return trend_part, seasonal_part


class SeriesAdapter(nn.Module):
    def __init__(self, input_dim):
        super(SeriesAdapter, self).__init__()
        
        self.decomp = SeriesDecomposition(input_dim)

        self.season = nn.Linear(input_dim, input_dim)
        self.trend = nn.Linear(input_dim, input_dim)

        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        

    def forward(self, x):
        x_trend_part, x_seasonal_part = self.decomp(x)

        trend_part = self.trend(x_trend_part)
        seasonal_part = self.season(x_seasonal_part)

        x_trend = self.layer_norm1(x_trend_part + trend_part)
        x_season = self.layer_norm2(x_seasonal_part + seasonal_part)

        
        output = x_trend + x_season
        
        return output
