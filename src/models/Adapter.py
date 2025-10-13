import torch
from torch import nn


class EncoderAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, forward_expansion: int = 1) -> None:
        super(EncoderAdapter, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(in_dim, forward_expansion * in_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * in_dim, in_dim),
        )

        self.adapter = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


class SeriesDecomposition(nn.Module):
    def __init__(self, input_dim: int):
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


if __name__ == '__main__':
    encoder_adapter = EncoderAdapter(100, 10)
    x = torch.randn(10, 100, 100)
    output = encoder_adapter(x)
    print(output.shape)

    series_decomposition = SeriesDecomposition(100)
    trend, seasonal = series_decomposition(x)
    print(f"trend: {trend.shape}, seasonal: {seasonal.shape}")

    series_adapter = SeriesAdapter(100)
    output = series_adapter(x)
    print(output.shape)
