import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralSE(nn.Module):
    """Squeeze-Excitation on spectral (channel) dimension.
    Input:  [B, C, H, W] -> Output: [B, C, H, W]
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w


class BandCorrelationModule(nn.Module):
    """Learnable band correlation attention using covariance-derived attention.
    Input: [B, C, H, W] -> Output: [B, C, H, W] (with residual)
    """
    def __init__(self, channels: int, proj: bool = True):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False) if proj else nn.Identity()
        self.out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_proj = self.proj(x)
        x_flat = x_proj.view(b, c, -1)  # [B, C, N]
        mean = x_flat.mean(dim=2, keepdim=True)
        xc = x_flat - mean
        # covariance-like matrix [B, C, C]
        cov = torch.bmm(xc, xc.transpose(1, 2)) / (h * w + 1e-6)
        attn = F.softmax(cov, dim=-1)
        mixed = torch.bmm(attn, x_flat).view(b, c, h, w)
        y = self.out(mixed)
        return x + y


class SpectralWiseTransformer(nn.Module):
    """Transformer operating across bands.
    We pool spatially to get per-band descriptors then attend across C.
    Input: [B, C, H, W] -> Output: [B, C, H, W] (channel reweighting)
    """
    def __init__(self, channels: int, d_model: int = 128, num_layers: int = 2, n_heads: int = 4, reduction: int = 4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.in_proj = nn.Linear(1, d_model)
        hidden = max(1, d_model // reduction)
        self.out_mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Sigmoid()
        )
        self.scale = nn.Parameter(torch.ones(1))

        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # GAP per band: [B, C, 1]
        band_desc = x.view(b, c, -1).mean(dim=2, keepdim=True)
        z = self.in_proj(band_desc)  # [B, C, d_model]
        z = self.encoder(z)          # [B, C, d_model]
        w = self.out_mlp(z).squeeze(-1)  # [B, C]
        w = w.view(b, c, 1, 1)
        return x * (1 + self.scale * w)


class SpectralEnhancer(nn.Module):
    """Wrapper to combine spectral modules.
    type: 'se' | 'corr' | 'transformer' | 'hybrid'
    """
    def __init__(self, channels: int, spectral_type: str = 'se'):
        super().__init__()
        spectral_type = spectral_type.lower()
        self.spectral_type = spectral_type
        if spectral_type == 'se':
            self.module = SpectralSE(channels)
        elif spectral_type == 'corr':
            self.module = BandCorrelationModule(channels)
        elif spectral_type == 'transformer':
            self.module = SpectralWiseTransformer(channels)
        elif spectral_type == 'hybrid':
            self.se = SpectralSE(channels)
            self.corr = BandCorrelationModule(channels)
            self.trans = SpectralWiseTransformer(channels)
        else:
            raise ValueError("Invalid spectral_type. Choose from 'se', 'corr', 'transformer', 'hybrid'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spectral_type == 'hybrid':
            x = self.se(x)
            x = self.corr(x)
            x = self.trans(x)
            return x
        return self.module(x)










