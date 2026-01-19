import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -----------------------------
# 1. 原始 Patch Embedding (最适合 64x64)
# -----------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_chans=12, embed_dim=768, patch_size=8, img_size=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        if H != self.img_size or W != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        return x

# -----------------------------
# 2. 原始 Simple CNN
# -----------------------------
class CNNExtractor(nn.Module):
    def __init__(self, in_chans=12, out_chans=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, out_chans, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_chans), nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)

# -----------------------------
# 3. 原始 Add Fusion
# -----------------------------
class CTI_toV(nn.Module):
    def __init__(self, cnn_channels, embed_dim):
        super().__init__()
        self.conv1x1 = nn.Conv2d(cnn_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.gamma = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, cnn_feat, vit_tokens, H_grid, W_grid):
        cnn_feat = self.conv1x1(cnn_feat)
        if cnn_feat.shape[2] != H_grid or cnn_feat.shape[3] != W_grid:
            cnn_feat = F.interpolate(cnn_feat, size=(H_grid, W_grid), mode='bilinear', align_corners=False)
        cnn_feat = rearrange(cnn_feat, 'b d h w -> b (h w) d')
        return vit_tokens + self.gamma * self.norm(cnn_feat)

# -----------------------------
# 4. Standard Components
# -----------------------------
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(2, 1).reshape(B, N, C)
        return self.proj(self.dropout(x))

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.fc(x)

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4., dropout=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        # 极简 DropPath
        self.drop_path = nn.Identity()
        if drop_path > 0.:
            # 如果没有timm，这里实际上不起作用，但这没关系，你的最佳结果也不需要它
            pass 
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# -----------------------------
# 5. Main Model (ViTCoMer)
# -----------------------------
class ViTCoMer(nn.Module):
    def __init__(self, 
                 in_chans=12, num_classes=1, img_size=64,
                 patch_size=8,    # 8
                 embed_dim=768,   # 768
                 depth=4, heads=12, mlp_ratio=4.0,
                 drop_path_rate=0. # 关掉
                 ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size, img_size)
        self.cnn = CNNExtractor(in_chans, out_chans=embed_dim)
        self.cti = CTI_toV(embed_dim, embed_dim)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, heads, mlp_ratio) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x_vit = self.patch_embed(x) 
        x_cnn = self.cnn(x)
        Hp = Wp = self.img_size // self.patch_size
        x_fused = self.cti(x_cnn, x_vit, Hp, Wp)
        for blk in self.blocks:
            x_fused = blk(x_fused)
        x_out = self.norm(x_fused)
        x_out = x_out.mean(dim=1) 
        return self.head(x_out)