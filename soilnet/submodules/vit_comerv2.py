import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -----------------------------
# Patch Embedding for ViT
# -----------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_chans=12, embed_dim=768, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):  # B x C x H x W
        x = self.proj(x)  # B x embed_dim x H/P x W/P
        x = x.flatten(2).transpose(1, 2)  # B x N x embed_dim
        return x


# -----------------------------
# Lightweight CNN Encoder
# -----------------------------
class CNNExtractor(nn.Module):
    def __init__(self, in_chans=12, out_chans=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),  # 降采样
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans), nn.ReLU(),
        )

    def forward(self, x):  # B x C x H x W
        return self.conv(x)  # B x out_chans x H/2 x W/2


# -----------------------------
# Multi-head Self-Attention
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
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each: B x N x heads x dim_head

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(2, 1).reshape(B, N, C)
        return self.proj(self.dropout(x))


# -----------------------------
# MLP for ViT block
# -----------------------------
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

    def forward(self, x):
        return self.fc(x)


# -----------------------------
# Basic ViT Block
# -----------------------------
class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------
# CTI Module: CNN → ViT Token Fusion
# -----------------------------
class CTI_toV(nn.Module):
    def __init__(self, cnn_channels, embed_dim):
        super().__init__()
        self.conv1x1 = nn.Conv2d(cnn_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的融合权重

    def forward(self, cnn_feat, vit_tokens, H, W):
        # cnn_feat: B x C x H x W → B x embed_dim x H x W
        B, _, _, _ = cnn_feat.shape
        B, N, D = vit_tokens.shape
        
        # 确保CNN特征与ViT token尺寸匹配
        if cnn_feat.shape[2] != H or cnn_feat.shape[3] != W:
            cnn_feat = F.interpolate(cnn_feat, size=(H, W), mode='bilinear', align_corners=False)
            
        fused = self.conv1x1(cnn_feat)
        fused = fused.flatten(2).transpose(1, 2)  # B x (H*W) x embed_dim
        
        # 确保fused与vit_tokens有相同数量的token
        if fused.shape[1] != N:
            fused = F.interpolate(fused.transpose(1, 2).view(B, D, int(N**0.5), int(N**0.5)), 
                                size=(int(N**0.5), int(N**0.5)), mode='bilinear', align_corners=False)
            fused = fused.flatten(2).transpose(1, 2)
            
        fused = self.norm(fused)
        return vit_tokens + self.gamma * fused


# -----------------------------
# CTI Module: ViT Token → CNN Feature Fusion (改进版)
# -----------------------------
class CTI_toC(nn.Module):
    def __init__(self, embed_dim, cnn_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, cnn_channels),
            nn.Dropout(0.2)  # 增加dropout防止过拟合
        )
        self.norm = nn.BatchNorm2d(cnn_channels)
        # 初始化为更小的值，减少初始阶段的影响
        self.gamma = nn.Parameter(torch.zeros(1) * 0.01)
        # 添加门控机制
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, vit_tokens, cnn_feat):
        # vit_tokens: B x N x D
        # cnn_feat: B x C x H x W
        B, N, D = vit_tokens.shape
        B, C, H, W = cnn_feat.shape
        
        # 计算门控值，决定是否使用ViT信息
        gate_value = self.gate(vit_tokens.mean(dim=1)).view(B, 1, 1, 1)
        
        # 投影ViT tokens到CNN通道维度
        vit_proj = self.proj(vit_tokens)  # B x N x C
        
        # 重塑为空间特征
        vit_spatial = vit_proj.transpose(1, 2).view(B, C, int(N**0.5), int(N**0.5))  # B x C x sqrt(N) x sqrt(N)
        
        # 确保尺寸匹配
        if vit_spatial.shape[2] != H or vit_spatial.shape[3] != W:
            vit_spatial = F.interpolate(vit_spatial, size=(H, W), mode='bilinear', align_corners=False)
        
        # 应用归一化
        vit_spatial = self.norm(vit_spatial)
        
        # 融合 - 使用门控机制和gamma参数共同控制
        return cnn_feat + self.gamma * gate_value * vit_spatial


# -----------------------------
# ViT-CoMerV2: Enhanced bidirectional CNN-ViT interaction
# -----------------------------
class ViTCoMerV2(nn.Module):
    def __init__(self, in_chans=12, num_classes=1, patch_size=16, embed_dim=768, depth=6, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)
        self.cnn = CNNExtractor(in_chans, out_chans=embed_dim)
        
        # 双向CTI模块
        self.cti_to_v = CTI_toV(embed_dim, embed_dim)
        self.cti_to_c = CTI_toC(embed_dim, embed_dim)
        
        # 保存patch_size以便在forward中使用
        self.patch_size = patch_size
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, (224//patch_size)**2, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer块
        self.blocks = nn.ModuleList([
            Block(embed_dim, heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):  # B x C x H x W
        B, C, H, W = x.shape
        
        # Patch嵌入
        x_patch = self.patch_embed(x)  # B x N x D
        
        # 添加位置编码
        if x_patch.shape[1] != self.pos_embed.shape[1]:
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2).reshape(1, -1, int(self.pos_embed.shape[1]**0.5), int(self.pos_embed.shape[1]**0.5)),
                size=(int(x_patch.shape[1]**0.5), int(x_patch.shape[1]**0.5)),
                mode='bilinear'
            ).flatten(2).transpose(1, 2)
        else:
            pos_embed = self.pos_embed
            
        x_patch = x_patch + pos_embed
        x_patch = self.pos_drop(x_patch)
        
        # CNN特征提取
        cnn_feat = self.cnn(x)  # B x embed_dim x H/2 x W/2
        
        # 计算Patch尺寸
        Hp, Wp = H // self.patch_size, W // self.patch_size
        
        # 双向交互 - 第一次
        x_patch = self.cti_to_v(cnn_feat, x_patch, Hp, Wp)  # CNN到ViT
        cnn_feat = self.cti_to_c(x_patch, cnn_feat)  # ViT到CNN (现在影响更小)
        
        # Transformer处理
        for i, blk in enumerate(self.blocks):
            x_patch = blk(x_patch)
            
            # 只在最后一层进行第二次双向交互，减少干扰
            if i == len(self.blocks) - 1:
                x_patch = self.cti_to_v(cnn_feat, x_patch, Hp, Wp)
                # 不再进行第二次ViT到CNN的交互
                # cnn_feat = self.cti_to_c(x_patch, cnn_feat)

        # 最终归一化
        x_out = self.norm(x_patch)
        
        # 全局平均池化
        x_out = x_out.mean(dim=1)  # B x D
        
        # 分类头
        return self.head(x_out)


if __name__ == '__main__':
    model = ViTCoMerV2(
        in_chans=12,
        num_classes=1,
        patch_size=16,
        embed_dim=384,
        depth=6,
        heads=6,
        mlp_ratio=4.0,
        dropout=0.1
    )
    dummy = torch.randn(4, 12, 128, 128)  # batch x channel x H x W
    out = model(dummy)
    print(f"Input shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}") 