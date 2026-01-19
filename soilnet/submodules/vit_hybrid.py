import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -----------------------------
# Overlap Patch Embedding (from MPViT)
# -----------------------------
class OverlapPatchEmbed(nn.Module):
    """重叠Patch嵌入，使用更小的步长来保留更多空间信息"""
    def __init__(self, img_size=224, patch_size=16, stride=14, in_chans=12, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        
        # 计算特征图大小
        self.H = (img_size - patch_size) // stride + 1
        self.W = (img_size - patch_size) // stride + 1
        self.num_patches = self.H * self.W
        
        # 重叠卷积
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=stride,
            padding=(patch_size - stride) // 2
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H', W']
        _, _, H, W = x.shape
        
        x = x.flatten(2).transpose(1, 2)  # [B, H'*W', embed_dim]
        x = self.norm(x)
        
        return x, H, W


# -----------------------------
# CNN Extractor (from ViT-CoMer)
# -----------------------------
class CNNExtractor(nn.Module):
    def __init__(self, in_chans=12, embed_dim=768):
        super().__init__()
        self.stages = nn.ModuleList([
            # Stage 1: 1/4 resolution
            nn.Sequential(
                nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ),
            # Stage 2: 1/8 resolution
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            # Stage 3: 1/16 resolution
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        ])
        
        # 投影层，将CNN特征映射到Transformer的维度
        self.projs = nn.ModuleList([
            nn.Conv2d(64, embed_dim, kernel_size=1),
            nn.Conv2d(128, embed_dim, kernel_size=1),
            nn.Conv2d(256, embed_dim, kernel_size=1)
        ])
        
    def forward(self, x):
        # x: [B, C, H, W]
        features = []
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(self.projs[i](x))
            
        return features  # 返回多尺度特征 [1/4, 1/8, 1/16]


# -----------------------------
# CTI Module: CNN → ViT Token Fusion (from ViT-CoMer)
# -----------------------------
class CTI_toV(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, vit_tokens, cnn_feat):
        # vit_tokens: [B, N, C]
        # cnn_feat: [B, C, H, W]
        
        B, C, H, W = cnn_feat.shape
        cnn_feat = cnn_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # 确保形状匹配
        if vit_tokens.shape[1] != cnn_feat.shape[1]:
            cnn_feat = F.interpolate(
                cnn_feat.transpose(1, 2).view(B, C, H, W),
                size=(int((vit_tokens.shape[1])**0.5), int((vit_tokens.shape[1])**0.5)),
                mode='bilinear'
            ).flatten(2).transpose(1, 2)
            
        # 融合CNN和ViT特征
        x = vit_tokens + self.fusion(cnn_feat)
        return x


# -----------------------------
# Multi-Path Attention (from MPViT)
# -----------------------------
class MultiPathAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., num_paths=3):
        super().__init__()
        self.num_heads = num_heads
        self.num_paths = num_paths
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # 为每个路径创建独立的QKV投影
        self.qkv_projs = nn.ModuleList([
            nn.Linear(dim, dim * 3, bias=qkv_bias) 
            for _ in range(num_paths)
        ])
        
        self.attn_drop = nn.Dropout(attn_drop)
        
        # 为每个路径创建独立的输出投影
        self.projs = nn.ModuleList([
            nn.Linear(dim, dim) 
            for _ in range(num_paths)
        ])
        
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 路径融合层
        self.path_fusion = nn.Linear(dim * num_paths, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        
        path_outputs = []
        for i in range(self.num_paths):
            qkv = self.qkv_projs[i](x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
            
            # 注意力计算
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            
            # 注意力应用
            x_path = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
            x_path = self.projs[i](x_path)
            x_path = self.proj_drop(x_path)
            
            path_outputs.append(x_path)
        
        # 融合多路径输出
        x_multi = torch.cat(path_outputs, dim=-1)  # [B, N, C*num_paths]
        x_fused = self.path_fusion(x_multi)  # [B, N, C]
        
        return x_fused


# -----------------------------
# MLP Block
# -----------------------------
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


# -----------------------------
# Hybrid Transformer Block
# -----------------------------
class HybridBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, 
                 drop=0., attn_drop=0., num_paths=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiPathAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, num_paths=num_paths
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------
# Hybrid ViT Stage
# -----------------------------
class HybridStage(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=True, 
                 drop=0., attn_drop=0., num_paths=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            HybridBlock(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                num_paths=num_paths
            )
            for _ in range(depth)
        ])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# -----------------------------
# HybridViT: 结合ViT-CoMer和MPViT的优点
# -----------------------------
class HybridViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=12, num_classes=1, 
                 embed_dim=768, depths=[2, 2, 6], num_heads=8, mlp_ratio=4., 
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., num_paths=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim
        
        # Overlap Patch Embedding
        self.patch_embed = OverlapPatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=patch_size-2,
            in_chans=in_chans, embed_dim=embed_dim
        )
        
        # CNN特征提取器
        self.cnn = CNNExtractor(in_chans=in_chans, embed_dim=embed_dim)
        
        # CNN到Transformer的交互模块
        self.cti_modules = nn.ModuleList([
            CTI_toV(embed_dim) for _ in range(len(depths))
        ])
        
        # 位置编码
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 多阶段Transformer编码器
        self.stages = nn.ModuleList()
        for i, depth in enumerate(depths):
            self.stages.append(
                HybridStage(
                    dim=embed_dim, depth=depth, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    num_paths=num_paths
                )
            )
            
        # 最终层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化权重
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward_features(self, x):
        # 提取CNN特征
        cnn_features = self.cnn(x)  # 多尺度特征 [1/4, 1/8, 1/16]
        
        # Patch嵌入
        x, H, W = self.patch_embed(x)  # [B, N, C]
        
        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 多阶段Transformer处理
        for i, (stage, cti) in enumerate(zip(self.stages, self.cti_modules)):
            # 在每个阶段前融合CNN特征
            if i < len(cnn_features):
                x = cti(x, cnn_features[i])
            x = stage(x)
            
        x = self.norm(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        # 全局平均池化
        x = x.mean(dim=1)
        x = self.head(x)
        return x


if __name__ == "__main__":
    # 测试代码
    model = HybridViT(
        img_size=128,
        patch_size=16,
        in_chans=12,
        num_classes=1,
        embed_dim=384,
        depths=[2, 2, 6],
        num_heads=6,
        mlp_ratio=4,
        num_paths=3
    )
    
    x = torch.randn(2, 12, 128, 128)
    out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}") 