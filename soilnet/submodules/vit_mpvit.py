import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbed(nn.Module):
    """将图像分割为patch并进行嵌入"""
    def __init__(self, img_size=224, patch_size=16, in_chans=12, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class MultiPathAttention(nn.Module):
    """多路径注意力机制，处理多尺度特征"""
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


class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiScaleFusion(nn.Module):
    """多尺度特征融合模块"""
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        self.fusion = nn.Conv2d(dim * 3, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, H, W):
        # 输入 x: [B, N, C]
        B, N, C = x.shape
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        
        # 多尺度卷积
        x1 = self.conv1(x_2d)
        x2 = self.conv2(x_2d)
        x3 = self.conv3(x_2d)
        
        # 融合
        x_cat = torch.cat([x1, x2, x3], dim=1)
        x_fused = self.fusion(x_cat)
        
        # 转回序列形式
        x_out = x_fused.flatten(2).transpose(1, 2)  # [B, N, C]
        x_out = self.norm(x_out)
        
        return x_out


class MPViTBlock(nn.Module):
    """MPViT的基本构建块"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_paths=3):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiPathAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                      attn_drop=attn_drop, proj_drop=drop, num_paths=num_paths)
        
        self.drop_path = nn.Identity() if drop_path == 0. else DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        # 多尺度特征融合
        self.msf = MultiScaleFusion(dim)
        self.norm3 = norm_layer(dim)
        
    def forward(self, x, H, W):
        # 多路径自注意力
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        # 多尺度特征融合
        x = x + self.drop_path(self.msf(self.norm3(x), H, W))
        
        return x


# -----------------------------
# MPViT: Multi-Path Vision Transformer
# -----------------------------
class MPViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=12, num_classes=1, 
                 embed_dim=768, depth=6, num_heads=8, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., num_paths=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim
        self.patch_size = patch_size
        
        # Patch嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # 位置嵌入
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 随机深度衰减
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer块
        self.blocks = nn.ModuleList([
            MPViTBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], num_paths=num_paths
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # 添加位置嵌入
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 计算patch的高度和宽度
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        
        # 通过Transformer块
        for blk in self.blocks:
            x = blk(x, H_patch, W_patch)
            
        x = self.norm(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        # 全局平均池化
        x = x.mean(dim=1)  # [B, embed_dim]
        x = self.head(x)  # [B, num_classes]
        return x


# 定义DropPath (随机深度)
class DropPath(nn.Module):
    """随机深度衰减"""
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二值化
        output = x.div(keep_prob) * random_tensor
        return output


if __name__ == "__main__":
    # 测试代码
    model = MPViT(
        img_size=128,
        patch_size=16,
        in_chans=12,
        num_classes=1,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4,
        num_paths=3
    )
    
    x = torch.randn(2, 12, 128, 128)
    out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}") 