import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """交叉模态注意力模块"""
    def __init__(self, vit_dim, climate_dim, fusion_dim):
        super().__init__()
        
        # 交叉注意力: ViT -> Climate
        self.vit_to_climate = nn.MultiheadAttention(
            embed_dim=climate_dim, num_heads=8, 
            kdim=vit_dim, vdim=vit_dim, batch_first=True
        )
        
        # 交叉注意力: Climate -> ViT  
        self.climate_to_vit = nn.MultiheadAttention(
            embed_dim=vit_dim, num_heads=8,
            kdim=climate_dim, vdim=climate_dim, batch_first=True
        )
        
        # 特征融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(vit_dim + climate_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 残差连接
        self.vit_residual = nn.Linear(vit_dim, fusion_dim)
        self.climate_residual = nn.Linear(climate_dim, fusion_dim)
        
    def forward(self, vit_features, climate_features):
        # vit_features: [B, N, vit_dim]
        # climate_features: [B, T, climate_dim]
        
        B, N, vit_dim = vit_features.shape
        B, T, climate_dim = climate_features.shape
        
        # 确保序列长度一致 - 使用全局平均池化
        if N != T:
            if N > T:
                # 如果ViT特征更长，对气候特征进行插值
                climate_features = F.interpolate(
                    climate_features.transpose(1, 2),  # [B, climate_dim, T]
                    size=N, mode='linear', align_corners=False
                ).transpose(1, 2)  # [B, N, climate_dim]
            else:
                # 如果气候特征更长，对ViT特征进行插值
                vit_features = F.interpolate(
                    vit_features.transpose(1, 2),  # [B, vit_dim, N]
                    size=T, mode='linear', align_corners=False
                ).transpose(1, 2)  # [B, T, vit_dim]
                N = T
        
        # ViT特征影响气候特征
        climate_enhanced, _ = self.vit_to_climate(
            climate_features, vit_features, vit_features
        )
        
        # 气候特征影响ViT特征
        vit_enhanced, _ = self.climate_to_vit(
            vit_features, climate_features, climate_features
        )
        
        # 残差连接
        vit_residual = self.vit_residual(vit_features)
        climate_residual = self.climate_residual(climate_features)
        
        # 融合
        fused_features = torch.cat([vit_enhanced, climate_enhanced], dim=-1)
        output = self.fusion_layer(fused_features)
        
        # 添加残差
        output = output + vit_residual + climate_residual
        
        return output


class GatedFusion(nn.Module):
    """门控融合机制"""
    def __init__(self, vit_dim, climate_dim, fusion_dim):
        super().__init__()
        
        self.vit_proj = nn.Linear(vit_dim, fusion_dim)
        self.climate_proj = nn.Linear(climate_dim, fusion_dim)
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(vit_dim + climate_dim, fusion_dim),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, vit_features, climate_features):
        # 投影到相同维度
        vit_proj = self.vit_proj(vit_features)
        climate_proj = self.climate_proj(climate_features)
        
        # 确保序列长度一致
        B, N, _ = vit_features.shape
        B, T, _ = climate_features.shape
        
        if N != T:
            if N > T:
                # 如果ViT特征更长，对气候特征进行插值
                climate_features = F.interpolate(
                    climate_features.transpose(1, 2),  # [B, climate_dim, T]
                    size=N, mode='linear', align_corners=False
                ).transpose(1, 2)  # [B, N, climate_dim]
                climate_proj = self.climate_proj(climate_features)
            else:
                # 如果气候特征更长，对ViT特征进行插值
                vit_features = F.interpolate(
                    vit_features.transpose(1, 2),  # [B, vit_dim, N]
                    size=T, mode='linear', align_corners=False
                ).transpose(1, 2)  # [B, T, vit_dim]
                vit_proj = self.vit_proj(vit_features)
                N = T
        
        # 计算门控权重
        gate_input = torch.cat([vit_features, climate_features], dim=-1)
        gate_weights = self.gate(gate_input)
        
        # 加权融合
        weighted_vit = gate_weights * vit_proj
        weighted_climate = (1 - gate_weights) * climate_proj
        
        # 最终融合
        fused = self.fusion(torch.cat([weighted_vit, weighted_climate], dim=-1))
        return fused


class HierarchicalFusion(nn.Module):
    """分层融合策略"""
    def __init__(self, vit_dim, climate_dim, fusion_dim=256):
        super().__init__()
        
        # 早期融合: 特征级交互
        self.early_fusion = CrossModalAttention(vit_dim, climate_dim, fusion_dim)
        
        # 中期融合: 注意力级交互
        self.mid_fusion = nn.MultiheadAttention(fusion_dim, num_heads=8, batch_first=True)
        
        # 晚期融合: 决策级融合
        self.late_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 输出投影
        self.output_proj = nn.Linear(fusion_dim, fusion_dim)
        
    def forward(self, vit_features, climate_features):
        # 早期融合
        early_fused = self.early_fusion(vit_features, climate_features)
        
        # 中期融合
        mid_fused, _ = self.mid_fusion(early_fused, early_fused, early_fused)
        
        # 晚期融合
        final_fused = self.late_fusion(torch.cat([early_fused, mid_fused], dim=-1))
        
        # 输出投影
        output = self.output_proj(final_fused)
        
        return output


if __name__ == "__main__":
    # 测试代码
    vit_dim = 384
    climate_dim = 128
    fusion_dim = 256
    
    # 测试交叉模态注意力
    cross_modal = CrossModalAttention(vit_dim, climate_dim, fusion_dim)
    vit_feat = torch.randn(4, 64, vit_dim)  # [B, N, vit_dim]
    climate_feat = torch.randn(4, 61, climate_dim)  # [B, T, climate_dim]
    
    out = cross_modal(vit_feat, climate_feat)
    print(f"CrossModalAttention output shape: {out.shape}")
    
    # 测试门控融合
    gated = GatedFusion(vit_dim, climate_dim, fusion_dim)
    out = gated(vit_feat, climate_feat)
    print(f"GatedFusion output shape: {out.shape}")
    
    # 测试分层融合
    hierarchical = HierarchicalFusion(vit_dim, climate_dim, fusion_dim)
    out = hierarchical(vit_feat, climate_feat)
    print(f"HierarchicalFusion output shape: {out.shape}") 