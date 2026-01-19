import torch
import torch.nn as nn
import torch.nn.functional as F
from src.transformer.transformer import TSTransformerEncoderClassiregressor


class SeasonalDecomposition(nn.Module):
    """季节性分解模块，增强时序特征"""
    def __init__(self, seq_len=61, embed_dim=64):
        super().__init__()
        self.seq_len = seq_len
        
        # 季节性模式学习
        self.seasonal_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        # 趋势提取 - 使用不同大小的卷积核
        self.trend_convs = nn.ModuleList([
            nn.Conv1d(1, 16, kernel_size=7, padding=3),    # 短期趋势
            nn.Conv1d(1, 16, kernel_size=15, padding=7),   # 中期趋势
            nn.Conv1d(1, 16, kernel_size=31, padding=15),  # 长期趋势
        ])
        
        # 特征融合
        self.trend_fusion = nn.Linear(48, embed_dim)
        
    def forward(self, x):
        # x: [B, T, F]
        B, T, F = x.shape
        
        # 季节性增强
        seasonal_feat = self.seasonal_embedding.expand(B, -1, -1)
        
        # 趋势特征提取
        trend_features = []
        x_mean = x.mean(dim=-1, keepdim=True)  # [B, T, 1]
        
        for conv in self.trend_convs:
            # [B, T, 1] -> [B, 1, T] -> [B, 16, T] -> [B, T, 16]
            feat = conv(x_mean.transpose(1, 2)).transpose(1, 2)
            trend_features.append(feat)
        
        # 拼接趋势特征
        trend_concat = torch.cat(trend_features, dim=-1)  # [B, T, 48]
        trend_feat = self.trend_fusion(trend_concat)      # [B, T, embed_dim]
        
        # 组合特征
        enhanced_x = torch.cat([x, seasonal_feat, trend_feat], dim=-1)
        return enhanced_x


class MultiScaleTemporalConv(nn.Module):
    """多尺度时序卷积模块"""
    def __init__(self, feat_dim=10, d_model=512):
        super().__init__()
        
        # 多尺度卷积捕获不同时间窗口的模式
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(feat_dim, d_model//4, kernel_size=3, padding=1),   # 短期模式
            nn.Conv1d(feat_dim, d_model//4, kernel_size=7, padding=3),   # 中期模式  
            nn.Conv1d(feat_dim, d_model//4, kernel_size=15, padding=7),  # 长期模式
            nn.Conv1d(feat_dim, d_model//4, kernel_size=31, padding=15), # 超长期模式
        ])
        
        # 批归一化和激活
        self.bn = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: [B, T, F]
        B, T, F = x.shape
        
        # 多尺度特征提取
        multi_scale_features = []
        for conv in self.temporal_convs:
            # [B, T, F] -> [B, F, T] -> [B, d_model//4, T] -> [B, T, d_model//4]
            feat = conv(x.transpose(1, 2)).transpose(1, 2)
            multi_scale_features.append(feat)
        
        # 拼接多尺度特征
        x = torch.cat(multi_scale_features, dim=-1)  # [B, T, d_model]
        
        # 批归一化 (需要调整维度)
        x = x.transpose(1, 2)  # [B, d_model, T]
        x = self.bn(x)
        x = self.relu(x)
        x = x.transpose(1, 2)  # [B, T, d_model]
        
        return x


class EnhancedClimateTransformer(nn.Module):
    """增强的气候Transformer模块"""
    def __init__(self, feat_dim=10, d_model=512, num_layers=6, seq_len=61):
        super().__init__()
        
        # 季节性分解
        self.seasonal_decomp = SeasonalDecomposition(seq_len, embed_dim=64)
        
        # 多尺度时序卷积
        enhanced_feat_dim = feat_dim + 64 + 64  # 原始 + 季节性 + 趋势
        self.multi_scale_conv = MultiScaleTemporalConv(enhanced_feat_dim, d_model)
        
        # 注意力机制捕获长期依赖
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        
        # Transformer编码器
        self.transformer = TSTransformerEncoderClassiregressor(
            feat_dim=d_model, max_len=seq_len, d_model=d_model,
            n_heads=8, num_layers=num_layers, dim_feedforward=2048,
            num_classes=d_model, dropout=0.1, pos_encoding="fixed",
            activation="gelu", norm="BatchNorm", freeze=False
        )
        
        # 输出投影
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x: [B, 61, 10]
        B, T, F = x.shape
        
        # 季节性分解增强
        enhanced_x = self.seasonal_decomp(x)  # [B, T, F+64+64]
        
        # 多尺度特征提取
        multi_scale_x = self.multi_scale_conv(enhanced_x)  # [B, T, d_model]
        
        # 自注意力处理
        attended_x, _ = self.attention(multi_scale_x, multi_scale_x, multi_scale_x)
        
        # Transformer处理
        transformer_output = self.transformer(attended_x)  # [B, d_model]
        
        # 输出投影
        output = self.output_proj(transformer_output)
        
        return output


if __name__ == "__main__":
    # 测试代码
    model = EnhancedClimateTransformer(
        feat_dim=10, d_model=512, num_layers=6, seq_len=61
    )
    
    x = torch.randn(4, 61, 10)  # batch_size=4, seq_len=61, feat_dim=10
    out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}") 