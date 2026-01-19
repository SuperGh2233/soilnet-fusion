"""
RegionEmbedding (RE) 模块实现
支持区域自适应学习，让模型自动学习不同地理区域的特征偏移
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, Optional
import pandas as pd


class RegionEmbedding(nn.Module):
    """
    区域嵌入模块，为不同地理区域学习可学习的特征偏移
    """
    def __init__(self, num_regions: int, embed_dim: int, dropout: float = 0.1):
        """
        Args:
            num_regions: 区域数量
            embed_dim: 嵌入维度，需要与backbone输出维度匹配
            dropout: dropout率
        """
        super().__init__()
        self.num_regions = num_regions
        self.embed_dim = embed_dim
        
        # 区域嵌入表
        self.region_embedding = nn.Embedding(num_regions, embed_dim)
        
        # 可选的区域门控机制
        self.region_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化：增大标准差，让区域特征在训练初期就有一定影响
        nn.init.normal_(self.region_embedding.weight, mean=0, std=0.1)
    
    def forward(self, region_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            region_ids: 区域ID [B] 或 [B, 1]
        Returns:
            region_features: 区域特征 [B, embed_dim]
        """
        if region_ids.dim() > 1:
            region_ids = region_ids.squeeze(-1)
        
        # 确保region_ids在有效范围内
        region_ids = torch.clamp(region_ids, 0, self.num_regions - 1)
        
        # 获取区域嵌入
        region_emb = self.region_embedding(region_ids)  # [B, embed_dim]
        
        # 应用门控机制
        gate = self.region_gate(region_emb)
        region_emb = region_emb * gate
        
        return self.dropout(region_emb)


class RegionalFusion(nn.Module):
    """
    区域感知的特征融合模块
    """
    def __init__(self, 
                 image_dim: int, 
                 climate_dim: int, 
                 region_dim: int,
                 hidden_dim: int = 512,
                 fusion_type: str = 'attention'):
        super().__init__()
        self.fusion_type = fusion_type
        
        # 增强区域嵌入处理：添加更强的投影层
        self.region_enhance = nn.Sequential(
            nn.Linear(region_dim, region_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(region_dim * 2, region_dim),
            nn.LayerNorm(region_dim)
        )
        
        if fusion_type == 'concat':
            # 增强的concat融合
            self.image_proj = nn.Sequential(
                nn.Linear(image_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.climate_proj = nn.Sequential(
                nn.Linear(climate_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.region_proj = nn.Sequential(
                nn.Linear(region_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim // 2 * 3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        elif fusion_type == 'attention':
            # 增强的注意力融合
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim, 
                num_heads=8, 
                dropout=0.1,
                batch_first=True
            )
            self.image_proj = nn.Sequential(
                nn.Linear(image_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.climate_proj = nn.Sequential(
                nn.Linear(climate_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.region_proj = nn.Sequential(
                nn.Linear(region_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.norm = nn.LayerNorm(hidden_dim)
        elif fusion_type == 'gated':
            # 增强的门控融合
            self.region_gate = nn.Sequential(
                nn.Linear(region_dim, hidden_dim),
                nn.Sigmoid()
            )
            self.image_proj = nn.Linear(image_dim, hidden_dim)
            self.climate_proj = nn.Linear(climate_dim, hidden_dim)
            self.final_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
    
    def forward(self, image_feat: torch.Tensor, 
                climate_feat: torch.Tensor, 
                region_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_feat: 图像特征 [B, image_dim]
            climate_feat: 气候特征 [B, climate_dim] 
            region_feat: 区域特征 [B, region_dim]
        Returns:
            fused_feat: 融合特征 [B, hidden_dim]
        """
        # 增强区域特征
        enhanced_region_feat = self.region_enhance(region_feat)
        
        if self.fusion_type == 'concat':
            # 使用增强的区域特征进行投影和融合
            img_proj = self.image_proj(image_feat)
            climate_proj = self.climate_proj(climate_feat)
            region_proj = self.region_proj(enhanced_region_feat)
            
            # 拼接并融合
            fused = torch.cat([img_proj, climate_proj, region_proj], dim=1)
            return self.fusion(fused)
        
        elif self.fusion_type == 'attention':
            # 投影到相同维度
            img_proj = self.image_proj(image_feat).unsqueeze(1)  # [B, 1, hidden_dim]
            climate_proj = self.climate_proj(climate_feat).unsqueeze(1)  # [B, 1, hidden_dim]
            region_proj = self.region_proj(enhanced_region_feat).unsqueeze(1)  # [B, 1, hidden_dim]
            
            # 拼接为序列
            features = torch.cat([img_proj, climate_proj, region_proj], dim=1)  # [B, 3, hidden_dim]
            
            # 自注意力
            attn_out, _ = self.attention(features, features, features)
            attn_out = self.norm(attn_out)
            
            # 全局平均池化
            return attn_out.mean(dim=1)  # [B, hidden_dim]
        
        elif self.fusion_type == 'gated':
            # 增强的区域门控
            region_gate = self.region_gate(enhanced_region_feat)
            
            # 投影特征
            img_proj = self.image_proj(image_feat)
            climate_proj = self.climate_proj(climate_feat)
            
            # 应用区域门控
            gated_img = img_proj * region_gate
            gated_climate = climate_proj * region_gate
            
            # 最终融合
            return self.final_fusion(torch.cat([gated_img, gated_climate], dim=1))


class RegionAssigner:
    """
    区域分配器，支持多种区域划分策略
    """
    def __init__(self, strategy: str = 'grid', **kwargs):
        self.strategy = strategy
        self.kwargs = kwargs
        
    def assign_regions(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        根据经纬度分配区域ID
        """
        if self.strategy == 'grid':
            return self._grid_assignment(lat, lon)
        elif self.strategy == 'geographic':
            return self._geographic_assignment(lat, lon)
        elif self.strategy == 'administrative':
            return self._administrative_assignment(lat, lon)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _grid_assignment(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """经纬度网格划分"""
        n_lat = self.kwargs.get('n_lat', 4)
        n_lon = self.kwargs.get('n_lon', 4)
        
        # 中国大致范围
        lat_min, lat_max = self.kwargs.get('lat_range', (18, 54))
        lon_min, lon_max = self.kwargs.get('lon_range', (73, 135))
        
        lat_bins = np.linspace(lat_min, lat_max, n_lat + 1)
        lon_bins = np.linspace(lon_min, lon_max, n_lon + 1)
        
        lat_id = np.digitize(lat, lat_bins) - 1
        lon_id = np.digitize(lon, lon_bins) - 1
        
        # 确保在有效范围内
        lat_id = np.clip(lat_id, 0, n_lat - 1)
        lon_id = np.clip(lon_id, 0, n_lon - 1)
        
        region_id = lat_id * n_lon + lon_id
        return region_id
    
    def _geographic_assignment(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """基于中国自然地理分区划分"""
        region_id = np.zeros_like(lat, dtype=int)
        
        # 东北地区 (Northeast)
        mask = (lat >= 40) & (lat <= 53) & (lon >= 118) & (lon <= 135)
        region_id[mask] = 0
        
        # 华北地区 (North China) 
        mask = (lat >= 34) & (lat <= 42) & (lon >= 110) & (lon <= 120)
        region_id[mask] = 1
        
        # 西北地区 (Northwest)
        mask = (lat >= 35) & (lat <= 50) & (lon >= 75) & (lon <= 110)
        region_id[mask] = 2
        
        # 华南地区 (South China)
        mask = (lat >= 18) & (lat <= 30) & (lon >= 105) & (lon <= 120)
        region_id[mask] = 3
        
        # 西南地区 (Southwest)
        mask = (lat >= 22) & (lat <= 32) & (lon >= 97) & (lon <= 107)
        region_id[mask] = 4
        
        # 其他地区 (Other)
        mask = region_id == 0
        region_id[mask] = 5
        
        return region_id
    
    def _administrative_assignment(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """基于行政区划划分（简化版）"""
        # 这里可以集成更详细的行政区划数据
        # 暂时使用简化的省级划分
        return self._grid_assignment(lat, lon)  # 回退到网格划分


def create_region_assigner(strategy: str = 'grid', **kwargs) -> RegionAssigner:
    """
    创建区域分配器
    """
    return RegionAssigner(strategy=strategy, **kwargs)


def analyze_regional_distribution(region_ids: np.ndarray, 
                                 lat: np.ndarray, 
                                 lon: np.ndarray,
                                 save_path: Optional[str] = None) -> dict:
    """
    分析区域分布情况
    """
    import matplotlib.pyplot as plt
    
    unique_regions = np.unique(region_ids)
    n_regions = len(unique_regions)
    
    # 统计每个区域的样本数量
    region_counts = {}
    for region_id in unique_regions:
        mask = region_ids == region_id
        region_counts[region_id] = {
            'count': np.sum(mask),
            'lat_mean': np.mean(lat[mask]),
            'lon_mean': np.mean(lon[mask]),
            'lat_std': np.std(lat[mask]),
            'lon_std': np.std(lon[mask])
        }
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 散点图
    scatter = axes[0].scatter(lon, lat, c=region_ids, cmap='tab20', alpha=0.6)
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title('Regional Distribution')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0])
    
    # 区域样本数量
    regions = list(region_counts.keys())
    counts = [region_counts[r]['count'] for r in regions]
    axes[1].bar(regions, counts)
    axes[1].set_xlabel('Region ID')
    axes[1].set_ylabel('Sample Count')
    axes[1].set_title('Samples per Region')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return region_counts


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    lat = np.random.uniform(18, 54, n_samples)
    lon = np.random.uniform(73, 135, n_samples)
    
    # 测试不同的区域划分策略
    print("=== 网格划分 ===")
    assigner_grid = create_region_assigner('grid', n_lat=4, n_lon=4)
    region_ids_grid = assigner_grid.assign_regions(lat, lon)
    print(f"区域数量: {len(np.unique(region_ids_grid))}")
    print(f"区域分布: {np.bincount(region_ids_grid)}")
    
    print("\n=== 气候带划分 ===")
    assigner_climate = create_region_assigner('climate')
    region_ids_climate = assigner_climate.assign_regions(lat, lon)
    print(f"区域数量: {len(np.unique(region_ids_climate))}")
    print(f"区域分布: {np.bincount(region_ids_climate)}")
    
    # 分析区域分布
    analyze_regional_distribution(region_ids_grid, lat, lon, 'regional_distribution.png')
