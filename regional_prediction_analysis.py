"""
区域性预测分析脚本
支持多种区域划分策略，分析模型在不同地理区域的预测性能
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from soilnet.submodules.region_embedding import RegionAssigner, analyze_regional_distribution
from soilnet.enhanced_soil_net_fixed import EnhancedSoilNetLSTM
import config


class RegionalPredictor:
    """
    区域预测分析器
    """
    def __init__(self, model_path: str, dataset_path: str, 
                 region_strategy: str = 'grid', n_regions: int = 16):
        """
        Args:
            model_path: 训练好的模型路径
            dataset_path: 数据集CSV路径
            region_strategy: 区域划分策略 ('grid', 'climate', 'administrative')
            n_regions: 区域数量
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.region_strategy = region_strategy
        self.n_regions = n_regions
        
        # 加载数据
        self.df = pd.read_csv(dataset_path)
        print(f"加载数据集: {len(self.df)} 个样本")
        
        # 创建区域分配器
        if region_strategy == 'grid':
            n_lat = int(np.sqrt(n_regions))
            n_lon = n_regions // n_lat
            self.assigner = RegionAssigner('grid', n_lat=n_lat, n_lon=n_lon)
        else:
            self.assigner = RegionAssigner(region_strategy)
        
        # 分配区域ID
        self.region_ids = self.assigner.assign_regions(
            self.df['Latitude'].values, 
            self.df['Longitude'].values
        )
        self.df['region_id'] = self.region_ids
        
        print(f"区域分布: {np.bincount(self.region_ids)}")
        
    def load_model(self, device: str = 'cuda'):
        """加载训练好的模型"""
        self.device = device
        
        # 这里需要根据你的模型配置调整
        model = EnhancedSoilNetLSTM(
            cnn_arch="resnet101",  # 根据你的模型调整
            rnn_arch="LSTM",
            cnn_in_channels=12,
            regresor_input_from_cnn=1024,
            lstm_n_features=14,
            lstm_n_layers=2,
            lstm_out=128,
            hidden_size=128,
            seq_len=61,
            img_size=64,
            use_enhanced_climate=False,
            use_cross_modal_fusion=False,
            use_regional_adaptation=True,
            num_regions=self.n_regions
        )
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.to(device)
        model.eval()
        self.model = model
        
        print(f"模型加载完成: {self.model_path}")
        
    def predict_regional(self, test_indices: np.ndarray) -> dict:
        """
        对测试集进行区域预测分析
        """
        test_df = self.df.iloc[test_indices].copy()
        
        # 按区域分组
        regional_results = {}
        unique_regions = np.unique(test_df['region_id'])
        
        for region_id in unique_regions:
            region_mask = test_df['region_id'] == region_id
            region_data = test_df[region_mask]
            
            if len(region_data) < 3:  # 样本太少跳过
                continue
                
            # 这里需要实现具体的预测逻辑
            # 由于需要加载图像和气候数据，这里提供框架
            regional_results[region_id] = {
                'n_samples': len(region_data),
                'lat_mean': region_data['Latitude'].mean(),
                'lon_mean': region_data['Longitude'].mean(),
                'soc_mean': region_data['SOC'].mean(),
                'soc_std': region_data['SOC'].std(),
                # 预测指标将在实际预测后填充
                'predictions': None,
                'r2': None,
                'rmse': None,
                'mae': None
            }
        
        return regional_results
    
    def visualize_regional_performance(self, regional_results: dict, 
                                     save_path: str = 'regional_analysis.png'):
        """
        可视化区域预测性能
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 区域样本分布
        regions = list(regional_results.keys())
        sample_counts = [regional_results[r]['n_samples'] for r in regions]
        
        axes[0, 0].bar(regions, sample_counts)
        axes[0, 0].set_xlabel('Region ID')
        axes[0, 0].set_ylabel('Sample Count')
        axes[0, 0].set_title('Samples per Region')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 区域R2分布
        r2_scores = [regional_results[r]['r2'] for r in regions 
                    if regional_results[r]['r2'] is not None]
        if r2_scores:
            axes[0, 1].bar(range(len(r2_scores)), r2_scores)
            axes[0, 1].set_xlabel('Region ID')
            axes[0, 1].set_ylabel('R² Score')
            axes[0, 1].set_title('R² Score by Region')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 地理分布散点图
        scatter = axes[1, 0].scatter(
            [regional_results[r]['lon_mean'] for r in regions],
            [regional_results[r]['lat_mean'] for r in regions],
            c=sample_counts, cmap='viridis', s=100, alpha=0.7
        )
        axes[1, 0].set_xlabel('Longitude')
        axes[1, 0].set_ylabel('Latitude')
        axes[1, 0].set_title('Regional Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='Sample Count')
        
        # 4. SOC分布热力图
        if len(regions) > 1:
            # 创建区域-SOC热力图
            soc_means = [regional_results[r]['soc_mean'] for r in regions]
            soc_stds = [regional_results[r]['soc_std'] for r in regions]
            
            # 简单的热力图表示
            heatmap_data = np.array([soc_means, soc_stds]).T
            sns.heatmap(heatmap_data, ax=axes[1, 1], 
                       xticklabels=['SOC Mean', 'SOC Std'],
                       yticklabels=regions, annot=True, fmt='.2f')
            axes[1, 1].set_title('SOC Statistics by Region')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"区域分析图保存至: {save_path}")
    
    def analyze_regional_patterns(self, regional_results: dict) -> dict:
        """
        分析区域预测模式
        """
        analysis = {
            'best_regions': [],
            'worst_regions': [],
            'regional_patterns': {},
            'recommendations': []
        }
        
        # 找出最佳和最差区域
        r2_scores = [(r, regional_results[r]['r2']) for r in regional_results.keys() 
                     if regional_results[r]['r2'] is not None]
        
        if r2_scores:
            r2_scores.sort(key=lambda x: x[1], reverse=True)
            analysis['best_regions'] = r2_scores[:3]  # 前3名
            analysis['worst_regions'] = r2_scores[-3:]  # 后3名
        
        # 分析区域模式
        for region_id, results in regional_results.items():
            if results['r2'] is None:
                continue
                
            # 样本数量分析
            if results['n_samples'] < 5:
                analysis['recommendations'].append(
                    f"区域 {region_id}: 样本数量过少 ({results['n_samples']} 个)，建议增加数据"
                )
            
            # 性能分析
            if results['r2'] > 0.7:
                analysis['regional_patterns'][region_id] = 'high_performance'
            elif results['r2'] < 0.3:
                analysis['regional_patterns'][region_id] = 'low_performance'
            else:
                analysis['regional_patterns'][region_id] = 'medium_performance'
        
        return analysis


def create_regional_dataset_with_ids(csv_path: str, output_path: str, 
                                   region_strategy: str = 'grid', n_regions: int = 16):
    """
    为数据集添加区域ID列
    """
    df = pd.read_csv(csv_path)
    
    # 创建区域分配器
    if region_strategy == 'grid':
        n_lat = int(np.sqrt(n_regions))
        n_lon = n_regions // n_lat
        assigner = RegionAssigner('grid', n_lat=n_lat, n_lon=n_lon)
    else:
        assigner = RegionAssigner(region_strategy)
    
    # 分配区域ID
    region_ids = assigner.assign_regions(
        df['Latitude'].values, 
        df['Longitude'].values
    )
    df['region_id'] = region_ids
    
    # 保存带区域ID的数据集
    df.to_csv(output_path, index=False)
    print(f"带区域ID的数据集保存至: {output_path}")
    print(f"区域分布: {np.bincount(region_ids)}")
    
    return df


def main():
    """
    主函数 - 区域性预测分析示例
    """
    print("=== 区域性预测分析 ===")
    
    # 1. 为数据集添加区域ID
    dataset_path = config.lucas_csv_path
    output_path = 'dataset/China-SOCD-Dataset-with-regions.csv'
    
    print("1. 为数据集添加区域ID...")
    df_with_regions = create_regional_dataset_with_ids(
        dataset_path, output_path, 
        region_strategy='grid', n_regions=16
    )
    
    # 2. 分析区域分布
    print("\n2. 分析区域分布...")
    analyze_regional_distribution(
        df_with_regions['region_id'].values,
        df_with_regions['Latitude'].values,
        df_with_regions['Longitude'].values,
        'regional_distribution_analysis.png'
    )
    
    # 3. 创建区域预测分析器
    print("\n3. 创建区域预测分析器...")
    # 注意：这里需要你提供实际的模型路径
    model_path = "results/best_model.pth"  # 替换为你的模型路径
    
    if os.path.exists(model_path):
        predictor = RegionalPredictor(
            model_path=model_path,
            dataset_path=output_path,
            region_strategy='grid',
            n_regions=16
        )
        
        # 加载模型
        predictor.load_model()
        
        # 这里可以继续实现具体的预测分析
        print("模型加载成功，可以进行区域预测分析")
    else:
        print(f"模型文件不存在: {model_path}")
        print("请先训练模型，然后更新模型路径")


if __name__ == "__main__":
    main()

