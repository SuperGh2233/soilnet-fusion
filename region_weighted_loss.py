"""
区域权重损失实现
在loss中为样本添加region_weight，平衡区域样本数量差异
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import Counter

class RegionWeightedLoss(nn.Module):
    """区域权重损失函数"""
    
    def __init__(self, region_weights=None, base_loss='mse', alpha=1.0):
        """
        Args:
            region_weights: 区域权重字典 {region_id: weight}
            base_loss: 基础损失函数 ('mse', 'mae', 'huber')
            alpha: 权重调节参数
        """
        super().__init__()
        self.region_weights = region_weights or {}
        self.alpha = alpha
        
        if base_loss == 'mse':
            self.base_criterion = nn.MSELoss(reduction='none')
        elif base_loss == 'mae':
            self.base_criterion = nn.L1Loss(reduction='none')
        elif base_loss == 'huber':
            self.base_criterion = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported base_loss: {base_loss}")
    
    def forward(self, predictions, targets, region_ids):
        """
        Args:
            predictions: 预测值 [B, 1]
            targets: 真实值 [B, 1]
            region_ids: 区域ID [B]
        Returns:
            weighted_loss: 加权损失
        """
        # 计算基础损失
        base_loss = self.base_criterion(predictions.squeeze(), targets.squeeze())
        
        # 获取区域权重
        if self.region_weights:
            weights = torch.tensor([self.region_weights.get(rid.item(), 1.0) for rid in region_ids], 
                                 device=predictions.device, dtype=predictions.dtype)
        else:
            weights = torch.ones_like(base_loss)
        
        # 应用权重
        weighted_loss = base_loss * weights
        
        return weighted_loss.mean()

def calculate_region_weights(df, method='inverse_frequency', alpha=1.0):
    """
    计算区域权重
    
    Args:
        df: 包含区域信息的DataFrame
        method: 权重计算方法
            - 'inverse_frequency': 反频率权重
            - 'balanced': 平衡权重
            - 'sqrt_inverse': 平方根反频率权重
        alpha: 权重平滑参数
    """
    region_counts = df['region'].value_counts()
    total_samples = len(df)
    num_regions = len(region_counts)
    
    print(f"区域样本分布:")
    for region, count in region_counts.items():
        print(f"  {region}: {count} 个样本 ({count/total_samples*100:.1f}%)")
    
    if method == 'inverse_frequency':
        # 反频率权重
        weights = {}
        for region, count in region_counts.items():
            weights[region] = total_samples / (num_regions * count)
    
    elif method == 'balanced':
        # 平衡权重：让每个区域的权重与其样本数成反比
        max_count = region_counts.max()
        weights = {}
        for region, count in region_counts.items():
            weights[region] = max_count / count
    
    elif method == 'sqrt_inverse':
        # 平方根反频率权重（更平滑）
        weights = {}
        for region, count in region_counts.items():
            weights[region] = np.sqrt(total_samples / (num_regions * count))
    
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # 应用平滑参数
    if alpha != 1.0:
        for region in weights:
            weights[region] = alpha * weights[region] + (1 - alpha) * 1.0
    
    print(f"\n计算的区域权重:")
    for region, weight in weights.items():
        print(f"  {region}: {weight:.3f}")
    
    return weights

def create_region_weighted_training():
    """创建区域权重训练脚本"""
    
    script_content = '''#!/bin/bash
# 区域权重损失训练脚本

echo "=== 区域权重损失训练实验 ==="

# 1. 反频率权重训练
echo "1. 反频率权重训练..."
python train.py -e EXP_RegionWeight_InvFreq -d CHINA -w 8 -cnn ViT-CoMer -rnn Transformer -trbs 32 -ne 80 -lr 0.00005 -ls plateau -srtm -lstm -ra -nr 6 -rs geographic -rw -rwm inverse_frequency -seed 1 2 3

# 2. 平衡权重训练
echo "2. 平衡权重训练..."
python train.py -e EXP_RegionWeight_Balanced -d CHINA -w 8 -cnn ViT-CoMer -rnn Transformer -trbs 32 -ne 80 -lr 0.00005 -ls plateau -srtm -lstm -ra -nr 6 -rs geographic -rw -rwm balanced -seed 1 2 3

# 3. 平方根反频率权重训练
echo "3. 平方根反频率权重训练..."
python train.py -e EXP_RegionWeight_SqrtInv -d CHINA -w 8 -cnn ViT-CoMer -rnn Transformer -trbs 32 -ne 80 -lr 0.00005 -ls plateau -srtm -lstm -ra -nr 6 -rs geographic -rw -rwm sqrt_inverse -seed 1 2 3

# 4. 对比：无权重训练
echo "4. 对比：无权重训练..."
python train.py -e EXP_RegionWeight_None -d CHINA -w 8 -cnn ViT-CoMer -rnn Transformer -trbs 32 -ne 80 -lr 0.00005 -ls plateau -srtm -lstm -ra -nr 6 -rs geographic -seed 1 2 3

echo "=== 训练完成 ==="
echo "结果文件保存在 results/ 目录"
'''
    
    with open('run_region_weighted_training.sh', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("区域权重训练脚本已创建: run_region_weighted_training.sh")

def analyze_region_imbalance():
    """分析区域不平衡问题"""
    print("=== 区域不平衡分析 ===")
    
    # 这里需要实际的CSV数据
    # df = pd.read_csv('dataset/CN-SOC-3500.csv')
    # df['region'] = df.apply(lambda row: assign_china_geographic_region(row['Latitude'], row['Longitude']), axis=1)
    
    # 模拟数据用于演示
    regions = ['NE', 'NC', 'NW', 'SC', 'SW', 'OTHER']
    counts = [800, 1200, 600, 900, 700, 300]  # 模拟各区域样本数
    
    df_sim = pd.DataFrame({
        'region': np.repeat(regions, counts),
        'SOC': np.random.normal(10, 5, sum(counts))
    })
    
    print("模拟区域分布:")
    region_counts = df_sim['region'].value_counts()
    total = len(df_sim)
    
    for region, count in region_counts.items():
        print(f"  {region}: {count} 个样本 ({count/total*100:.1f}%)")
    
    # 计算不平衡度
    max_count = region_counts.max()
    min_count = region_counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"\n不平衡度分析:")
    print(f"  最大区域样本数: {max_count}")
    print(f"  最小区域样本数: {min_count}")
    print(f"  不平衡比例: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 3:
        print("  ⚠️  严重不平衡，建议使用区域权重")
    elif imbalance_ratio > 2:
        print("  ⚠️  中等不平衡，建议考虑区域权重")
    else:
        print("  ✅ 相对平衡")
    
    # 计算不同权重策略
    print(f"\n不同权重策略:")
    
    # 反频率权重
    inv_freq_weights = {}
    for region, count in region_counts.items():
        inv_freq_weights[region] = total / (len(region_counts) * count)
    
    print("反频率权重:")
    for region, weight in inv_freq_weights.items():
        print(f"  {region}: {weight:.3f}")
    
    # 平衡权重
    balanced_weights = {}
    for region, count in region_counts.items():
        balanced_weights[region] = max_count / count
    
    print("\n平衡权重:")
    for region, weight in balanced_weights.items():
        print(f"  {region}: {weight:.3f}")

def main():
    """主函数"""
    print("=== 区域权重损失分析 ===")
    
    # 分析区域不平衡
    analyze_region_imbalance()
    
    # 创建训练脚本
    create_region_weighted_training()
    
    print("\n=== 使用建议 ===")
    print("1. 先分析你的数据集区域分布")
    print("2. 根据不平衡程度选择合适的权重策略")
    print("3. 运行对应的训练命令")
    print("4. 对比有无权重的效果差异")

if __name__ == "__main__":
    main()

