"""
诊断区域自适应训练问题的脚本
"""
import torch
import numpy as np
import pandas as pd
from soilnet.submodules.region_embedding import RegionEmbedding

def diagnose_regional_training():
    """诊断区域自适应训练问题"""
    
    print("=" * 60)
    print("区域自适应训练诊断")
    print("=" * 60)
    
    # 1. 检查区域嵌入初始化
    print("\n1. 区域嵌入初始化检查")
    num_regions = 6
    embed_dim = 128
    region_emb = RegionEmbedding(num_regions, embed_dim)
    
    # 检查初始权重
    init_weights = region_emb.region_embedding.weight.data
    print(f"   区域嵌入权重形状: {init_weights.shape}")
    print(f"   权重均值: {init_weights.mean().item():.6f}")
    print(f"   权重标准差: {init_weights.std().item():.6f}")
    print(f"   权重范围: [{init_weights.min().item():.6f}, {init_weights.max().item():.6f}]")
    
    # 测试前向传播
    test_region_ids = torch.tensor([0, 1, 2, 3, 4, 5])
    region_features = region_emb(test_region_ids)
    print(f"   区域特征输出形状: {region_features.shape}")
    print(f"   区域特征均值: {region_features.mean().item():.6f}")
    print(f"   区域特征标准差: {region_features.std().item():.6f}")
    
    # 2. 检查维度匹配
    print("\n2. 维度匹配检查")
    cnn_dim = 384  # ViT-CoMer
    lstm_dim = 128
    static_dim = 128  # 经过MLP编码后
    region_dim = 128
    
    total_input_dim = cnn_dim + lstm_dim + static_dim + region_dim
    print(f"   CNN维度: {cnn_dim}")
    print(f"   LSTM维度: {lstm_dim}")
    print(f"   静态特征维度: {static_dim}")
    print(f"   区域特征维度: {region_dim}")
    print(f"   总输入维度: {total_input_dim}")
    print(f"   拼接后维度: {cnn_dim + lstm_dim + static_dim + region_dim}")
    
    # 3. 检查区域分布
    print("\n3. 建议检查项")
    print("   - 区域ID是否正确对齐数据样本顺序")
    print("   - 区域分布是否均匀（某些区域样本过少会导致过拟合）")
    print("   - 区域嵌入学习率是否过大（建议使用较小的学习率）")
    print("   - 是否应该分阶段训练（先训练基础特征，再启用区域自适应）")
    
    # 4. 建议的修复方案
    print("\n4. 建议的修复方案")
    print("   方案A: 降低区域嵌入的学习率")
    print("      - 为区域嵌入模块设置独立的学习率（如主学习率的0.1倍）")
    print("   方案B: 简化区域特征融合")
    print("      - 移除区域门控机制，直接使用区域嵌入")
    print("      - 或者使用残差连接而非拼接")
    print("   方案C: 分阶段训练")
    print("      - 前N个epoch不启用区域自适应")
    print("      - 后续epoch再启用区域自适应")
    print("   方案D: 检查区域ID对齐")
    print("      - 验证数据加载顺序与区域ID数组顺序一致")
    print("      - 添加调试输出确认每个batch的区域ID正确")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    diagnose_regional_training()










