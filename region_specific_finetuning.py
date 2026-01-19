"""
区域自适应性微调实验
针对每个地理分区单独进行fine-tuning
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from dataset.dataset_loader_china import ChinaSNDatasetClimate
from soilnet.enhanced_soil_net_fixed import EnhancedSoilNetLSTM
from soilnet.submodules.region_embedding import RegionAssigner

def assign_china_geographic_region(lat, lon):
    """基于经纬度分配中国自然地理分区"""
    if 40 <= lat <= 53 and 118 <= lon <= 135:
        return "NE"  # 东北地区
    elif 34 <= lat <= 42 and 110 <= lon <= 120:
        return "NC"  # 华北地区
    elif 35 <= lat <= 50 and 75 <= lon <= 110:
        return "NW"  # 西北地区
    elif 18 <= lat <= 30 and 105 <= lon <= 120:
        return "SC"  # 华南地区
    elif 22 <= lat <= 32 and 97 <= lon <= 107:
        return "SW"  # 西南地区
    else:
        return "OTHER"  # 其他地区

def load_regional_data():
    """加载并划分区域数据"""
    print("=== 加载区域数据 ===")
    
    # 读取CSV数据
    df = pd.read_csv(config.lucas_csv_path)
    
    # 分配区域
    df['region'] = df.apply(lambda row: assign_china_geographic_region(row['Latitude'], row['Longitude']), axis=1)
    
    # 区域统计
    region_counts = df['region'].value_counts()
    print("各区域样本数量:")
    for region, count in region_counts.items():
        print(f"  {region}: {count} 个样本")
    
    return df

def create_regional_datasets(df, region_name, train_ratio=0.7, val_ratio=0.15):
    """为特定区域创建数据集"""
    region_data = df[df['region'] == region_name].copy()
    
    if len(region_data) < 10:
        print(f"警告: {region_name} 区域样本数量过少 ({len(region_data)} 个)")
        return None, None, None
    
    # 随机划分
    np.random.seed(42)
    indices = np.random.permutation(len(region_data))
    
    train_size = int(len(region_data) * train_ratio)
    val_size = int(len(region_data) * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_data = region_data.iloc[train_indices]
    val_data = region_data.iloc[val_indices]
    test_data = region_data.iloc[test_indices]
    
    print(f"{region_name} 区域划分: 训练集 {len(train_data)}, 验证集 {len(val_data)}, 测试集 {len(test_data)}")
    
    return train_data, val_data, test_data

def fine_tune_for_region(region_name, train_data, val_data, test_data, base_model_path=None):
    """为特定区域进行fine-tuning"""
    print(f"\n=== 为 {region_name} 区域进行Fine-tuning ===")
    
    # 创建数据集
    train_dataset = ChinaSNDatasetClimate(
        csv_file=train_data,
        root_dir=config.train_l8_folder_path,
        transform=None,
        return_point_id=True
    )
    
    val_dataset = ChinaSNDatasetClimate(
        csv_file=val_data,
        root_dir=config.val_l8_folder_path,
        transform=None,
        return_point_id=True
    )
    
    test_dataset = ChinaSNDatasetClimate(
        csv_file=test_data,
        root_dir=config.test_l8_folder_path,
        transform=None,
        return_point_id=True
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 创建模型
    model = EnhancedSoilNetLSTM(
        cnn_arch="ViT-CoMer",
        rnn_arch="Transformer",
        use_regional_adaptation=True,
        num_regions=6,
        regional_fusion_type='attention'
    )
    
    # 如果提供了预训练模型，加载权重
    if base_model_path and os.path.exists(base_model_path):
        print(f"加载预训练模型: {base_model_path}")
        checkpoint = torch.load(base_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 只训练区域相关参数
    for name, param in model.named_parameters():
        if 'region' not in name:
            param.requires_grad = False
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # 优化器和损失函数
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    criterion = nn.MSELoss()
    
    # 训练
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(50):  # 较少的epoch用于fine-tuning
        # 训练
        model.train()
        train_loss = 0
        for batch_idx, (images, climate, targets, point_ids) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 获取区域ID（简化处理，假设所有样本属于同一区域）
            region_ids = torch.zeros(len(images), dtype=torch.long)
            
            outputs = model(images, climate, region_ids)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, climate, targets, point_ids in val_loader:
                region_ids = torch.zeros(len(images), dtype=torch.long)
                outputs = model(images, climate, region_ids)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, f'results/regional_model_{region_name}_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停于第 {epoch+1} 轮")
                break
    
    # 测试
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for images, climate, targets, point_ids in test_loader:
            region_ids = torch.zeros(len(images), dtype=torch.long)
            outputs = model(images, climate, region_ids)
            test_predictions.extend(outputs.squeeze().cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
    
    # 计算指标
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    
    rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
    r2 = r2_score(test_targets, test_predictions)
    mae = mean_absolute_error(test_targets, test_predictions)
    
    print(f"{region_name} 区域测试结果:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    return {
        'region': region_name,
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'n_samples': len(test_data)
    }

def run_regional_finetuning():
    """运行区域fine-tuning实验"""
    print("=== 区域自适应性微调实验 ===")
    
    # 加载数据
    df = load_regional_data()
    
    # 获取所有区域
    regions = df['region'].unique()
    print(f"\n发现 {len(regions)} 个区域: {list(regions)}")
    
    # 为每个区域进行fine-tuning
    results = []
    
    for region in regions:
        if region == 'OTHER':  # 跳过其他区域
            continue
            
        # 创建区域数据集
        train_data, val_data, test_data = create_regional_datasets(df, region)
        
        if train_data is None:
            continue
        
        # Fine-tuning
        result = fine_tune_for_region(region, train_data, val_data, test_data)
        results.append(result)
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/regional_finetuning_results.csv', index=False)
    
    print("\n=== 区域Fine-tuning结果汇总 ===")
    print(results_df)
    
    return results_df

if __name__ == "__main__":
    run_regional_finetuning()

