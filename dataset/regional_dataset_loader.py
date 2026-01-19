"""
区域数据集加载器
在现有数据集基础上自动添加区域ID
"""

import pandas as pd
import numpy as np
from soilnet.submodules.region_embedding import create_region_assigner


def add_regional_ids_to_dataset(csv_path: str, region_strategy: str = 'grid', 
                               n_regions: int = 16, output_path: str = None):
    """
    为数据集添加区域ID列
    
    Args:
        csv_path: 原始CSV文件路径
        region_strategy: 区域划分策略 ('grid', 'climate')
        n_regions: 区域数量
        output_path: 输出文件路径，如果为None则覆盖原文件
    
    Returns:
        pd.DataFrame: 带区域ID的数据框
    """
    # 读取原始数据
    df = pd.read_csv(csv_path)
    
    # 创建区域分配器
    if region_strategy == 'grid':
        n_lat = int(np.sqrt(n_regions))
        n_lon = n_regions // n_lat
        assigner = create_region_assigner('grid', n_lat=n_lat, n_lon=n_lon)
    else:
        assigner = create_region_assigner(region_strategy)
    
    # 分配区域ID
    region_ids = assigner.assign_regions(
        df['Latitude'].values, 
        df['Longitude'].values
    )
    df['region_id'] = region_ids
    
    # 保存结果
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"带区域ID的数据集保存至: {output_path}")
    else:
        df.to_csv(csv_path, index=False)
        print(f"原数据集已更新，添加了区域ID列")
    
    print(f"区域分布: {np.bincount(region_ids)}")
    return df


def get_regional_indices(csv_path: str, indices: list, region_strategy: str = 'grid', 
                        n_regions: int = 16):
    """
    获取指定索引对应的区域ID
    
    Args:
        csv_path: CSV文件路径
        indices: 样本索引列表
        region_strategy: 区域划分策略
        n_regions: 区域数量
    
    Returns:
        np.ndarray: 区域ID数组
    """
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 创建区域分配器
    if region_strategy == 'grid':
        n_lat = int(np.sqrt(n_regions))
        n_lon = n_regions // n_lat
        assigner = create_region_assigner('grid', n_lat=n_lat, n_lon=n_lon)
    else:
        assigner = create_region_assigner(region_strategy)
    
    # 获取指定索引的经纬度
    subset_df = df.iloc[indices]
    
    # 分配区域ID
    region_ids = assigner.assign_regions(
        subset_df['Latitude'].values, 
        subset_df['Longitude'].values
    )
    
    return region_ids


if __name__ == "__main__":
    # 测试区域分配
    import config
    
    # 为LUCAS数据集添加区域ID
    df = add_regional_ids_to_dataset(
        config.lucas_csv_path,
        region_strategy='grid',
        n_regions=16,
        output_path='dataset/LUCAS_with_regions.csv'
    )
    
    print(f"数据集形状: {df.shape}")
    print(f"区域ID范围: {df['region_id'].min()} - {df['region_id'].max()}")
    print(f"区域分布:\n{df['region_id'].value_counts().sort_index()}")

