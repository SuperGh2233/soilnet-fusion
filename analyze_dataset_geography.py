#!/usr/bin/env python3
"""
数据集地理分布分析脚本
分析训练/验证/测试集的地理分布是否均匀，检查空间聚集问题
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_dataset_info(csv_path, train_folder, val_folder, test_folder):
    """加载数据集信息"""
    print("=== 加载数据集信息 ===")
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    print(f"总样本数: {len(df)}")
    print(f"CSV列名: {list(df.columns)}")
    
    # 检查是否有经纬度信息
    lat_cols = [col for col in df.columns if 'lat' in col.lower()]
    lon_cols = [col for col in df.columns if 'lon' in col.lower()]
    
    if not lat_cols or not lon_cols:
        print("❌ 未找到经纬度列，请检查CSV文件")
        return None
    
    lat_col, lon_col = lat_cols[0], lon_cols[0]
    print(f"使用纬度列: {lat_col}, 经度列: {lon_col}")
    
    # 检查数据完整性
    valid_coords = df.dropna(subset=[lat_col, lon_col])
    print(f"有效坐标样本数: {len(valid_coords)}")
    
    if len(valid_coords) < len(df) * 0.8:
        print("⚠️  警告: 超过20%的样本缺少坐标信息")
    
    return df, lat_col, lon_col

def analyze_geographic_distribution(df, lat_col, lon_col, train_folder, val_folder, test_folder):
    """分析地理分布"""
    print("\n=== 地理分布分析 ===")
    
    # 获取各文件夹中的文件名
    train_files = set([f for f in os.listdir(train_folder) if f.endswith('.tif')])
    val_files = set([f for f in os.listdir(val_folder) if f.endswith('.tif')])
    test_files = set([f for f in os.listdir(test_folder) if f.endswith('.tif')])
    
    print(f"训练集文件数: {len(train_files)}")
    print(f"验证集文件数: {len(val_files)}")
    print(f"测试集文件数: {len(test_files)}")
    
    # 根据文件名匹配数据
    def get_split_info(point_id):
        filename = f"{point_id}_*.tif"
        if any(f.startswith(f"{point_id}_") for f in train_files):
            return 'train'
        elif any(f.startswith(f"{point_id}_") for f in val_files):
            return 'val'
        elif any(f.startswith(f"{point_id}_") for f in test_files):
            return 'test'
        else:
            return 'unknown'
    
    # 为每个样本分配数据集标签
    df['split'] = df['Point_id'].apply(get_split_info)
    
    # 统计各数据集样本数
    split_counts = df['split'].value_counts()
    print(f"\n数据集划分统计:")
    for split, count in split_counts.items():
        print(f"  {split}: {count} ({count/len(df)*100:.1f}%)")
    
    # 检查是否有未匹配的样本
    unknown_count = split_counts.get('unknown', 0)
    if unknown_count > 0:
        print(f"⚠️  警告: {unknown_count} 个样本未匹配到任何数据集")
    
    return df

def spatial_clustering_analysis(df, lat_col, lon_col):
    """空间聚类分析"""
    print("\n=== 空间聚类分析 ===")
    
    # 获取有效坐标
    valid_df = df.dropna(subset=[lat_col, lon_col])
    coords = valid_df[[lat_col, lon_col]].values
    
    # 尝试不同的聚类数
    silhouette_scores = []
    k_range = range(2, min(11, len(coords)//10))
    
    for k in k_range:
        if len(coords) < k:
            continue
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords)
        score = silhouette_score(coords, cluster_labels)
        silhouette_scores.append((k, score))
    
    if silhouette_scores:
        best_k, best_score = max(silhouette_scores, key=lambda x: x[1])
        print(f"最佳聚类数: {best_k} (轮廓系数: {best_score:.3f})")
        
        # 进行最佳聚类
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        valid_df['cluster'] = kmeans.fit_predict(coords)
        
        # 分析各聚类的数据集分布
        cluster_split = valid_df.groupby(['cluster', 'split']).size().unstack(fill_value=0)
        print(f"\n各聚类的数据集分布:")
        print(cluster_split)
        
        # 检查是否有聚类完全属于某个数据集（空间聚集问题）
        for cluster_id in range(best_k):
            cluster_data = cluster_split.loc[cluster_id]
            max_split = cluster_data.idxmax()
            max_ratio = cluster_data.max() / cluster_data.sum()
            
            if max_ratio > 0.8:
                print(f"⚠️  聚类 {cluster_id} 中 {max_split} 占 {max_ratio:.1%}，可能存在空间聚集")
        
        return valid_df, best_k
    else:
        print("❌ 无法进行聚类分析（样本数不足）")
        return valid_df, 0

def geographic_statistics(df, lat_col, lon_col):
    """地理统计信息"""
    print("\n=== 地理统计信息 ===")
    
    valid_df = df.dropna(subset=[lat_col, lon_col])
    
    for split in ['train', 'val', 'test']:
        split_data = valid_df[valid_df['split'] == split]
        if len(split_data) == 0:
            continue
            
        print(f"\n{split.upper()} 集:")
        print(f"  样本数: {len(split_data)}")
        print(f"  纬度范围: {split_data[lat_col].min():.4f} ~ {split_data[lat_col].max():.4f}")
        print(f"  经度范围: {split_data[lon_col].min():.4f} ~ {split_data[lon_col].max():.4f}")
        print(f"  纬度均值: {split_data[lat_col].mean():.4f} ± {split_data[lat_col].std():.4f}")
        print(f"  经度均值: {split_data[lon_col].mean():.4f} ± {split_data[lon_col].std():.4f}")

def create_visualizations(df, lat_col, lon_col, output_dir='geography_analysis'):
    """创建可视化图表"""
    print(f"\n=== 创建可视化图表 (保存到 {output_dir}) ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    valid_df = df.dropna(subset=[lat_col, lon_col])
    
    # 1. 数据集分布散点图
    plt.figure(figsize=(12, 8))
    splits = ['train', 'val', 'test']
    colors = ['blue', 'orange', 'green']
    
    for i, (split, color) in enumerate(zip(splits, colors)):
        split_data = valid_df[valid_df['split'] == split]
        if len(split_data) > 0:
            plt.scatter(split_data[lon_col], split_data[lat_col], 
                       c=color, alpha=0.6, label=f'{split} ({len(split_data)})', s=20)
    
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.title('数据集地理分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 密度热力图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (split, color) in enumerate(zip(splits, colors)):
        split_data = valid_df[valid_df['split'] == split]
        if len(split_data) > 0:
            axes[i].hexbin(split_data[lon_col], split_data[lat_col], 
                          gridsize=20, cmap='Blues', alpha=0.8)
            axes[i].set_title(f'{split.upper()} 集密度分布')
            axes[i].set_xlabel('经度')
            axes[i].set_ylabel('纬度')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/density_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 聚类结果可视化
    if 'cluster' in valid_df.columns:
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(valid_df[lon_col], valid_df[lat_col], 
                            c=valid_df['cluster'], cmap='tab10', alpha=0.7, s=20)
        plt.colorbar(scatter, label='聚类ID')
        plt.xlabel('经度')
        plt.ylabel('纬度')
        plt.title('空间聚类结果')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/spatial_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. 数据集比例饼图
    split_counts = valid_df['split'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(split_counts.values, labels=split_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('数据集划分比例')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/split_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 图表已保存到 {output_dir}/ 目录")

def generate_recommendations(df, lat_col, lon_col):
    """生成改进建议"""
    print("\n=== 改进建议 ===")
    
    valid_df = df.dropna(subset=[lat_col, lon_col])
    
    # 检查数据集大小平衡
    split_counts = valid_df['split'].value_counts()
    min_count = split_counts.min()
    max_count = split_counts.max()
    
    if max_count / min_count > 3:
        print("⚠️  数据集大小不平衡，建议:")
        print("   - 重新划分数据集，确保各集样本数相对均衡")
        print("   - 考虑使用分层抽样")
    
    # 检查空间分布
    if 'cluster' in valid_df.columns:
        cluster_split = valid_df.groupby(['cluster', 'split']).size().unstack(fill_value=0)
        
        # 检查是否有聚类完全属于某个数据集
        spatial_bias = False
        for cluster_id in range(len(cluster_split)):
            cluster_data = cluster_split.loc[cluster_id]
            max_ratio = cluster_data.max() / cluster_data.sum()
            if max_ratio > 0.8:
                spatial_bias = True
                break
        
        if spatial_bias:
            print("⚠️  存在空间聚集问题，建议:")
            print("   - 使用空间分层抽样重新划分数据集")
            print("   - 考虑按地理区域进行交叉验证")
            print("   - 增加数据增强以改善空间泛化")
        else:
            print("✅ 空间分布相对均匀")
    
    # 检查坐标范围
    lat_range = valid_df[lat_col].max() - valid_df[lat_col].min()
    lon_range = valid_df[lon_col].max() - valid_df[lon_col].min()
    
    if lat_range < 1 or lon_range < 1:
        print("⚠️  地理覆盖范围较小，可能影响模型泛化能力")
        print("   - 考虑收集更大地理范围的数据")
        print("   - 使用数据增强模拟不同地理条件")
    
    print("\n💡 其他建议:")
    print("   - 定期检查数据集划分的随机性")
    print("   - 考虑使用时间分层（如果有时间信息）")
    print("   - 监控不同地理区域的预测性能")

def main():
    """主函数"""
    print("🔍 数据集地理分布分析工具")
    print("=" * 50)
    
    # 配置路径（请根据实际情况修改）
    csv_path = "dataset/CN-SOC-3500.csv"  # 修改为你的CSV路径
    train_folder = "dataset/l8_images_CN_split/train"  # 修改为你的训练集路径
    val_folder = "dataset/l8_images_CN_split/val"      # 修改为你的验证集路径
    test_folder = "dataset/l8_images_CN_split/test"    # 修改为你的测试集路径
    
    try:
        # 1. 加载数据
        result = load_dataset_info(csv_path, train_folder, val_folder, test_folder)
        if result is None:
            return
        
        df, lat_col, lon_col = result
        
        # 2. 分析地理分布
        df = analyze_geographic_distribution(df, lat_col, lon_col, train_folder, val_folder, test_folder)
        
        # 3. 空间聚类分析
        valid_df, n_clusters = spatial_clustering_analysis(df, lat_col, lon_col)
        
        # 4. 地理统计
        geographic_statistics(df, lat_col, lon_col)
        
        # 5. 创建可视化
        create_visualizations(df, lat_col, lon_col)
        
        # 6. 生成建议
        generate_recommendations(df, lat_col, lon_col)
        
        print("\n✅ 分析完成！")
        
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("请检查路径配置是否正确")
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
