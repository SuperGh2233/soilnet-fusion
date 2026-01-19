"""
区域性能分析脚本
基于中国自然地理分区分析各区域的SOC预测性能
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config


def assign_china_geographic_region(lat, lon):
    """
    基于经纬度分配中国自然地理分区
    
    Args:
        lat: 纬度
        lon: 经度
    
    Returns:
        str: 区域名称
    """
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


def analyze_regional_performance(csv_path, results_dir="results"):
    """
    分析各区域的预测性能
    
    Args:
        csv_path: 预测结果CSV文件路径
        results_dir: 结果保存目录
    """
    # 读取预测结果
    df = pd.read_csv(csv_path)
    # If denormalized columns exist, prefer them
    if 'y_real_denorm' in df.columns and 'y_pred_denorm' in df.columns:
        df['y_real'] = df['y_real_denorm']
        df['y_pred'] = df['y_pred_denorm']
    
    # 分配区域
    df['region'] = df.apply(lambda row: assign_china_geographic_region(row['Latitude'], row['Longitude']), axis=1)
    
    # 区域名称映射
    region_names = {
        "NE": "东北地区",
        "NC": "华北地区", 
        "NW": "西北地区",
        "SC": "华南地区",
        "SW": "西南地区",
        "OTHER": "其他地区"
    }
    
    # 计算各区域性能指标
    regional_results = {}
    
    for region in df['region'].unique():
        region_data = df[df['region'] == region]
        
        if len(region_data) < 3:  # 样本太少跳过
            continue
            
        y_true = region_data['y_real'].values
        y_pred = region_data['y_pred'].values
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        regional_results[region] = {
            'region_name': region_names.get(region, region),
            'n_samples': len(region_data),
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'lat_mean': region_data['Latitude'].mean(),
            'lon_mean': region_data['Longitude'].mean(),
            'soc_mean': y_true.mean(),
            'soc_std': y_true.std()
        }
    
    return regional_results, df


def create_regional_heatmap(regional_results, save_path="regional_performance_heatmap.png"):
    """
    创建区域性能热力图
    """
    # 准备数据
    regions = list(regional_results.keys())
    metrics = ['rmse', 'r2', 'mae']
    
    # 创建热力图数据
    heatmap_data = []
    for region in regions:
        row = []
        for metric in metrics:
            row.append(regional_results[region][metric])
        heatmap_data.append(row)
    
    heatmap_data = np.array(heatmap_data)
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # RMSE热力图
    sns.heatmap(heatmap_data[:, 0:1].T, 
                xticklabels=[regional_results[r]['region_name'] for r in regions],
                yticklabels=['RMSE'],
                annot=True, fmt='.3f', cmap='Reds',
                ax=axes[0])
    axes[0].set_title('区域RMSE性能热力图')
    
    # R²热力图
    sns.heatmap(heatmap_data[:, 1:2].T,
                xticklabels=[regional_results[r]['region_name'] for r in regions],
                yticklabels=['R²'],
                annot=True, fmt='.3f', cmap='Greens',
                ax=axes[1])
    axes[1].set_title('区域R²性能热力图')
    
    # MAE热力图
    sns.heatmap(heatmap_data[:, 2:3].T,
                xticklabels=[regional_results[r]['region_name'] for r in regions],
                yticklabels=['MAE'],
                annot=True, fmt='.3f', cmap='Blues',
                ax=axes[2])
    axes[2].set_title('区域MAE性能热力图')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"区域性能热力图保存至: {save_path}")


def create_regional_scatter_plot(df, save_path="regional_scatter_plot.png"):
    """
    创建区域散点图
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 地理分布散点图
    scatter = axes[0, 0].scatter(df['Longitude'], df['Latitude'], 
                                c=df['region'].astype('category').cat.codes, 
                                cmap='tab10', alpha=0.6)
    axes[0, 0].set_xlabel('经度')
    axes[0, 0].set_ylabel('纬度')
    axes[0, 0].set_title('样本地理分布')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加区域标签
    region_names = {
        "NE": "东北", "NC": "华北", "NW": "西北",
        "SC": "华南", "SW": "西南", "OTHER": "其他"
    }
    for region in df['region'].unique():
        region_data = df[df['region'] == region]
        axes[0, 0].text(region_data['Longitude'].mean(), 
                       region_data['Latitude'].mean(),
                       region_names.get(region, region),
                       ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 2. 预测vs真实值散点图
    axes[0, 1].scatter(df['y_real'], df['y_pred'], alpha=0.6)
    axes[0, 1].plot([df['y_real'].min(), df['y_real'].max()], 
                   [df['y_real'].min(), df['y_real'].max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('真实值')
    axes[0, 1].set_ylabel('预测值')
    axes[0, 1].set_title('预测vs真实值')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 各区域样本数量柱状图
    region_counts = df['region'].value_counts()
    axes[1, 0].bar(range(len(region_counts)), region_counts.values)
    axes[1, 0].set_xticks(range(len(region_counts)))
    axes[1, 0].set_xticklabels([region_names.get(r, r) for r in region_counts.index], rotation=45)
    axes[1, 0].set_ylabel('样本数量')
    axes[1, 0].set_title('各区域样本分布')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 各区域SOC分布箱线图
    region_data_list = []
    region_labels = []
    for region in df['region'].unique():
        region_data = df[df['region'] == region]['y_real']
        if len(region_data) > 0:
            region_data_list.append(region_data)
            region_labels.append(region_names.get(region, region))
    
    axes[1, 1].boxplot(region_data_list, labels=region_labels)
    axes[1, 1].set_ylabel('SOC值')
    axes[1, 1].set_title('各区域SOC分布')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"区域散点图保存至: {save_path}")


def print_regional_summary(regional_results):
    """
    打印区域性能总结
    """
    print("\n" + "="*80)
    print("区域性能分析总结")
    print("="*80)
    
    # 按R²排序
    sorted_regions = sorted(regional_results.items(), key=lambda x: x[1]['r2'], reverse=True)
    
    print(f"{'区域':<12} {'样本数':<8} {'RMSE':<8} {'R²':<8} {'MAE':<8} {'SOC均值':<8} {'SOC标准差':<8}")
    print("-"*80)
    
    for region, results in sorted_regions:
        print(f"{results['region_name']:<12} {results['n_samples']:<8} "
              f"{results['rmse']:<8.3f} {results['r2']:<8.3f} {results['mae']:<8.3f} "
              f"{results['soc_mean']:<8.3f} {results['soc_std']:<8.3f}")
    
    # 找出最佳和最差区域
    best_region = max(regional_results.items(), key=lambda x: x[1]['r2'])
    worst_region = min(regional_results.items(), key=lambda x: x[1]['r2'])
    
    print(f"\n最佳预测区域: {best_region[1]['region_name']} (R² = {best_region[1]['r2']:.3f})")
    print(f"最差预测区域: {worst_region[1]['region_name']} (R² = {worst_region[1]['r2']:.3f})")
    
    # 分析可能的原因
    print(f"\n区域性能分析:")
    for region, results in regional_results.items():
        if results['r2'] < 0.3:
            print(f"- {results['region_name']}: 预测性能较差 (R² = {results['r2']:.3f})")
            if results['n_samples'] < 20:
                print(f"  可能原因: 样本数量过少 ({results['n_samples']} 个)")
            if results['soc_std'] > 10:
                print(f"  可能原因: SOC变异性较大 (标准差 = {results['soc_std']:.3f})")


def main():
    """
    主函数
    """
    print("=== 区域性能分析 ===")
    
    # 查找最新的预测结果文件
    results_dir = "results"
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('_best.csv')]
    
    if not csv_files:
        print("未找到预测结果文件，请先运行训练")
        return
    
    # 使用最新的文件
    latest_csv = max(csv_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
    csv_path = os.path.join(results_dir, latest_csv)
    
    print(f"分析文件: {csv_path}")
    
    # 分析区域性能
    regional_results, df = analyze_regional_performance(csv_path)
    
    # 打印总结
    print_regional_summary(regional_results)
    
    # 创建可视化
    create_regional_heatmap(regional_results, "regional_performance_heatmap.png")
    create_regional_scatter_plot(df, "regional_scatter_plot.png")
    
    # 保存详细结果
    results_df = pd.DataFrame(regional_results).T
    results_df.to_csv("regional_performance_summary.csv")
    print(f"\n详细结果保存至: regional_performance_summary.csv")


if __name__ == "__main__":
    main()
