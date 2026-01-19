"""
区域可视化验证
画出中国地图，展示每个样点预测误差分布
可视化不同区域的偏差趋势
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

def create_china_map_base():
    """创建中国地图基础框架"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 中国大致边界
    ax.set_xlim(70, 140)
    ax.set_ylim(15, 55)
    
    # 添加主要地理要素
    ax.axhline(y=23.5, color='gray', linestyle='--', alpha=0.5, label='北回归线')
    ax.axhline(y=40, color='gray', linestyle='--', alpha=0.5, label='40°N')
    
    # 添加主要河流（简化）
    rivers = [
        {'name': '长江', 'lons': [90, 120], 'lats': [30, 30], 'color': 'blue'},
        {'name': '黄河', 'lons': [100, 120], 'lats': [35, 35], 'color': 'yellow'},
        {'name': '珠江', 'lons': [110, 115], 'lats': [23, 23], 'color': 'green'}
    ]
    
    for river in rivers:
        ax.plot(river['lons'], river['lats'], color=river['color'], 
                linewidth=2, alpha=0.7, label=river['name'])
    
    # 添加区域边界
    regions = [
        {'name': '东北', 'bounds': [(118, 135), (40, 53)], 'color': 'lightblue'},
        {'name': '华北', 'bounds': [(110, 120), (34, 42)], 'color': 'lightgreen'},
        {'name': '西北', 'bounds': [(75, 110), (35, 50)], 'color': 'lightyellow'},
        {'name': '华南', 'bounds': [(105, 120), (18, 30)], 'color': 'lightcoral'},
        {'name': '西南', 'bounds': [(97, 107), (22, 32)], 'color': 'lightpink'}
    ]
    
    for region in regions:
        rect = Rectangle((region['bounds'][0][0], region['bounds'][1][0]),
                        region['bounds'][0][1] - region['bounds'][0][0],
                        region['bounds'][1][1] - region['bounds'][1][0],
                        facecolor=region['color'], alpha=0.3, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(np.mean(region['bounds'][0]), np.mean(region['bounds'][1]), 
                region['name'], ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('经度 (°E)')
    ax.set_ylabel('纬度 (°N)')
    ax.set_title('中国自然地理分区图', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return fig, ax

def visualize_prediction_errors(csv_path, save_path='regional_error_analysis.png'):
    """可视化预测误差分布"""
    print("=== 区域预测误差可视化 ===")
    
    # 读取预测结果
    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取预测结果: {len(df)} 个样本")
    except FileNotFoundError:
        print(f"未找到预测结果文件: {csv_path}")
        print("使用模拟数据进行演示...")
        # 创建模拟数据
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'Latitude': np.random.uniform(18, 53, n_samples),
            'Longitude': np.random.uniform(75, 135, n_samples),
            'y_real': np.random.normal(10, 5, n_samples),
            'y_pred': np.random.normal(10, 5, n_samples)
        })
    
    # 分配区域
    df['region'] = df.apply(lambda row: assign_china_geographic_region(row['Latitude'], row['Longitude']), axis=1)
    
    # 计算误差
    df['error'] = df['y_pred'] - df['y_real']
    df['abs_error'] = np.abs(df['error'])
    df['rel_error'] = df['abs_error'] / (df['y_real'] + 1e-8)  # 相对误差
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 地理分布散点图 - 绝对误差
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['Longitude'], df['Latitude'], 
                         c=df['abs_error'], cmap='Reds', alpha=0.6, s=20)
    ax1.set_xlabel('经度 (°E)')
    ax1.set_ylabel('纬度 (°N)')
    ax1.set_title('预测绝对误差地理分布')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='绝对误差')
    
    # 2. 地理分布散点图 - 相对误差
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(df['Longitude'], df['Latitude'], 
                          c=df['rel_error'], cmap='Blues', alpha=0.6, s=20)
    ax2.set_xlabel('经度 (°E)')
    ax2.set_ylabel('纬度 (°N)')
    ax2.set_title('预测相对误差地理分布')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='相对误差')
    
    # 3. 各区域误差箱线图
    ax3 = axes[1, 0]
    region_data = [df[df['region'] == region]['abs_error'].values 
                   for region in df['region'].unique() if region != 'OTHER']
    region_labels = [region for region in df['region'].unique() if region != 'OTHER']
    
    bp = ax3.boxplot(region_data, labels=region_labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
    
    ax3.set_ylabel('绝对误差')
    ax3.set_title('各区域误差分布')
    ax3.grid(True, alpha=0.3)
    
    # 4. 区域性能热力图
    ax4 = axes[1, 1]
    region_stats = df.groupby('region').agg({
        'abs_error': ['mean', 'std'],
        'rel_error': 'mean',
        'y_real': 'count'
    }).round(3)
    
    # 创建热力图数据
    heatmap_data = region_stats[('abs_error', 'mean')].values.reshape(-1, 1)
    im = ax4.imshow(heatmap_data, cmap='Reds', aspect='auto')
    
    # 设置标签
    ax4.set_xticks([0])
    ax4.set_xticklabels(['平均绝对误差'])
    ax4.set_yticks(range(len(region_stats)))
    ax4.set_yticklabels(region_stats.index)
    
    # 添加数值标注
    for i in range(len(region_stats)):
        ax4.text(0, i, f"{heatmap_data[i, 0]:.3f}", 
                ha='center', va='center', fontweight='bold')
    
    ax4.set_title('各区域平均绝对误差')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"可视化结果保存至: {save_path}")
    
    return df

def analyze_regional_bias(df):
    """分析区域偏差趋势"""
    print("\n=== 区域偏差分析 ===")
    
    # 按区域统计
    region_stats = df.groupby('region').agg({
        'error': ['mean', 'std', 'count'],
        'abs_error': ['mean', 'std'],
        'rel_error': 'mean',
        'y_real': ['mean', 'std']
    }).round(4)
    
    print("各区域详细统计:")
    print(region_stats)
    
    # 识别偏差模式
    print("\n偏差模式分析:")
    
    for region in df['region'].unique():
        if region == 'OTHER':
            continue
            
        region_data = df[df['region'] == region]
        mean_error = region_data['error'].mean()
        std_error = region_data['error'].std()
        
        if abs(mean_error) > std_error:
            bias_type = "高估" if mean_error > 0 else "低估"
            print(f"  {region}: {bias_type} (平均偏差: {mean_error:.3f})")
        else:
            print(f"  {region}: 无明显偏差 (平均偏差: {mean_error:.3f})")
    
    # 创建偏差趋势图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 区域平均偏差
    ax1 = axes[0]
    region_means = df.groupby('region')['error'].mean()
    region_means = region_means[region_means.index != 'OTHER']
    
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink']
    bars = ax1.bar(region_means.index, region_means.values, color=colors[:len(region_means)])
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.set_ylabel('平均偏差')
    ax1.set_title('各区域平均预测偏差')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars, region_means.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 2. 区域误差变异性
    ax2 = axes[1]
    region_stds = df.groupby('region')['abs_error'].std()
    region_stds = region_stds[region_stds.index != 'OTHER']
    
    bars2 = ax2.bar(region_stds.index, region_stds.values, color=colors[:len(region_stds)])
    ax2.set_ylabel('误差标准差')
    ax2.set_title('各区域预测误差变异性')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars2, region_stds.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('regional_bias_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("偏差分析图保存至: regional_bias_analysis.png")

def create_comprehensive_analysis():
    """创建综合分析报告"""
    print("=== 区域预测综合分析 ===")
    
    # 创建中国地图
    fig, ax = create_china_map_base()
    plt.savefig('china_regional_map.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("中国区域地图保存至: china_regional_map.png")
    
    # 查找最新的预测结果文件
    results_dir = "results"
    if os.path.exists(results_dir):
        csv_files = [f for f in os.listdir(results_dir) if f.endswith('_best.csv')]
        if csv_files:
            latest_csv = max(csv_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
            csv_path = os.path.join(results_dir, latest_csv)
            print(f"使用预测结果文件: {csv_path}")
            
            # 可视化预测误差
            df = visualize_prediction_errors(csv_path)
            
            # 分析区域偏差
            analyze_regional_bias(df)
        else:
            print("未找到预测结果文件，使用模拟数据...")
            df = visualize_prediction_errors("dummy_path")
            analyze_regional_bias(df)
    else:
        print("results目录不存在，使用模拟数据...")
        df = visualize_prediction_errors("dummy_path")
        analyze_regional_bias(df)

def main():
    """主函数"""
    print("=== 区域可视化验证工具 ===")
    
    create_comprehensive_analysis()
    
    print("\n=== 使用说明 ===")
    print("1. 确保有预测结果CSV文件（包含Latitude, Longitude, y_real, y_pred列）")
    print("2. 运行此脚本进行区域可视化分析")
    print("3. 查看生成的地图和统计图表")
    print("4. 根据分析结果调整模型或训练策略")

if __name__ == "__main__":
    main()

