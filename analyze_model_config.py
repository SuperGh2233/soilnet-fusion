"""
分析最佳模型配置和区域性能验证准备
不依赖PyTorch，先分析模型结构和数据
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns

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

def analyze_model_config():
    """分析模型配置"""
    print("=== 最佳模型配置分析 ===")
    
    # 查找相关的配置文件
    config_files = []
    results_dir = "results"
    
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith('.json') and '9.18' in file:
                config_files.append(os.path.join(results_dir, file))
    
    if not config_files:
        print("未找到9.18相关的配置文件")
        return None
    
    # 读取最新的配置文件
    latest_config = max(config_files, key=os.path.getctime)
    print(f"使用配置文件: {latest_config}")
    
    with open(latest_config, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    print(f"\n模型配置信息:")
    print(f"  CNN架构: {config_data.get('CNN_ARCHITECTURE', 'Unknown')}")
    print(f"  RNN架构: {config_data.get('RNN_ARCHITECTURE', 'Unknown')}")
    print(f"  学习率: {config_data.get('LEARNING_RATE', 'Unknown')}")
    print(f"  训练轮次: {config_data.get('NUM_EPOCHS', 'Unknown')}")
    print(f"  批次大小: {config_data.get('TRAIN_BATCH_SIZE', 'Unknown')}")
    print(f"  使用SRTM: {config_data.get('USE_SRTM', 'Unknown')}")
    print(f"  使用LSTM分支: {config_data.get('USE_LSTM_BRANCH', 'Unknown')}")
    print(f"  最佳种子: {config_data.get('Best Seed', 'Unknown')}")
    
    # 性能指标
    print(f"\n模型性能:")
    print(f"  RMSE: {config_data.get('RMSE_MEAN', 'Unknown')}")
    print(f"  R2: {config_data.get('R2_MEAN', 'Unknown')}")
    print(f"  MAE: {config_data.get('MAE_MEAN', 'Unknown')}")
    
    return config_data

def analyze_test_data_distribution():
    """分析测试数据的地理分布"""
    print(f"\n=== 测试数据地理分布分析 ===")
    
    # 读取测试数据
    test_csv_path = "dataset/CN-SOC-3500.csv"  # 根据你的配置调整
    
    if not os.path.exists(test_csv_path):
        print(f"测试数据文件不存在: {test_csv_path}")
        return None
    
    df = pd.read_csv(test_csv_path)
    print(f"测试数据总样本数: {len(df)}")
    
    # 分配区域
    df['region'] = df.apply(lambda row: assign_china_geographic_region(row['Latitude'], row['Longitude']), axis=1)
    
    # 区域统计
    region_counts = df['region'].value_counts()
    print(f"\n各区域样本分布:")
    for region, count in region_counts.items():
        percentage = count / len(df) * 100
        print(f"  {region}: {count} 个样本 ({percentage:.1f}%)")
    
    # 分析SOC分布
    print(f"\nSOC统计信息:")
    print(f"  SOC范围: {df['SOC'].min():.2f} - {df['SOC'].max():.2f}")
    print(f"  SOC均值: {df['SOC'].mean():.2f}")
    print(f"  SOC标准差: {df['SOC'].std():.2f}")
    
    # 按区域分析SOC分布
    print(f"\n各区域SOC分布:")
    for region in region_counts.index:
        region_data = df[df['region'] == region]
        if len(region_data) > 0:
            print(f"  {region}: 均值={region_data['SOC'].mean():.2f}, "
                  f"标准差={region_data['SOC'].std():.2f}, "
                  f"范围={region_data['SOC'].min():.2f}-{region_data['SOC'].max():.2f}")
    
    return df

def create_geographic_visualization(df):
    """创建地理分布可视化"""
    print(f"\n=== 创建地理分布可视化 ===")
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 样本地理分布
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['Longitude'], df['Latitude'], 
                         c=df['region'].astype('category').cat.codes, 
                         cmap='tab10', alpha=0.6, s=10)
    ax1.set_xlabel('经度 (°E)')
    ax1.set_ylabel('纬度 (°N)')
    ax1.set_title('测试样本地理分布')
    ax1.grid(True, alpha=0.3)
    
    # 添加区域标签
    region_names = {
        "NE": "东北", "NC": "华北", "NW": "西北",
        "SC": "华南", "SW": "西南", "OTHER": "其他"
    }
    for region in df['region'].unique():
        region_data = df[df['region'] == region]
        if len(region_data) > 0:
            ax1.text(region_data['Longitude'].mean(), 
                    region_data['Latitude'].mean(),
                    region_names.get(region, region),
                    ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 2. SOC值地理分布
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(df['Longitude'], df['Latitude'], 
                          c=df['SOC'], cmap='viridis', alpha=0.6, s=10)
    ax2.set_xlabel('经度 (°E)')
    ax2.set_ylabel('纬度 (°N)')
    ax2.set_title('SOC值地理分布')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='SOC值')
    
    # 3. 各区域样本数量
    ax3 = axes[1, 0]
    region_counts = df['region'].value_counts()
    bars = ax3.bar(range(len(region_counts)), region_counts.values)
    ax3.set_xticks(range(len(region_counts)))
    ax3.set_xticklabels([region_names.get(r, r) for r in region_counts.index], rotation=45)
    ax3.set_ylabel('样本数量')
    ax3.set_title('各区域样本分布')
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, count in zip(bars, region_counts.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom')
    
    # 4. 各区域SOC分布箱线图
    ax4 = axes[1, 1]
    region_data_list = []
    region_labels = []
    for region in df['region'].unique():
        region_subset = df[df['region'] == region]
        if len(region_subset) > 0:
            region_data_list.append(region_subset['SOC'].values)
            region_labels.append(region_names.get(region, region))
    
    bp = ax4.boxplot(region_data_list, labels=region_labels)
    ax4.set_ylabel('SOC值')
    ax4.set_title('各区域SOC分布')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('test_data_geographic_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    
    print(f"地理分布图保存至: test_data_geographic_distribution.png")

def create_regional_analysis_plan():
    """创建区域分析计划"""
    print(f"\n=== 区域性能分析计划 ===")
    
    print("基于你的最佳模型 '9.18 加载预训练 0.4798.tar'，建议进行以下分析：")
    
    print(f"\n1. 模型加载和预测:")
    print(f"   - 加载模型权重")
    print(f"   - 对测试集进行预测")
    print(f"   - 计算整体性能指标")
    
    print(f"\n2. 区域性能分析:")
    print(f"   - 按地理分区计算RMSE、R2、MAE")
    print(f"   - 识别最佳和最差预测区域")
    print(f"   - 分析区域间的性能差异")
    
    print(f"\n3. 可视化分析:")
    print(f"   - 预测误差地理分布图")
    print(f"   - 各区域性能热力图")
    print(f"   - 区域误差箱线图")
    
    print(f"\n4. 深度分析:")
    print(f"   - 区域特征差异分析")
    print(f"   - 预测偏差模式识别")
    print(f"   - 改进建议")
    
    print(f"\n要运行完整的区域性能验证，需要:")
    print(f"1. 安装PyTorch: pip install torch torchvision")
    print(f"2. 运行: python test_regional_performance.py")

def main():
    """主函数"""
    print("=== 最佳模型区域性能分析准备 ===")
    
    # 1. 分析模型配置
    config_data = analyze_model_config()
    
    # 2. 分析测试数据分布
    df = analyze_test_data_distribution()
    
    if df is not None:
        # 3. 创建地理分布可视化
        create_geographic_visualization(df)
    
    # 4. 创建分析计划
    create_regional_analysis_plan()
    
    print(f"\n下一步:")
    print(f"1. 安装PyTorch环境")
    print(f"2. 运行完整的区域性能验证脚本")
    print(f"3. 分析结果并优化模型")

if __name__ == "__main__":
    main()
