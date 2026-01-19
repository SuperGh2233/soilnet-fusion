#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版气候变量重要性分析（仅使用相关性分析，不需要模型）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def analyze_climate_correlation(csv_path, climate_csv_folder, target_col='OC', 
                                output_dir='climate_analysis'):
    """
    分析每个气候变量与SOC的相关性
    
    Args:
        csv_path: 主CSV文件路径（包含SOC和Point_ID）
        climate_csv_folder: 气候数据CSV文件夹
        target_col: 目标列名（通常是'OC'）
        output_dir: 输出目录
    """
    print("="*80)
    print("气候变量相关性分析")
    print("="*80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取主CSV
    print(f"\n读取主数据: {csv_path}")
    df_main = pd.read_csv(csv_path)
    print(f"  样本数: {len(df_main)}")
    print(f"  列: {df_main.columns.tolist()}")
    
    # 检查目标列
    if target_col not in df_main.columns:
        print(f"错误: 未找到目标列 '{target_col}'")
        print(f"可用列: {df_main.columns.tolist()}")
        return None
    
    # 获取气候变量文件列表
    climate_csv_files = sorted([f for f in os.listdir(climate_csv_folder) 
                               if f.endswith('.csv')])
    print(f"\n检测到 {len(climate_csv_files)} 个气候变量:")
    for i, f in enumerate(climate_csv_files, 1):
        print(f"  {i:2d}. {f}")
    
    # 分析每个变量
    results = []
    
    print(f"\n计算每个气候变量与 {target_col} 的相关性...")
    for csv_file in climate_csv_files:
        csv_path_full = os.path.join(climate_csv_folder, csv_file)
        df_climate = pd.read_csv(csv_path_full)
        
        # 检查Point_ID列
        id_col = None
        for col in df_climate.columns:
            if 'point' in col.lower() or 'id' in col.lower():
                id_col = col
                break
        
        if id_col is None:
            print(f"  警告: {csv_file} 未找到Point_ID列，跳过")
            continue
        
        # 合并数据
        df_merged = df_main.merge(df_climate, left_on='Point_ID', right_on=id_col, how='inner')
        
        if len(df_merged) == 0:
            print(f"  警告: {csv_file} 无法合并数据，跳过")
            continue
        
        # 提取日期列（8位数字格式：YYYYMMDD）
        date_cols = [col for col in df_climate.columns 
                    if str(col).isdigit() and len(str(col)) == 8]
        
        if len(date_cols) == 0:
            print(f"  警告: {csv_file} 未找到日期列，跳过")
            continue
        
        # 计算每个时间步的相关性
        correlations = []
        for date_col in date_cols:
            if date_col in df_merged.columns:
                # 移除缺失值
                valid_data = df_merged[[target_col, date_col]].dropna()
                if len(valid_data) > 10:  # 至少需要10个有效样本
                    corr = valid_data[target_col].corr(valid_data[date_col])
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        if correlations:
            mean_corr = np.mean(correlations)
            std_corr = np.std(correlations)
            max_corr = np.max(correlations)
            min_corr = np.min(correlations)
            
            results.append({
                'Variable': csv_file.replace('.csv', '').replace('_merged_filled_norm', ''),
                'Mean_Correlation': mean_corr,
                'Std_Correlation': std_corr,
                'Max_Correlation': max_corr,
                'Min_Correlation': min_corr,
                'Num_TimeSteps': len(correlations)
            })
            
            print(f"  {csv_file:40s} | 平均相关性: {mean_corr:.4f} ± {std_corr:.4f}")
        else:
            print(f"  {csv_file:40s} | 无法计算相关性")
    
    # 转换为DataFrame并排序
    df_results = pd.DataFrame(results)
    if len(df_results) > 0:
        df_results = df_results.sort_values('Mean_Correlation', ascending=False)
        
        # 保存CSV
        csv_output = os.path.join(output_dir, 'climate_correlation_results.csv')
        df_results.to_csv(csv_output, index=False)
        print(f"\n结果已保存: {csv_output}")
        
        # 绘制图表
        plot_correlation_results(df_results, output_dir)
        
        return df_results
    else:
        print("\n未生成任何结果")
        return None


def plot_correlation_results(df_results, output_dir):
    """绘制相关性分析结果"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. 平均相关性条形图
    ax1 = axes[0]
    var_names = df_results['Variable'].values
    corr_values = df_results['Mean_Correlation'].values
    
    colors = plt.cm.RdYlGn(corr_values / corr_values.max() if corr_values.max() > 0 else corr_values)
    bars = ax1.barh(var_names, corr_values, color=colors)
    ax1.set_xlabel('平均相关系数 (绝对值)', fontsize=12)
    ax1.set_title('气候变量与SOC的平均相关性', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, corr_values)):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    # 2. 相关性分布（带误差棒）
    ax2 = axes[1]
    y_pos = np.arange(len(var_names))
    ax2.errorbar(corr_values, y_pos, 
                xerr=df_results['Std_Correlation'].values,
                fmt='o', capsize=5, capthick=2, markersize=8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(var_names)
    ax2.set_xlabel('相关系数 (平均值 ± 标准差)', fontsize=12)
    ax2.set_title('气候变量相关性分布', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(output_dir, 'climate_correlation_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {plot_path}")
    plt.close()


def analyze_temporal_patterns(csv_path, climate_csv_folder, target_col='OC',
                              output_dir='climate_analysis', top_n=5):
    """分析重要变量的时间模式"""
    print("\n" + "="*80)
    print("分析重要变量的时间模式")
    print("="*80)
    
    # 先进行相关性分析
    df_results = analyze_climate_correlation(csv_path, climate_csv_folder, target_col, output_dir)
    
    if df_results is None or len(df_results) == 0:
        return
    
    # 选择前N个重要变量
    top_vars = df_results.head(top_n)['Variable'].values
    
    print(f"\n分析前 {top_n} 个重要变量的时间模式...")
    
    # 读取主数据
    df_main = pd.read_csv(csv_path)
    
    # 为每个重要变量绘制时间序列
    fig, axes = plt.subplots(top_n, 1, figsize=(14, 3*top_n))
    if top_n == 1:
        axes = [axes]
    
    for idx, var_name in enumerate(top_vars):
        # 查找对应的CSV文件
        csv_file = None
        for f in os.listdir(climate_csv_folder):
            if var_name in f and f.endswith('.csv'):
                csv_file = f
                break
        
        if csv_file is None:
            continue
        
        csv_path_full = os.path.join(climate_csv_folder, csv_file)
        df_climate = pd.read_csv(csv_path_full)
        
        # 合并数据
        id_col = None
        for col in df_climate.columns:
            if 'point' in col.lower() or 'id' in col.lower():
                id_col = col
                break
        
        if id_col is None:
            continue
        
        df_merged = df_main.merge(df_climate, left_on='Point_ID', right_on=id_col, how='inner')
        
        # 提取日期列
        date_cols = sorted([col for col in df_climate.columns 
                          if str(col).isdigit() and len(str(col)) == 8])
        
        if len(date_cols) == 0:
            continue
        
        # 计算每个时间步的平均值和相关性
        time_steps = []
        mean_values = []
        correlations = []
        
        for date_col in date_cols:
            if date_col in df_merged.columns:
                valid_data = df_merged[[target_col, date_col]].dropna()
                if len(valid_data) > 10:
                    time_steps.append(date_col)
                    mean_values.append(valid_data[date_col].mean())
                    corr = valid_data[target_col].corr(valid_data[date_col])
                    correlations.append(abs(corr) if not np.isnan(corr) else 0)
        
        # 绘制
        ax = axes[idx]
        ax2 = ax.twinx()
        
        # 平均值曲线
        line1 = ax.plot(range(len(time_steps)), mean_values, 'b-', marker='o', 
                       label='平均值', linewidth=2)
        ax.set_ylabel('变量平均值', color='b', fontsize=10)
        ax.tick_params(axis='y', labelcolor='b')
        
        # 相关性曲线
        line2 = ax2.plot(range(len(time_steps)), correlations, 'r--', marker='s',
                        label='与SOC相关性', linewidth=2)
        ax2.set_ylabel('相关系数', color='r', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax.set_xlabel('时间步', fontsize=10)
        ax.set_title(f'{var_name} (平均相关性: {df_results.iloc[idx]["Mean_Correlation"]:.4f})',
                    fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(time_steps)))
        ax.set_xticklabels([t[:6] for t in time_steps], rotation=45, ha='right')
        ax.grid(alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'climate_temporal_patterns.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"时间模式图表已保存: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='气候变量相关性分析（简化版，不需要模型）')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='主CSV文件路径（包含SOC和Point_ID）')
    parser.add_argument('--climate_folder', type=str, required=True,
                       help='气候数据CSV文件夹路径')
    parser.add_argument('--target_col', type=str, default='OC',
                       help='目标列名（默认: OC）')
    parser.add_argument('--output_dir', type=str, default='climate_analysis',
                       help='输出目录（默认: climate_analysis）')
    parser.add_argument('--temporal', action='store_true',
                       help='是否分析时间模式')
    
    args = parser.parse_args()
    
    # 基本相关性分析
    df_results = analyze_climate_correlation(
        args.csv_path, 
        args.climate_folder, 
        args.target_col,
        args.output_dir
    )
    
    if df_results is not None:
        print("\n" + "="*80)
        print("分析结果摘要")
        print("="*80)
        print(f"\n前10个最重要的气候变量:")
        print(df_results.head(10).to_string(index=False))
        
        print(f"\n后5个最不重要的气候变量:")
        print(df_results.tail(5).to_string(index=False))
    
    # 时间模式分析
    if args.temporal:
        analyze_temporal_patterns(
            args.csv_path,
            args.climate_folder,
            args.target_col,
            args.output_dir
        )


if __name__ == '__main__':
    main()

