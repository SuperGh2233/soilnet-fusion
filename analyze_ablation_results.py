#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析四组消融实验的结果

运行方式:
    python analyze_ablation_results.py
"""

import json
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

def find_latest_results():
    """查找最新的实验结果文件"""
    experiments = {
        'baseline_raw_mse': None,
        'log1p_mse': None,
        'log1p_huber': None,
        'log1p_huber_w': None
    }
    
    results_dir = Path('results')
    
    for exp_name in experiments.keys():
        # 查找匹配的JSON文件
        pattern = f"RUN_{exp_name}_D_*.json"
        matching_files = list(results_dir.glob(pattern))
        
        if matching_files:
            # 选择最新的文件
            latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)
            experiments[exp_name] = latest_file
            print(f"找到 {exp_name}: {latest_file.name}")
        else:
            print(f"警告: 未找到 {exp_name} 的结果文件")
    
    return experiments

def load_experiment_results(json_path):
    """加载单个实验的结果"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return {
        'label_mode': data.get('LABEL_MODE', 'N/A'),
        'rmse_mean': data.get('RMSE_MEAN', np.nan),
        'r2_mean': data.get('R2_MEAN', np.nan),
        'mae_mean': data.get('MAE_MEAN', np.nan),
        'best_rmse': data.get('best_dict', {}).get('RMSE', np.nan),
        'best_r2': data.get('best_dict', {}).get('R2', np.nan),
        'best_mae': data.get('best_dict', {}).get('MAE', np.nan),
        'best_rpiq': data.get('best_dict', {}).get('RPIQ', np.nan),
        'best_ccc': data.get('best_dict', {}).get('CCC', np.nan),
        'num_epochs': data.get('NUM_EPOCHS', 'N/A'),
        'seeds': data.get('SEEDS', []),
        'huber_beta': data.get('HUBER_BETA', 'N/A'),
        'tail_threshold': data.get('TAIL_THRESHOLD', 'N/A'),
        'tail_weight': data.get('TAIL_WEIGHT', 'N/A')
    }

def analyze_high_value_samples(csv_path, threshold=30.0):
    """分析高值样本的预测效果"""
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    
    # 检查列名
    if 'y_real_raw' in df.columns and 'y_pred_raw' in df.columns:
        y_true = df['y_real_raw'].values
        y_pred = df['y_pred_raw'].values
    elif 'y_real' in df.columns and 'y_pred' in df.columns:
        y_true = df['y_real'].values
        y_pred = df['y_pred'].values
    else:
        return None
    
    # 整体统计
    overall_rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    overall_mae = np.mean(np.abs(y_pred - y_true))
    
    # 高值样本统计
    high_mask = y_true > threshold
    if high_mask.sum() > 0:
        high_rmse = np.sqrt(np.mean((y_pred[high_mask] - y_true[high_mask])**2))
        high_mae = np.mean(np.abs(y_pred[high_mask] - y_true[high_mask]))
        high_count = high_mask.sum()
        high_ratio = high_mask.mean() * 100
    else:
        high_rmse = high_mae = high_count = high_ratio = np.nan
    
    # 低值样本统计
    low_mask = y_true <= threshold
    if low_mask.sum() > 0:
        low_rmse = np.sqrt(np.mean((y_pred[low_mask] - y_true[low_mask])**2))
        low_mae = np.mean(np.abs(y_pred[low_mask] - y_true[low_mask]))
    else:
        low_rmse = low_mae = np.nan
    
    return {
        'overall_rmse': overall_rmse,
        'overall_mae': overall_mae,
        'high_value_rmse': high_rmse,
        'high_value_mae': high_mae,
        'high_value_count': high_count,
        'high_value_ratio': high_ratio,
        'low_value_rmse': low_rmse,
        'low_value_mae': low_mae
    }

def print_comparison_table(results_dict):
    """打印对比表格"""
    print("\n" + "="*100)
    print("四组消融实验结果对比")
    print("="*100)
    
    # 创建DataFrame
    rows = []
    for exp_name, result in results_dict.items():
        if result is None:
            continue
        rows.append({
            '实验': exp_name,
            'RMSE↓': f"{result['best_rmse']:.4f}",
            'MAE↓': f"{result['best_mae']:.4f}",
            'R²↑': f"{result['best_r2']:.4f}",
            'RPIQ↑': f"{result['best_rpiq']:.4f}",
            'CCC↑': f"{result['best_ccc']:.4f}",
            'Seeds': len(result['seeds'])
        })
    
    df = pd.DataFrame(rows)
    print("\n整体性能指标:")
    print(df.to_string(index=False))
    
    # 打印参数配置
    print("\n" + "-"*100)
    print("实验配置:")
    print("-"*100)
    for exp_name, result in results_dict.items():
        if result is None:
            continue
        print(f"\n{exp_name}:")
        print(f"  Label Mode: {result['label_mode']}")
        if result['huber_beta'] != 'N/A':
            print(f"  Huber Beta: {result['huber_beta']}")
        if result['tail_threshold'] != 'N/A':
            print(f"  Tail Threshold: {result['tail_threshold']}")
            print(f"  Tail Weight: {result['tail_weight']}")

def print_high_value_analysis(exp_files, threshold=30.0):
    """打印高值样本分析"""
    print("\n" + "="*100)
    print(f"高值样本分析 (SOC > {threshold} g/kg)")
    print("="*100)
    
    rows = []
    for exp_name, json_path in exp_files.items():
        if json_path is None:
            continue
        
        # 查找对应的CSV文件
        csv_pattern = str(json_path).replace('.json', '_best.csv')
        if os.path.exists(csv_pattern):
            analysis = analyze_high_value_samples(csv_pattern, threshold)
            if analysis:
                rows.append({
                    '实验': exp_name,
                    '整体RMSE': f"{analysis['overall_rmse']:.4f}",
                    '整体MAE': f"{analysis['overall_mae']:.4f}",
                    '高值RMSE': f"{analysis['high_value_rmse']:.4f}",
                    '高值MAE': f"{analysis['high_value_mae']:.4f}",
                    '低值RMSE': f"{analysis['low_value_rmse']:.4f}",
                    '低值MAE': f"{analysis['low_value_mae']:.4f}",
                    '高值样本数': int(analysis['high_value_count']),
                    '高值占比%': f"{analysis['high_value_ratio']:.2f}"
                })
    
    if rows:
        df = pd.DataFrame(rows)
        print("\n")
        print(df.to_string(index=False))
        print("\n说明: RMSE和MAE越小越好")
    else:
        print("\n未找到CSV文件进行分析")

def calculate_improvements(results_dict):
    """计算相对于baseline的改进"""
    if 'baseline_raw_mse' not in results_dict or results_dict['baseline_raw_mse'] is None:
        print("\n警告: 未找到baseline实验结果，无法计算改进")
        return
    
    baseline = results_dict['baseline_raw_mse']
    baseline_rmse = baseline['best_rmse']
    baseline_mae = baseline['best_mae']
    baseline_r2 = baseline['best_r2']
    
    print("\n" + "="*100)
    print("相对于 baseline_raw_mse 的改进")
    print("="*100)
    
    rows = []
    for exp_name, result in results_dict.items():
        if result is None or exp_name == 'baseline_raw_mse':
            continue
        
        rmse_improve = (baseline_rmse - result['best_rmse']) / baseline_rmse * 100
        mae_improve = (baseline_mae - result['best_mae']) / baseline_mae * 100
        r2_improve = (result['best_r2'] - baseline_r2) / abs(baseline_r2) * 100
        
        rows.append({
            '实验': exp_name,
            'RMSE改进%': f"{rmse_improve:+.2f}",
            'MAE改进%': f"{mae_improve:+.2f}",
            'R²改进%': f"{r2_improve:+.2f}"
        })
    
    df = pd.DataFrame(rows)
    print("\n")
    print(df.to_string(index=False))
    print("\n说明: 正值表示改进，负值表示退化")

def main():
    """主函数"""
    print("\n" + "="*100)
    print(" 标签策略消融实验结果分析")
    print("="*100 + "\n")
    
    # 查找实验结果文件
    print("正在查找实验结果文件...")
    exp_files = find_latest_results()
    
    # 加载结果
    results = {}
    for exp_name, json_path in exp_files.items():
        if json_path:
            try:
                results[exp_name] = load_experiment_results(json_path)
            except Exception as e:
                print(f"警告: 加载 {exp_name} 失败: {e}")
                results[exp_name] = None
    
    if not any(results.values()):
        print("\n错误: 未找到任何有效的实验结果")
        print("请先运行实验:")
        print("  bash run_ablation_experiments.sh")
        return
    
    # 打印对比表格
    print_comparison_table(results)
    
    # 打印高值样本分析
    print_high_value_analysis(exp_files, threshold=30.0)
    
    # 计算改进
    calculate_improvements(results)
    
    print("\n" + "="*100)
    print("分析完成!")
    print("="*100 + "\n")
    
    # 给出建议
    print("实验建议:")
    print("1. 如果高值样本预测效果显著改善，说明标签策略有效")
    print("2. 如果 log1p_* 模式整体性能更好，说明对数变换有助于缓解长尾分布")
    print("3. 如果 log1p_huber_w 在高值样本上表现最好，说明加权策略有效")
    print("4. 可以尝试调整 huber_beta, tail_threshold, tail_weight 参数进一步优化")
    print()

if __name__ == "__main__":
    main()



































