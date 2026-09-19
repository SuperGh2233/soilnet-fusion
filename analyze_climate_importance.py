#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
气候变量重要性分析工具

提供多种方法分析17个气候变量的贡献度：
1. 消融实验（Ablation Study）- 逐个移除变量
2. 梯度重要性（Gradient-based Importance）
3. 注意力权重分析（如果使用Transformer）
4. 相关性分析
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_model_and_config(model_path, device='cuda'):
    """加载模型和配置"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # 从checkpoint或JSON文件获取配置
    config_path = model_path.replace('.pth.tar', '.json').replace('RUN_', 'Metrics_')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    return checkpoint, config


def method1_ablation_study(model, test_dataloader, climate_csv_files, device='cuda', 
                          num_samples=100):
    """
    方法1：消融实验
    逐个移除每个气候变量，观察性能下降
    """
    print("\n" + "="*80)
    print("方法1: 消融实验 (Ablation Study)")
    print("="*80)
    
    model.eval()
    baseline_errors = []
    ablation_results = {}
    
    # 1. 计算基线性能（使用所有变量）
    print("\n计算基线性能（使用所有17个气候变量）...")
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(test_dataloader, desc="Baseline")):
            if isinstance(X, tuple):
                X = [t.to(device) for t in X]
                y = y.to(device)
            else:
                X, y = X.to(device), y.to(device)
            
            y_pred = model(X)
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]
            
            # 计算误差
            errors = torch.abs(y_pred.squeeze() - y).cpu().numpy()
            baseline_errors.extend(errors)
            
            if batch_idx * test_dataloader.batch_size >= num_samples:
                break
    
    baseline_mae = np.mean(baseline_errors)
    print(f"基线 MAE: {baseline_mae:.4f}")
    
    # 2. 逐个移除变量
    print("\n逐个移除气候变量，计算性能变化...")
    for var_idx, var_name in enumerate(tqdm(climate_csv_files, desc="Ablation")):
        # 创建修改后的数据集（移除该变量）
        # 注意：这里需要修改数据集加载逻辑，暂时跳过
        # 实际实现需要修改 dataset_loader 来支持变量mask
        ablation_results[var_name] = {
            'importance': 0.0,  # 占位符
            'mae_increase': 0.0
        }
    
    return ablation_results, baseline_mae


def method2_gradient_importance(model, test_dataloader, climate_csv_files, device='cuda',
                                num_samples=50):
    """
    方法2：基于梯度的特征重要性
    计算每个气候变量对输出的梯度，梯度越大说明越重要
    """
    print("\n" + "="*80)
    print("方法2: 梯度重要性分析 (Gradient-based Importance)")
    print("="*80)
    
    model.eval()
    importance_scores = {name: [] for name in climate_csv_files}
    
    print(f"\n计算梯度重要性（使用 {num_samples} 个样本）...")
    
    sample_count = 0
    with torch.set_grad_enabled(True):
        for batch_idx, (X, y) in enumerate(test_dataloader):
            if sample_count >= num_samples:
                break
                
            if isinstance(X, tuple):
                X_list = [t.to(device) for t in X]
                climate_data = X_list[1]  # 假设第二个是气候数据 [B, T, F]
                y = y.to(device)
            else:
                continue
            
            climate_data.requires_grad = True
            
            # 前向传播
            y_pred = model(X_list)
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]
            
            # 计算损失
            loss = nn.functional.mse_loss(y_pred.squeeze(), y)
            
            # 反向传播
            model.zero_grad()
            loss.backward()
            
            # 计算每个变量的梯度重要性
            if climate_data.grad is not None:
                # climate_data.grad: [B, T, F]
                # 对时间和batch维度求平均，得到每个特征的重要性
                grad_importance = torch.abs(climate_data.grad).mean(dim=(0, 1)).cpu().numpy()  # [F]
                
                for var_idx, var_name in enumerate(climate_csv_files):
                    if var_idx < len(grad_importance):
                        importance_scores[var_name].append(grad_importance[var_idx])
            
            sample_count += climate_data.shape[0]
    
    # 计算平均重要性
    avg_importance = {}
    for var_name, scores in importance_scores.items():
        if scores:
            avg_importance[var_name] = np.mean(scores)
        else:
            avg_importance[var_name] = 0.0
    
    # 归一化到 [0, 1]
    max_imp = max(avg_importance.values()) if avg_importance.values() else 1.0
    if max_imp > 0:
        avg_importance = {k: v / max_imp for k, v in avg_importance.items()}
    
    return avg_importance


def method3_attention_weights(model, test_dataloader, climate_csv_files, device='cuda',
                              num_samples=50):
    """
    方法3：注意力权重分析（如果使用Transformer）
    提取Transformer的注意力权重，分析每个变量的关注度
    """
    print("\n" + "="*80)
    print("方法3: 注意力权重分析 (Attention Weights)")
    print("="*80)
    
    # 检查模型是否使用Transformer
    if not hasattr(model, 'lstm') or not hasattr(model.lstm, 'get_attention_weights'):
        print("警告: 模型不支持注意力权重提取，跳过此方法")
        return None
    
    model.eval()
    attention_scores = {name: [] for name in climate_csv_files}
    
    print(f"\n提取注意力权重（使用 {num_samples} 个样本）...")
    
    sample_count = 0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_dataloader):
            if sample_count >= num_samples:
                break
            
            if isinstance(X, tuple):
                X_list = [t.to(device) for t in X]
            else:
                continue
            
            # 获取注意力权重
            try:
                attn_weights = model.lstm.get_attention_weights(X_list[1])  # [B, heads, T, T] or [B, T, T]
                
                # 对时间维度求平均，得到每个时间步的平均注意力
                if len(attn_weights.shape) == 4:
                    attn_weights = attn_weights.mean(dim=1)  # [B, T, T]
                
                # 对目标时间步（通常是最后一个）的注意力求平均
                target_attn = attn_weights[:, -1, :].mean(dim=0).cpu().numpy()  # [T]
                
                # 对特征维度求平均（假设每个时间步对应一个特征）
                # 这里需要根据实际模型结构调整
                for var_idx, var_name in enumerate(climate_csv_files):
                    if var_idx < len(target_attn):
                        attention_scores[var_name].append(target_attn[var_idx])
                
                sample_count += X_list[1].shape[0]
            except Exception as e:
                print(f"无法提取注意力权重: {e}")
                return None
    
    # 计算平均注意力
    avg_attention = {}
    for var_name, scores in attention_scores.items():
        if scores:
            avg_attention[var_name] = np.mean(scores)
        else:
            avg_attention[var_name] = 0.0
    
    # 归一化
    max_attn = max(avg_attention.values()) if avg_attention.values() else 1.0
    if max_attn > 0:
        avg_attention = {k: v / max_attn for k, v in avg_attention.items()}
    
    return avg_attention


def method4_correlation_analysis(csv_path, climate_csv_folder, target_col='OC'):
    """
    方法4：相关性分析
    计算每个气候变量与SOC的相关性
    """
    print("\n" + "="*80)
    print("方法4: 相关性分析 (Correlation Analysis)")
    print("="*80)
    
    # 读取主CSV
    df_main = pd.read_csv(csv_path)
    
    # 读取所有气候变量CSV
    climate_csv_files = sorted([f for f in os.listdir(climate_csv_folder) if f.endswith('.csv')])
    
    correlations = {}
    
    print("\n计算每个气候变量与SOC的相关性...")
    
    for csv_file in tqdm(climate_csv_files):
        csv_path = os.path.join(climate_csv_folder, csv_file)
        df_climate = pd.read_csv(csv_path)
        
        # 合并数据
        df_merged = df_main.merge(df_climate, on='Point_ID', how='inner')
        
        # 提取日期列（8位数字）
        date_cols = [col for col in df_climate.columns 
                    if str(col).isdigit() and len(str(col)) == 8]
        
        if len(date_cols) == 0:
            continue
        
        # 计算平均相关性
        corrs = []
        for date_col in date_cols:
            if date_col in df_merged.columns and target_col in df_merged.columns:
                corr = df_merged[[target_col, date_col]].corr().iloc[0, 1]
                if not np.isnan(corr):
                    corrs.append(abs(corr))
        
        if corrs:
            correlations[csv_file] = np.mean(corrs)
        else:
            correlations[csv_file] = 0.0
    
    return correlations


def plot_importance_results(results_dict, save_path='climate_importance_analysis.png'):
    """绘制重要性分析结果"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('气候变量重要性分析', fontsize=16, fontweight='bold')
    
    # 提取变量名（去掉.csv后缀）
    var_names = [name.replace('.csv', '').replace('_merged_filled_norm', '') 
                 for name in results_dict.keys()]
    
    # 1. 梯度重要性
    if 'gradient' in results_dict:
        ax1 = axes[0, 0]
        grad_scores = list(results_dict['gradient'].values())
        ax1.barh(var_names, grad_scores, color='steelblue')
        ax1.set_xlabel('梯度重要性 (归一化)')
        ax1.set_title('方法2: 梯度重要性')
        ax1.grid(axis='x', alpha=0.3)
    
    # 2. 注意力权重
    if 'attention' in results_dict and results_dict['attention']:
        ax2 = axes[0, 1]
        attn_scores = list(results_dict['attention'].values())
        ax2.barh(var_names, attn_scores, color='coral')
        ax2.set_xlabel('注意力权重 (归一化)')
        ax2.set_title('方法3: 注意力权重')
        ax2.grid(axis='x', alpha=0.3)
    
    # 3. 相关性
    if 'correlation' in results_dict:
        ax3 = axes[1, 0]
        corr_scores = list(results_dict['correlation'].values())
        ax3.barh(var_names, corr_scores, color='green')
        ax3.set_xlabel('与SOC的相关系数')
        ax3.set_title('方法4: 相关性分析')
        ax3.grid(axis='x', alpha=0.3)
    
    # 4. 综合排名
    if len(results_dict) > 0:
        ax4 = axes[1, 1]
        
        # 计算综合得分（如果有多个方法）
        combined_scores = {}
        for var_name in var_names:
            scores = []
            if 'gradient' in results_dict:
                scores.append(results_dict['gradient'].get(var_name + '.csv', 0))
            if 'attention' in results_dict and results_dict['attention']:
                scores.append(results_dict['attention'].get(var_name + '.csv', 0))
            if 'correlation' in results_dict:
                scores.append(results_dict['correlation'].get(var_name + '.csv', 0))
            
            if scores:
                combined_scores[var_name] = np.mean(scores)
            else:
                combined_scores[var_name] = 0.0
        
        # 排序
        sorted_vars = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        var_names_sorted = [v[0] for v in sorted_vars]
        scores_sorted = [v[1] for v in sorted_vars]
        
        ax4.barh(var_names_sorted, scores_sorted, color='purple')
        ax4.set_xlabel('综合重要性得分')
        ax4.set_title('综合排名（平均所有方法）')
        ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: {save_path}")
    plt.close()


def save_results_to_csv(results_dict, save_path='climate_importance_results.csv'):
    """保存结果到CSV"""
    # 提取变量名
    var_names = []
    for method_results in results_dict.values():
        if method_results:
            var_names = [name.replace('.csv', '') for name in method_results.keys()]
            break
    
    # 构建DataFrame
    data = {'Variable': var_names}
    
    if 'gradient' in results_dict:
        data['Gradient_Importance'] = [results_dict['gradient'].get(name + '.csv', 0) 
                                       for name in var_names]
    
    if 'attention' in results_dict and results_dict['attention']:
        data['Attention_Weight'] = [results_dict['attention'].get(name + '.csv', 0) 
                                   for name in var_names]
    
    if 'correlation' in results_dict:
        data['Correlation'] = [results_dict['correlation'].get(name + '.csv', 0) 
                              for name in var_names]
    
    # 计算综合得分
    if len(data) > 1:
        scores = []
        for name in var_names:
            method_scores = []
            if 'Gradient_Importance' in data:
                method_scores.append(data['Gradient_Importance'][var_names.index(name)])
            if 'Attention_Weight' in data:
                method_scores.append(data['Attention_Weight'][var_names.index(name)])
            if 'Correlation' in data:
                method_scores.append(data['Correlation'][var_names.index(name)])
            scores.append(np.mean(method_scores) if method_scores else 0.0)
        data['Combined_Score'] = scores
    
    df = pd.DataFrame(data)
    df = df.sort_values('Combined_Score' if 'Combined_Score' in df.columns else 'Variable', 
                       ascending=False)
    df.to_csv(save_path, index=False)
    print(f"\n结果已保存: {save_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='气候变量重要性分析')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='模型checkpoint路径')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='主CSV文件路径（包含SOC数据）')
    parser.add_argument('--climate_folder', type=str, required=True,
                       help='气候数据CSV文件夹路径')
    parser.add_argument('--test_csv', type=str, default=None,
                       help='测试集CSV路径（如果与主CSV不同）')
    parser.add_argument('--method', type=str, choices=['all', 'gradient', 'correlation', 'attention'],
                       default='all', help='分析方法')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='用于分析的样本数')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    
    args = parser.parse_args()
    
    print("="*80)
    print("气候变量重要性分析工具")
    print("="*80)
    
    # 获取气候变量列表
    climate_csv_files = sorted([f for f in os.listdir(args.climate_folder) 
                               if f.endswith('.csv')])
    print(f"\n检测到 {len(climate_csv_files)} 个气候变量:")
    for i, f in enumerate(climate_csv_files, 1):
        print(f"  {i}. {f}")
    
    results = {}
    
    # 方法4：相关性分析（不需要模型）
    if args.method in ['all', 'correlation']:
        try:
            correlations = method4_correlation_analysis(args.csv_path, args.climate_folder)
            results['correlation'] = correlations
            print("\n相关性分析结果:")
            sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            for var_name, corr in sorted_corr[:5]:
                print(f"  {var_name}: {corr:.4f}")
        except Exception as e:
            print(f"相关性分析失败: {e}")
    
    # 方法2和3需要模型
    if args.method in ['all', 'gradient', 'attention']:
        try:
            # 加载模型
            print(f"\n加载模型: {args.model_path}")
            checkpoint, config = load_model_and_config(args.model_path, args.device)
            
            # 这里需要根据你的实际模型结构加载
            # 由于模型结构复杂，这里提供框架，需要根据实际情况调整
            print("警告: 模型加载需要根据实际结构实现")
            print("请参考以下方法手动实现:")
            print("  1. 使用 method2_gradient_importance() 计算梯度重要性")
            print("  2. 使用 method3_attention_weights() 提取注意力权重")
            
        except Exception as e:
            print(f"模型分析失败: {e}")
            print("提示: 可以只使用相关性分析（不需要模型）")
    
    # 保存结果
    if results:
        df = save_results_to_csv(results)
        plot_importance_results(results)
        
        print("\n" + "="*80)
        print("分析完成!")
        print("="*80)
        print("\n前5个最重要的气候变量:")
        print(df.head().to_string(index=False))
    else:
        print("\n未生成任何结果，请检查输入参数")


if __name__ == '__main__':
    main()




























