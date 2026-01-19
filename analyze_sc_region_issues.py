"""
分析华南地区(SC)预测效果差的可能原因
并提供改进建议
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config

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

def analyze_sc_region_issues(results_csv_path=None):
    """分析SC地区预测效果差的原因"""
    print("=" * 80)
    print("华南地区(SC)预测问题诊断分析")
    print("=" * 80)
    
    # 1. 读取预测结果（如果提供了路径）
    if results_csv_path and os.path.exists(results_csv_path):
        print(f"\n[1] 读取预测结果: {results_csv_path}")
        df_pred = pd.read_csv(results_csv_path)
        
        # 使用反归一化的值（如果存在）
        if 'y_real_denorm' in df_pred.columns and 'y_pred_denorm' in df_pred.columns:
            df_pred['y_real'] = df_pred['y_real_denorm']
            df_pred['y_pred'] = df_pred['y_pred_denorm']
        
        # 分配区域
        df_pred['region'] = df_pred.apply(
            lambda row: assign_china_geographic_region(row.get('Latitude', np.nan), row.get('Longitude', np.nan)), 
            axis=1
        )
        
        # SC地区分析
        sc_data = df_pred[df_pred['region'] == 'SC'].copy()
        print(f"   SC地区样本数: {len(sc_data)}")
        if len(sc_data) > 0:
            sc_rmse = np.sqrt(mean_squared_error(sc_data['y_real'], sc_data['y_pred']))
            sc_r2 = r2_score(sc_data['y_real'], sc_data['y_pred'])
            sc_mae = mean_absolute_error(sc_data['y_real'], sc_data['y_pred'])
            print(f"   SC地区指标: RMSE={sc_rmse:.3f}, R²={sc_r2:.3f}, MAE={sc_mae:.3f}")
            
            # 误差分析
            sc_data['error'] = sc_data['y_pred'] - sc_data['y_real']
            sc_data['abs_error'] = np.abs(sc_data['error'])
            sc_data['rel_error'] = sc_data['abs_error'] / (sc_data['y_real'] + 1e-8)
            
            print(f"\n   SC地区误差统计:")
            print(f"   平均绝对误差: {sc_data['abs_error'].mean():.3f}")
            print(f"   平均相对误差: {sc_data['rel_error'].mean():.3f}")
            print(f"   误差标准差: {sc_data['error'].std():.3f}")
            print(f"   系统性偏差: {sc_data['error'].mean():.3f}")
            
            # 分析高误差样本的特征
            high_error_threshold = sc_data['abs_error'].quantile(0.75)
            high_error_samples = sc_data[sc_data['abs_error'] > high_error_threshold]
            print(f"\n   高误差样本(>75%分位数)占比: {len(high_error_samples)/len(sc_data)*100:.1f}%")
            if len(high_error_samples) > 0:
                print(f"   高误差样本SOC范围: [{high_error_samples['y_real'].min():.2f}, {high_error_samples['y_real'].max():.2f}]")
                print(f"   高误差样本平均SOC: {high_error_samples['y_real'].mean():.2f}")
    
    # 2. 分析训练数据中的区域分布
    print(f"\n[2] 分析训练数据区域分布")
    if os.path.exists(config.lucas_csv_path):
        df_train = pd.read_csv(config.lucas_csv_path)
        if 'Latitude' in df_train.columns and 'Longitude' in df_train.columns:
            df_train['region'] = df_train.apply(
                lambda row: assign_china_geographic_region(row['Latitude'], row['Longitude']), 
                axis=1
            )
            
            region_counts = df_train['region'].value_counts()
            total = len(df_train)
            
            print(f"   总样本数: {total}")
            print(f"\n   各区域样本分布:")
            for region in ['NE', 'NC', 'NW', 'SC', 'SW', 'OTHER']:
                if region in region_counts.index:
                    count = region_counts[region]
                    pct = count / total * 100
                    if 'SOC' in df_train.columns:
                        soc_mean = df_train[df_train['region'] == region]['SOC'].mean()
                        print(f"     {region}: {count:4d} ({pct:5.1f}%) | SOC均值: {soc_mean:.2f}")
                    else:
                        print(f"     {region}: {count:4d} ({pct:5.1f}%)")
            
            # 检查SC地区样本是否不足
            if 'SC' in region_counts.index:
                sc_train_count = region_counts['SC']
                if sc_train_count < total * 0.1:
                    print(f"\n   ⚠️  警告: SC地区训练样本占比 < 10%，可能导致欠拟合")
                if sc_train_count < region_counts.max() * 0.5:
                    print(f"   ⚠️  警告: SC地区样本数明显少于其他区域，存在不平衡问题")
    
    # 3. 分析气候特征对SC地区的区分度
    print(f"\n[3] 建议添加的气候特征（针对湿润地区）")
    print("   华南地区特点: 高温、高湿、多雨、强蒸散")
    print("   建议添加的特征:")
    print("   1. P_minus_PET: 降水量-潜在蒸散量（水分盈亏指数）")
    print("   2. Aridity_index: 干燥指数 = PET/P")
    print("   3. Water_deficit: 水分亏缺指数")
    print("   4. Seasonal_variability: 季节性变异系数（降雨/温度的CV）")
    print("   5. Extreme_rainfall_days: 极端降雨日数")
    
    # 4. 改进建议
    print(f"\n[4] 改进建议")
    print("   方案A: 区域加权损失函数")
    print("      - 在损失函数中为SC地区样本分配更高权重")
    print("      - 使用 Focal Loss 或 Class-balanced Loss")
    print("\n   方案B: 区域自适应模型")
    print("      - 使用区域嵌入层(Region Embedding)")
    print("      - 为不同区域学习不同的特征表示")
    print("\n   方案C: 数据增强")
    print("      - 对SC地区样本进行更多数据增强")
    print("      - 使用SMOTE等技术增加SC样本")
    print("\n   方案D: 后处理校正")
    print("      - 针对SC地区训练专门的校正模型")
    print("      - 使用线性/非线性映射校正预测值")
    print("\n   方案E: 特征工程")
    print("      - 添加更多针对湿润地区的气候特征")
    print("      - 使用交互特征（温度×湿度等）")
    print("      - 季节性分解特征")
    
    return df_pred if results_csv_path and os.path.exists(results_csv_path) else None

def create_sc_improvement_script():
    """创建SC地区改进脚本"""
    script_content = """#!/usr/bin/env python
# -*- coding: utf-8 -*-
\"\"\"
华南地区改进方案: 生成P-PET水分盈亏特征
\"\"\"

import os
import pandas as pd
import numpy as np
from pathlib import Path

def create_water_deficit_feature():
    \"\"\"创建P-PET（水分盈亏）特征\"\"\"
    script_dir = Path(__file__).parent
    merges_dir = script_dir / "dataset" / "Climate" / "climate_exports_indexed" / "output_merges"
    
    pr_path = merges_dir / "pr_merged.csv"
    pet_path = merges_dir / "PET_merged.csv"
    
    if not pr_path.exists() or not pet_path.exists():
        print("错误: 未找到 pr_merged.csv 或 PET_merged.csv")
        return
    
    print("读取数据...")
    df_pr = pd.read_csv(pr_path)
    df_pet = pd.read_csv(pet_path)
    
    # 识别日期列
    date_cols = [c for c in df_pr.columns if str(c).isdigit()]
    
    # 识别Point_ID列
    def find_pid_col(df):
        for c in df.columns:
            if str(c).lower() in ['point_id', 'pointid', 'pid']:
                return c
        return None
    
    pid_pr = find_pid_col(df_pr)
    pid_pet = find_pid_col(df_pet)
    
    if pid_pr and pid_pet:
        # 基于Point_ID对齐
        pr_subset = df_pr[[pid_pr] + date_cols].copy()
        pet_subset = df_pet[[pid_pet] + date_cols].copy()
        pet_subset = pet_subset.rename(columns={pid_pet: pid_pr})
        
        # 计算P-PET
        result_df = pr_subset[[pid_pr]].copy()
        if 'Latitude' in df_pr.columns:
            result_df['Latitude'] = df_pr.set_index(pid_pr).loc[result_df[pid_pr], 'Latitude'].values
        if 'Longitude' in df_pr.columns:
            result_df['Longitude'] = df_pr.set_index(pid_pr).loc[result_df[pid_pr], 'Longitude'].values
        
        for d in date_cols:
            pr_vals = pd.to_numeric(pr_subset[d], errors='coerce').values
            pet_vals = pd.to_numeric(pet_subset.set_index(pid_pr).loc[pr_subset[pid_pr], d], errors='coerce').values
            result_df[d] = pr_vals - pet_vals
        
        output_path = merges_dir / "P_minus_PET_merged.csv"
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✓ 已生成: {output_path}")
        print("  提示: 运行 preprocess_merged_climate.py 进行填充和归一化")
    else:
        print("警告: 未找到Point_ID列，跳过")

if __name__ == "__main__":
    create_water_deficit_feature()
"""
    
    output_path = "create_sc_improvement_features.py"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    print(f"\n[5] 已创建改进脚本: {output_path}")
    print("   运行该脚本可生成P-PET特征（针对湿润地区的水分盈亏指数）")

if __name__ == "__main__":
    # 尝试查找最新的预测结果文件
    results_dir = "results"
    if os.path.exists(results_dir):
        csv_files = [f for f in os.listdir(results_dir) if f.endswith('_best.csv')]
        if csv_files:
            latest_csv = sorted(csv_files, key=lambda x: os.path.getmtime(os.path.join(results_dir, x)))[-1]
            results_csv_path = os.path.join(results_dir, latest_csv)
            print(f"使用最新预测结果: {results_csv_path}\n")
            analyze_sc_region_issues(results_csv_path)
        else:
            analyze_sc_region_issues()
    else:
        analyze_sc_region_issues()
    
    create_sc_improvement_script()

