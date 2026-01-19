"""
从Seed 3的训练结果更新原始JSON的best_dict
"""
import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Seed 3的训练结果文件
seed3_exp_name = '11.03_add_static_seed3_recovery'
seed3_json_path = f'results/RUN_{seed3_exp_name}_D_*.json'
seed3_csv_path = f'results/RUN_{seed3_exp_name}_*_best.csv'

# 原始JSON文件
original_json_path = 'results/RUN_11.03_add_static_D_2025_11_03_T_18_56.json'

print("=" * 60)
print("从Seed 3结果更新原始JSON")
print("=" * 60)

# 查找Seed 3的结果文件
import glob
seed3_json_files = glob.glob(f'results/RUN_{seed3_exp_name}_*.json')
seed3_csv_files = glob.glob(f'results/RUN_{seed3_exp_name}_*_best.csv')

if not seed3_json_files:
    print(f"错误：找不到Seed 3的JSON结果文件")
    print(f"请确保训练已完成，文件模式: results/RUN_{seed3_exp_name}_*.json")
    exit(1)

if not seed3_csv_files:
    print(f"错误：找不到Seed 3的CSV结果文件")
    print(f"请确保训练已完成，文件模式: results/RUN_{seed3_exp_name}_*_best.csv")
    exit(1)

seed3_json_path = seed3_json_files[0]
seed3_csv_path = seed3_csv_files[0]

print(f"找到Seed 3的JSON: {seed3_json_path}")
print(f"找到Seed 3的CSV: {seed3_csv_path}")

# 读取Seed 3的结果
with open(seed3_json_path, 'r', encoding='utf-8') as f:
    seed3_data = json.load(f)

df = pd.read_csv(seed3_csv_path)
oc_max = seed3_data.get('OC_MAX', 30.8)

# 反归一化
y_true = df['y_real'].values * oc_max
y_pred = df['y_pred'].values * oc_max

# 计算指标（使用修复后的RPIQ）
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
q1 = np.percentile(y_true, 25)
q3 = np.percentile(y_true, 75)
rpiq = (q3 - q1) / rmse
mae = np.mean(np.abs(y_true - y_pred))
mec = np.mean(y_true - y_pred)

def concordance_correlation_coefficient(y_real, y_pred):
    dct = {'y_real': y_real, 'y_pred': y_pred}
    df_temp = pd.DataFrame(dct).dropna()
    y_real = df_temp['y_real']
    y_pred = df_temp['y_pred']
    cor = np.corrcoef(y_real, y_pred)[0][1]
    mean_real = np.mean(y_real)
    mean_pred = np.mean(y_pred)
    var_real = np.var(y_real)
    var_pred = np.var(y_pred)
    sd_real = np.std(y_real)
    sd_pred = np.std(y_pred)
    numerator = 2 * cor * sd_real * sd_pred
    denominator = var_real + var_pred + (mean_real - mean_pred)**2
    return numerator / denominator

ccc = concordance_correlation_coefficient(y_true, y_pred)

print("\nSeed 3的测试集指标:")
print(f"RMSE: {rmse:.10f}")
print(f"R2:   {r2:.10f}")
print(f"RPIQ: {rpiq:.10f}")
print(f"MAE:  {mae:.10f}")
print(f"MEC:  {mec:.10f}")
print(f"CCC:  {ccc:.10f}")

# 更新原始JSON
with open(original_json_path, 'r', encoding='utf-8') as f:
    original_data = json.load(f)

old_best = original_data['best_dict'].copy()

original_data['best_dict'] = {
    'RMSE': float(rmse),
    'R2': float(r2),
    'RPIQ': float(rpiq),
    'MAE': float(mae),
    'MEC': float(mec),
    'CCC': float(ccc)
}

with open(original_json_path, 'w', encoding='utf-8') as f:
    json.dump(original_data, f, indent=4, ensure_ascii=False)

print("\n" + "=" * 60)
print("对比（Seed 6 vs Seed 3）:")
print("=" * 60)
print(f"旧 RMSE (Seed 6): {old_best['RMSE']:.10f} -> 新 RMSE (Seed 3): {rmse:.10f}")
print(f"旧 R2:   {old_best['R2']:.10f} -> 新 R2:   {r2:.10f}")
print(f"旧 RPIQ: {old_best['RPIQ']:.10f} -> 新 RPIQ: {rpiq:.10f}")
print(f"旧 MAE:  {old_best['MAE']:.10f} -> 新 MAE:  {mae:.10f}")
print(f"旧 MEC:  {old_best['MEC']:.10f} -> 新 MEC:  {mec:.10f}")
print(f"旧 CCC:  {old_best['CCC']:.10f} -> 新 CCC:  {ccc:.10f}")

print(f"\n原始JSON已更新: {original_json_path}")
















