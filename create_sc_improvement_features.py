#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
华南地区改进方案: 生成P-PET水分盈亏特征
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def create_water_deficit_feature():
    """创建P-PET（水分盈亏）特征"""
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
