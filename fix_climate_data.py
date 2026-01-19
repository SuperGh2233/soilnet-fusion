"""
修复气候数据处理脚本
处理中国数据集的气候数据文件
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def fix_climate_data(climate_folder_path, output_folder_path=None):
    """
    修复气候数据文件
    
    Args:
        climate_folder_path: 原始气候数据文件夹路径
        output_folder_path: 输出文件夹路径（如果为None，则覆盖原文件）
    """
    
    if output_folder_path is None:
        output_folder_path = climate_folder_path
    
    # 确保输出目录存在
    os.makedirs(output_folder_path, exist_ok=True)
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(climate_folder_path) if f.endswith('.csv')]
    print(f"找到 {len(csv_files)} 个气候数据文件")
    
    # 处理每个文件
    for csv_file in csv_files:
        print(f"\n处理文件: {csv_file}")
        
        # 读取原始数据
        df = pd.read_csv(os.path.join(climate_folder_path, csv_file))
        print(f"  原始数据形状: {df.shape}")
        
        # 识别日期列（8位数字格式）
        date_columns = [col for col in df.columns if str(col).isdigit() and len(str(col)) == 8]
        print(f"  找到 {len(date_columns)} 个日期列")
        
        # 识别Point_id列（处理大小写变化）
        point_id_col = None
        for col in df.columns:
            if col.lower() in ['point_id', 'pointid', 'pid']:
                point_id_col = col
                break
        
        if point_id_col is None:
            print(f"  警告: 未找到Point_id列，跳过文件 {csv_file}")
            continue
        
        print(f"  使用Point_id列: {point_id_col}")
        
        # 创建新的DataFrame
        new_df = pd.DataFrame()
        new_df['Point_ID'] = df[point_id_col]  # 统一使用Point_ID
        
        # 处理日期列
        for date_col in date_columns:
            # 将-999替换为NaN
            values = df[date_col].replace(-999, np.nan)
            new_df[date_col] = values
        
        # 检查数据质量
        print(f"  数据质量检查:")
        print(f"    - 总样本数: {len(new_df)}")
        print(f"    - 有效Point_ID数: {new_df['Point_ID'].notna().sum()}")
        
        # 检查每个气候变量的数据范围
        for date_col in date_columns[:3]:  # 只显示前3个日期列
            valid_data = new_df[date_col].dropna()
            if len(valid_data) > 0:
                print(f"    - {date_col}: 范围 [{valid_data.min():.2f}, {valid_data.max():.2f}], 有效值 {len(valid_data)}")
        
        # 保存修复后的文件
        output_file = os.path.join(output_folder_path, f"fixed_{csv_file}")
        new_df.to_csv(output_file, index=False)
        print(f"  保存到: {output_file}")
    
    print(f"\n气候数据修复完成！")
    print(f"修复后的文件保存在: {output_folder_path}")

def create_normalized_climate_data(climate_folder_path, output_folder_path):
    """
    创建归一化的气候数据文件
    
    Args:
        climate_folder_path: 修复后的气候数据文件夹路径
        output_folder_path: 归一化数据输出路径
    """
    
    os.makedirs(output_folder_path, exist_ok=True)
    
    # 获取所有修复后的CSV文件
    csv_files = [f for f in os.listdir(climate_folder_path) if f.startswith('fixed_') and f.endswith('.csv')]
    print(f"\n开始归一化处理，找到 {len(csv_files)} 个文件")
    
    for csv_file in csv_files:
        print(f"\n归一化文件: {csv_file}")
        
        # 读取修复后的数据
        df = pd.read_csv(os.path.join(climate_folder_path, csv_file))
        
        # 识别日期列
        date_columns = [col for col in df.columns if str(col).isdigit() and len(str(col)) == 8]
        
        # 创建新的DataFrame
        new_df = pd.DataFrame()
        new_df['Point_ID'] = df['Point_ID']
        
        # 对每个日期列进行归一化
        scaler = MinMaxScaler()
        
        for date_col in date_columns:
            # 获取有效数据（非NaN）
            valid_mask = df[date_col].notna()
            valid_data = df.loc[valid_mask, date_col].values.reshape(-1, 1)
            
            if len(valid_data) > 0:
                # 归一化
                normalized_data = scaler.fit_transform(valid_data).flatten()
                
                # 创建完整列（包含NaN）
                full_column = np.full(len(df), np.nan)
                full_column[valid_mask] = normalized_data
                new_df[date_col] = full_column
            else:
                # 如果所有数据都是NaN，保持NaN
                new_df[date_col] = df[date_col]
        
        # 保存归一化后的文件
        output_file = os.path.join(output_folder_path, f"normalized_{csv_file.replace('fixed_', '')}")
        new_df.to_csv(output_file, index=False)
        print(f"  保存到: {output_file}")
        
        # 显示归一化后的数据范围
        for date_col in date_columns[:3]:
            valid_data = new_df[date_col].dropna()
            if len(valid_data) > 0:
                print(f"    - {date_col}: 归一化范围 [{valid_data.min():.3f}, {valid_data.max():.3f}]")
    
    print(f"\n气候数据归一化完成！")
    print(f"归一化后的文件保存在: {output_folder_path}")

def test_climate_data_loading(climate_folder_path):
    """
    测试气候数据加载
    """
    print(f"\n测试气候数据加载...")
    
    # 获取一个示例文件
    csv_files = [f for f in os.listdir(climate_folder_path) if f.endswith('.csv')]
    if not csv_files:
        print("未找到CSV文件")
        return
    
    test_file = csv_files[0]
    print(f"测试文件: {test_file}")
    
    df = pd.read_csv(os.path.join(climate_folder_path, test_file))
    
    # 识别Point_id列
    point_id_col = None
    for col in df.columns:
        if col.lower() in ['point_id', 'pointid', 'pid']:
            point_id_col = col
            break
    
    if point_id_col:
        print(f"Point_id列: {point_id_col}")
        print(f"Point_id示例: {df[point_id_col].head().tolist()}")
    
    # 识别日期列
    date_columns = [col for col in df.columns if str(col).isdigit() and len(str(col)) == 8]
    print(f"日期列数量: {len(date_columns)}")
    print(f"日期列示例: {date_columns[:5]}")
    
    # 检查数据范围
    if date_columns:
        sample_col = date_columns[0]
        valid_data = df[sample_col].replace(-999, np.nan).dropna()
        print(f"示例列 {sample_col} 数据范围: [{valid_data.min():.2f}, {valid_data.max():.2f}]")
        print(f"有效数据比例: {len(valid_data)/len(df)*100:.1f}%")

if __name__ == "__main__":
    # 设置路径
    climate_folder = "dataset/Climate/CN/CN"
    fixed_folder = "dataset/Climate/CN/CN_fixed"
    normalized_folder = "dataset/Climate/CN/CN_normalized"
    
    print("=== 气候数据处理脚本 ===")
    
    # 1. 测试原始数据
    test_climate_data_loading(climate_folder)
    
    # 2. 修复数据
    fix_climate_data(climate_folder, fixed_folder)
    
    # 3. 创建归一化数据
    create_normalized_climate_data(fixed_folder, normalized_folder)
    
    print("\n=== 处理完成 ===")
    print("建议在config.py中更新气候数据路径为:")
    print(f"climate_csv_folder_path = '{normalized_folder}'")


