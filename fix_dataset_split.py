#!/usr/bin/env python3
"""
修复数据集划分问题
确保训练集、验证集和测试集没有重叠
"""

import os
import shutil
import random
import pandas as pd

def fix_dataset_split():
    """修复数据集划分，确保没有重叠"""
    
    # 配置参数
    csv_file = 'dataset/CN-SOC-3500.csv'  # 您的CSV文件路径
    img_root = 'dataset/l8_images_CN'     # 原始图像文件夹
    dst_root = 'dataset/l8_images_CN_split_fixed'  # 修复后的文件夹
    
    # 划分比例
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    print("开始修复数据集划分...")
    
    # 1. 读取CSV文件
    if not os.path.exists(csv_file):
        print(f"错误: CSV文件不存在: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    print(f"CSV文件包含 {len(df)} 个样本")
    
    # 2. 获取所有可用的图像文件
    if not os.path.exists(img_root):
        print(f"错误: 图像文件夹不存在: {img_root}")
        return
    
    all_images = [f for f in os.listdir(img_root) if f.endswith('.tif')]
    print(f"图像文件夹包含 {len(all_images)} 个图像文件")
    
    # 3. 检查CSV和图像的匹配
    available_point_ids = set()
    for img_name in all_images:
        point_id = img_name.split('_')[0]
        available_point_ids.add(point_id)
    
    # 过滤CSV中存在的样本
    df_filtered = df[df['Point_id'].astype(str).isin(available_point_ids)]
    print(f"匹配的样本数: {len(df_filtered)}")
    
    # 4. 随机划分样本ID（不是图像文件）
    point_ids = df_filtered['Point_id'].astype(str).unique().tolist()
    random.seed(42)  # 固定随机种子
    random.shuffle(point_ids)
    
    total_points = len(point_ids)
    train_end = int(total_points * train_ratio)
    val_end = train_end + int(total_points * val_ratio)
    
    train_ids = set(point_ids[:train_end])
    val_ids = set(point_ids[train_end:val_end])
    test_ids = set(point_ids[val_end:])
    
    print(f"训练集: {len(train_ids)} 个样本")
    print(f"验证集: {len(val_ids)} 个样本")
    print(f"测试集: {len(test_ids)} 个样本")
    
    # 5. 检查是否有重叠
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("警告: 发现重叠样本!")
        if train_val_overlap:
            print(f"训练集-验证集重叠: {len(train_val_overlap)}")
        if train_test_overlap:
            print(f"训练集-测试集重叠: {len(train_test_overlap)}")
        if val_test_overlap:
            print(f"验证集-测试集重叠: {len(val_test_overlap)}")
    else:
        print("✅ 没有重叠样本")
    
    # 6. 创建目标文件夹
    train_folder = os.path.join(dst_root, 'train')
    val_folder = os.path.join(dst_root, 'val')
    test_folder = os.path.join(dst_root, 'test')
    
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    # 7. 复制图像文件到对应文件夹
    def copy_images_by_point_id(point_ids_set, dst_folder):
        """根据点ID复制对应的图像文件"""
        copied_count = 0
        for img_name in all_images:
            point_id = img_name.split('_')[0]
            if point_id in point_ids_set:
                src_path = os.path.join(img_root, img_name)
                dst_path = os.path.join(dst_folder, img_name)
                shutil.copy2(src_path, dst_path)
                copied_count += 1
        return copied_count
    
    print("\n开始复制文件...")
    train_count = copy_images_by_point_id(train_ids, train_folder)
    val_count = copy_images_by_point_id(val_ids, val_folder)
    test_count = copy_images_by_point_id(test_ids, test_folder)
    
    print(f"训练集: 复制了 {train_count} 个图像文件")
    print(f"验证集: 复制了 {val_count} 个图像文件")
    print(f"测试集: 复制了 {test_count} 个图像文件")
    
    # 8. 创建对应的CSV文件
    def create_csv_by_point_id(point_ids_set, dst_folder, suffix):
        """为每个数据集创建对应的CSV文件"""
        subset_df = df_filtered[df_filtered['Point_id'].astype(str).isin(point_ids_set)]
        csv_path = os.path.join(dst_folder, f'data_{suffix}.csv')
        subset_df.to_csv(csv_path, index=False)
        print(f"{suffix} CSV: {len(subset_df)} 个样本 -> {csv_path}")
    
    create_csv_by_point_id(train_ids, train_folder, 'train')
    create_csv_by_point_id(val_ids, val_folder, 'val')
    create_csv_by_point_id(test_ids, test_folder, 'test')
    
    print(f"\n✅ 数据集划分修复完成!")
    print(f"修复后的文件夹: {dst_root}")
    print(f"请更新您的训练脚本中的路径:")
    print(f"  train_l8_folder_path = '{train_folder}'")
    print(f"  val_l8_folder_path = '{val_folder}'")
    print(f"  test_l8_folder_path = '{test_folder}'")
    print(f"  lucas_csv_path = '{train_folder}/data_train.csv'")

if __name__ == "__main__":
    fix_dataset_split()
