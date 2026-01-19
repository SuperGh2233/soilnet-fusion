"""
准备静态特征数据（土地利用、土壤质地等）
处理缺失值、归一化等
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def prepare_static_features(static_csv_path, output_path=None):
    """
    准备静态特征数据
    
    Args:
        static_csv_path: 原始静态特征CSV路径
        output_path: 输出路径（默认在同一目录下添加_processed后缀）
    """
    print("=" * 80)
    print("静态特征数据预处理")
    print("=" * 80)
    
    if not os.path.exists(static_csv_path):
        raise FileNotFoundError(f"未找到文件: {static_csv_path}")
    
    print(f"\n[1] 读取数据: {static_csv_path}")
    df = pd.read_csv(static_csv_path)
    print(f"   原始数据形状: {df.shape}")
    print(f"   列名: {list(df.columns)}")
    
    # 识别Point_ID列
    pid_col = None
    for col in df.columns:
        if col.lower() in ['point_id', 'pointid', 'pid', 'point_id']:
            pid_col = col
            break
    
    if pid_col is None:
        raise ValueError("未找到Point_ID列！请确保CSV中包含Point_ID或类似列名")
    
    print(f"\n[2] 识别Point_ID列: {pid_col}")
    
    # 识别静态特征列（排除Point_ID、Latitude、Longitude等）
    exclude_cols = {pid_col, 'Latitude', 'Longitude', 'SOC', 'Year', 'system:index'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"   静态特征列 ({len(feature_cols)}): {feature_cols}")
    
    # 处理分类特征（土地利用等）
    categorical_cols = []
    numerical_cols = []
    
    for col in feature_cols:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    
    print(f"\n[3] 特征分类:")
    print(f"   数值特征 ({len(numerical_cols)}): {numerical_cols}")
    print(f"   分类特征 ({len(categorical_cols)}): {categorical_cols}")
    
    # 处理分类特征：Label Encoding
    label_encoders = {}
    df_processed = df.copy()
    
    for col in categorical_cols:
        print(f"\n   处理分类特征 '{col}':")
        # 处理缺失值（用"Unknown"填充）
        df_processed[col] = df_processed[col].fillna('Unknown')
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
        print(f"     编码后唯一值数量: {df_processed[col].nunique()}")
        print(f"     唯一值: {df_processed[col].unique()[:10]}")
    
    # 处理数值特征的缺失值
    print(f"\n[4] 处理数值特征缺失值:")
    for col in numerical_cols:
        missing_count = df_processed[col].isna().sum()
        if missing_count > 0:
            print(f"   {col}: {missing_count} 个缺失值 ({missing_count/len(df)*100:.1f}%)")
    
    # 使用KNN填充缺失值
    if len(numerical_cols) > 0:
        # 准备KNN填充的数据
        knn_data = df_processed[numerical_cols].copy()
        
        # 如果所有列都有缺失，先用均值填充
        for col in numerical_cols:
            if knn_data[col].isna().all():
                knn_data[col] = 0  # 全缺失则填充0
            elif knn_data[col].isna().any():
                knn_data[col] = knn_data[col].fillna(knn_data[col].median())
        
        # KNN填充
        if knn_data.isna().any().any():
            imputer = KNNImputer(n_neighbors=5)
            knn_imputed = imputer.fit_transform(knn_data)
            df_processed[numerical_cols] = pd.DataFrame(knn_imputed, columns=numerical_cols, index=df_processed.index)
            print(f"   完成KNN填充")
    
    # 归一化数值特征
    print(f"\n[5] 归一化数值特征 (Min-Max Scaling):")
    scaler = MinMaxScaler()
    if len(numerical_cols) > 0:
        df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
        print(f"   已完成 {len(numerical_cols)} 个数值特征的归一化")
    
    # 保存处理后的数据
    if output_path is None:
        base_name = os.path.splitext(static_csv_path)[0]
        output_path = f"{base_name}_processed.csv"
    
    # 确保列顺序：Point_ID, Latitude, Longitude (如果存在), 然后是特征列
    output_cols = [pid_col]
    if 'Latitude' in df_processed.columns:
        output_cols.append('Latitude')
    if 'Longitude' in df_processed.columns:
        output_cols.append('Longitude')
    output_cols.extend(sorted(feature_cols))
    
    df_output = df_processed[output_cols]
    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n[6] 保存处理后的数据:")
    print(f"   输出路径: {output_path}")
    print(f"   数据形状: {df_output.shape}")
    print(f"   特征统计:")
    for col in feature_cols:
        print(f"     {col}: min={df_output[col].min():.4f}, max={df_output[col].max():.4f}, mean={df_output[col].mean():.4f}")
    
    # 保存编码器和缩放器信息（用于预测时的转换）
    import pickle
    meta_path = output_path.replace('.csv', '_metadata.pkl')
    metadata = {
        'label_encoders': label_encoders,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'pid_col': pid_col
    }
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"\n   元数据已保存: {meta_path}")
    
    print("\n" + "=" * 80)
    print("预处理完成！")
    print("=" * 80)
    
    return df_output, metadata

if __name__ == "__main__":
    # 示例用法
    # 假设静态特征CSV路径
    static_csv_path = "dataset/static_features.csv"  # 用户需要提供实际路径
    
    if len(sys.argv) > 1:
        static_csv_path = sys.argv[1]
    
    if not os.path.exists(static_csv_path):
        print(f"错误: 文件不存在: {static_csv_path}")
        print("\n使用方法:")
        print("  python prepare_static_features.py <静态特征CSV路径>")
        print("\nCSV文件应包含:")
        print("  - Point_ID: 点位ID（必须）")
        print("  - LandUse: 土地利用类型（分类变量）")
        print("  - Clay_Content: 黏粒含量（数值，%）")
        print("  - CEC: 阳离子交换量（数值，cmol/kg）")
        print("  - 其他土壤属性...")
        sys.exit(1)
    
    prepare_static_features(static_csv_path)

