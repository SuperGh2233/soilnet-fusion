"""
支持静态特征的中国数据集加载器
扩展 ChinaSNDatasetClimate 以支持土地利用、土壤质地等静态特征
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from skimage import io
import os
import pandas as pd
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# 导入基础类和函数
try:
    from dataset.dataset_loader_china import ChinaSNDatasetClimate, reshape_tensor, reshape_array
except ImportError:
    from .dataset_loader_china import ChinaSNDatasetClimate, reshape_tensor, reshape_array


class ChinaSNDatasetClimateStatic(ChinaSNDatasetClimate):
    """
    支持静态特征的气候数据集加载器
    继承自 ChinaSNDatasetClimate，添加静态特征支持
    """
    
    def __init__(self, l8_dir, csv_dir, climate_csv_folder,
                 static_csv_path=None,  # 静态特征CSV路径
                 l8_bands=None, transform=None,
                 dates=None, climate_dtype=torch.float32, 
                 normalize_climate=True, return_point_id=False):
        """
        Args:
            static_csv_path: 预处理后的静态特征CSV路径（包含归一化后的特征）
        """
        # 调用父类初始化
        super().__init__(l8_dir, csv_dir, climate_csv_folder,
                        l8_bands=l8_bands, transform=transform,
                        dates=dates, climate_dtype=climate_dtype,
                        normalize_climate=normalize_climate, 
                        return_point_id=return_point_id)
        
        # 加载静态特征
        self.static_features = None
        self.static_feature_cols = None
        
        if static_csv_path and os.path.exists(static_csv_path):
            print(f"加载静态特征: {static_csv_path}")
            static_df = pd.read_csv(static_csv_path)
            
            # 识别Point_ID列
            pid_col = None
            for col in static_df.columns:
                if col.lower() in ['point_id', 'pointid', 'pid']:
                    pid_col = col
                    break
            
            if pid_col is None:
                raise ValueError(f"静态特征CSV中未找到Point_ID列: {static_csv_path}")
            
            # 识别特征列（排除Point_ID、Latitude、Longitude等）
            exclude_cols = {pid_col, 'Latitude', 'Longitude', 'SOC', 'Year'}
            self.static_feature_cols = [col for col in static_df.columns 
                                       if col not in exclude_cols]
            
            if len(self.static_feature_cols) == 0:
                raise ValueError(f"静态特征CSV中未找到特征列: {static_csv_path}")
            
            # 将非数值型（如LULC/CLCD）保留为“索引”以供Embedding；数值列做均值填充
            self.static_category_maps = {}
            feats_df_raw = static_df[self.static_feature_cols].copy()
            categorical_cols = []
            numeric_cols = []
            for col in feats_df_raw.columns:
                if feats_df_raw[col].dtype == object or str(feats_df_raw[col].dtype).startswith('category'):
                    categorical_cols.append(col)
                else:
                    numeric_cols.append(col)

            # 只支持单一主类别列作为Embedding（如CLCD/LULC），如有多个，取第一个
            self.lulc_col = None
            if len(categorical_cols) > 0:
                self.lulc_col = categorical_cols[0]
            
            # 数值列：转换为数值并缺失填充为均值
            feats_num = pd.DataFrame(index=feats_df_raw.index)
            if len(numeric_cols) > 0:
                feats_num = feats_df_raw[numeric_cols].apply(pd.to_numeric, errors='coerce')
                for col in numeric_cols:
                    if feats_num[col].isna().any():
                        feats_num[col] = feats_num[col].fillna(feats_num[col].mean())

            # 保存数值列次序，用于模型输入维度
            self.static_numeric_cols = list(feats_num.columns)

            # 类别列：保存索引映射（用于Embedding），仅对选定的lulc_col生效
            self.lulc_index_map = {}
            self.num_lulc_classes = 0
            if self.lulc_col is not None:
                series = feats_df_raw[self.lulc_col].astype(str).str.strip()
                codes, uniques = pd.factorize(series)
                self.lulc_index_map = {str(v): int(i) for i, v in enumerate(uniques)}
                self.num_lulc_classes = len(uniques)
                lulc_codes = codes.astype(np.int64)
            else:
                # 如果没有类别列，使用-1占位
                lulc_codes = np.full((feats_df_raw.shape[0],), -1, dtype=np.int64)

            # 归一化数值特征（关键修复：神经网络需要归一化的特征）
            self.static_scaler = None
            numeric_values = feats_num[self.static_numeric_cols].values.astype(np.float32) if len(self.static_numeric_cols) > 0 else np.zeros((feats_df_raw.shape[0], 0), dtype=np.float32)
            
            if len(self.static_numeric_cols) > 0 and numeric_values.shape[0] > 0:
                # 使用StandardScaler进行标准化（均值0，方差1）
                # 这与气候特征的归一化方式保持一致
                self.static_scaler = StandardScaler()
                numeric_values_normalized = self.static_scaler.fit_transform(numeric_values)
                print(f"  ✓ 已对静态数值特征进行标准化（StandardScaler）")
                
                # 格式化统计信息（处理数组格式化问题）
                n_show = min(3, len(self.static_numeric_cols))
                mean_before = numeric_values.mean(axis=0)[:n_show]
                std_before = numeric_values.std(axis=0)[:n_show]
                mean_after = numeric_values_normalized.mean(axis=0)[:n_show]
                std_after = numeric_values_normalized.std(axis=0)[:n_show]
                
                print(f"    归一化前统计: mean={mean_before}, std={std_before}")
                print(f"    归一化后统计: mean={mean_after}, std={std_after}")
            else:
                numeric_values_normalized = numeric_values
            
            # 构建查找表
            self.static_numeric_features = {}
            self.lulc_indices = {}
            for idx, row in static_df.iterrows():
                point_id = str(row[pid_col]).replace('.0', '').strip()
                self.static_numeric_features[point_id] = numeric_values_normalized[idx]
                self.lulc_indices[point_id] = int(lulc_codes[idx])
            
            print(f"  静态数值特征维度: {len(self.static_numeric_cols)}")
            if self.lulc_col is not None:
                print(f"  类别特征（用于Embedding）: {self.lulc_col}（类别数={self.num_lulc_classes}）")
            print(f"  加载的静态特征样本数: {len(self.static_numeric_features)}")
        else:
            if static_csv_path:
                print(f"警告: 静态特征文件不存在，将不使用静态特征: {static_csv_path}")
    
    def __getitem__(self, index):
        """
        返回: (影像, 气候数据, 静态数值特征, LULC索引[可选]), SOC, (可选: Point_ID)
        """
        # 获取point_id（用于查找静态特征）
        l8_img_name = self.l8_names[index]
        point_id = l8_img_name.split('_')[0]
        
        # 获取基础数据（影像、气候、SOC）
        result = super().__getitem__(index)
        if self.return_point_id:
            (l8_img, clim_arr), socd, _ = result
        else:
            (l8_img, clim_arr), socd = result
        
        # 获取静态数值特征与LULC索引
        static_numeric = None
        lulc_idx = None
        if getattr(self, 'static_numeric_features', None) is not None:
            # 尝试不同的Point_ID格式匹配
            pid_key = str(point_id).replace('.0', '').strip()
            if pid_key in self.static_numeric_features:
                static_numeric = self.static_numeric_features[pid_key]
                lulc_idx = self.lulc_indices.get(pid_key, -1)
            else:
                # 尝试整数匹配
                try:
                    pid_int = int(float(point_id))
                    pid_key_int = str(pid_int)
                    if pid_key_int in self.static_numeric_features:
                        static_numeric = self.static_numeric_features[pid_key_int]
                        lulc_idx = self.lulc_indices.get(pid_key_int, -1)
                except:
                    pass
            
            if static_numeric is None:
                static_numeric = np.zeros(len(getattr(self, 'static_numeric_cols', [])), dtype=np.float32)
                lulc_idx = -1
            
            static_numeric = torch.from_numpy(static_numeric).to(self.clim_dtype)
            lulc_idx = torch.tensor(lulc_idx, dtype=torch.long)
        
        # 返回结果
        if static_numeric is not None:
            if self.return_point_id:
                return (l8_img, clim_arr, static_numeric, lulc_idx), socd, point_id
            else:
                return (l8_img, clim_arr, static_numeric, lulc_idx), socd
        else:
            # 如果没有静态特征，返回原有格式
            if self.return_point_id:
                return (l8_img, clim_arr), socd, point_id
            else:
                return (l8_img, clim_arr), socd
    
    def get_static_feature_dim(self):
        """返回静态特征维度"""
        if getattr(self, 'static_numeric_cols', None) is not None:
            return len(self.static_numeric_cols)
        return 0

    def get_static_numeric_dim(self):
        """返回静态数值特征维度（不含类别）"""
        if getattr(self, 'static_numeric_cols', None) is not None:
            return len(self.static_numeric_cols)
        return 0

    def get_lulc_num_classes(self):
        """返回LULC类别数（若无类别列则为0）"""
        return int(getattr(self, 'num_lulc_classes', 0))

