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
                 normalize_climate=True, return_point_id=False,
                 point_ids=None, climate_stats=None, fit_climate_stats=False,
                 static_stats=None, fit_static_stats=False):
        """
        Args:
            static_csv_path: 预处理后的静态特征CSV路径（包含归一化后的特征）
        """
        # 调用父类初始化
        super().__init__(l8_dir, csv_dir, climate_csv_folder,
                        l8_bands=l8_bands, transform=transform,
                        dates=dates, climate_dtype=climate_dtype,
                        normalize_climate=normalize_climate, 
                        return_point_id=return_point_id,
                        point_ids=point_ids,
                        climate_stats=climate_stats,
                        fit_climate_stats=fit_climate_stats)
        
        # 加载静态特征
        self.static_features = None
        self.static_feature_cols = None
        self.static_stats = None
        
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
            
            # 将非数值型（如LULC/CLCD）保留为“索引”以供Embedding。
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
            
            pid_values = static_df[pid_col].map(lambda value: str(value).replace('.0', '').strip())
            training_ids = {
                str(name).split('_')[0].replace('.0', '').strip()
                for name in self.l8_names
            }
            fit_mask = pid_values.isin(training_ids) if fit_static_stats else pd.Series(True, index=static_df.index)
            if fit_static_stats and not fit_mask.any():
                raise ValueError('No training rows found while fitting static preprocessing stats')

            if static_stats is None:
                feats_num = feats_df_raw[numeric_cols].apply(pd.to_numeric, errors='coerce')
                fit_numeric = feats_num.loc[fit_mask]
                means = fit_numeric.mean()
                fit_filled = fit_numeric.fillna(means)
                scales = fit_filled.std(ddof=0).replace(0, 1.0).fillna(1.0)
                categories = []
                if self.lulc_col is not None:
                    categories = list(dict.fromkeys(
                        feats_df_raw.loc[fit_mask, self.lulc_col]
                        .dropna().astype(str).str.strip().tolist()
                    ))
                self.static_stats = {
                    'numeric_cols': list(numeric_cols),
                    'mean': {col: float(means[col]) for col in numeric_cols},
                    'scale': {col: float(scales[col]) for col in numeric_cols},
                    'lulc_col': self.lulc_col,
                    'lulc_categories': categories,
                }
            else:
                self.static_stats = static_stats
                numeric_cols = list(self.static_stats.get('numeric_cols', []))
                self.lulc_col = self.static_stats.get('lulc_col')

            self.static_numeric_cols = list(numeric_cols)
            feats_num = feats_df_raw[self.static_numeric_cols].apply(pd.to_numeric, errors='coerce')
            means = pd.Series(self.static_stats.get('mean', {}), dtype=float)
            scales = pd.Series(self.static_stats.get('scale', {}), dtype=float).replace(0, 1.0)
            feats_num = feats_num.fillna(means)
            numeric_values_normalized = (
                (feats_num - means) / scales
            ).to_numpy(dtype=np.float32) if self.static_numeric_cols else np.zeros((len(static_df), 0), dtype=np.float32)

            categories = [str(value) for value in self.static_stats.get('lulc_categories', [])]
            self.lulc_index_map = {value: index for index, value in enumerate(categories)}
            self.num_lulc_classes = len(categories)
            if self.lulc_col is not None:
                lulc_codes = (
                    feats_df_raw[self.lulc_col].astype(str).str.strip()
                    .map(self.lulc_index_map).fillna(-1).astype(np.int64).to_numpy()
                )
            else:
                lulc_codes = np.full((len(static_df),), -1, dtype=np.int64)

            if self.static_numeric_cols:
                scope = 'training subset' if fit_static_stats or static_stats is not None else 'complete CSV (legacy mode)'
                print(f"  [OK] 已使用 {scope} 的统计量标准化静态数值特征")
            
            # 构建查找表
            self.static_numeric_features = {}
            self.lulc_indices = {}
            for position, (_, row) in enumerate(static_df.iterrows()):
                point_id = str(row[pid_col]).replace('.0', '').strip()
                self.static_numeric_features[point_id] = numeric_values_normalized[position]
                self.lulc_indices[point_id] = int(lulc_codes[position])
            
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

    def get_static_stats(self):
        return self.static_stats
