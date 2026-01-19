"""
兼容版区域性能验证脚本
使用与训练时相同的模型架构
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from dataset.dataset_loader_china import ChinaSNDatasetClimate
from dataset.dataset_loader_china_static import ChinaSNDatasetClimateStatic
from soilnet.soil_net import SoilNetLSTM
from soilnet.soil_net_static import SoilNetLSTMWithStatic

def custom_collate_fn(batch):
    """自定义collate函数，处理不同尺寸的图像与可选静态特征"""
    # 兼容两种返回格式：
    # - ((images, climate), targets, point_ids)
    # - ((images, climate, static), targets, point_ids)
    # - ((images, climate, static_numeric, lulc_idx), targets, point_ids)
    data_tuples, targets, point_ids = zip(*batch)

    # 分离images、climate、static(可选)
    first_tuple = data_tuples[0]
    has_static3 = len(first_tuple) == 3
    has_static4 = len(first_tuple) == 4
    if has_static4:
        images, climate, static, lulc_idx = zip(*data_tuples)
    elif has_static3:
        images, climate, static = zip(*data_tuples)
        lulc_idx = None
    else:
        images, climate = zip(*data_tuples)
        static = None
        lulc_idx = None

    # 找到最大尺寸
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    # 填充所有图像到相同尺寸
    padded_images = []
    for img in images:
        # 确保图像是张量
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).float()
        
        if img.shape[1] < max_h or img.shape[2] < max_w:
            # 使用零填充
            pad_h = max_h - img.shape[1]
            pad_w = max_w - img.shape[2]
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_images.append(img)

    # 堆叠张量
    images = torch.stack(padded_images)
    # 确保气候数据是张量
    climate_tensors = []
    for clim in climate:
        if not isinstance(clim, torch.Tensor):
            clim = torch.from_numpy(clim).float()
        climate_tensors.append(clim)
    climate = torch.stack(climate_tensors)

    # 处理静态特征（如果存在）
    static_tensor = None
    if (has_static3 or has_static4) and static is not None:
        static_tensors = []
        for st in static:
            if not isinstance(st, torch.Tensor):
                st = torch.from_numpy(st).float()
            static_tensors.append(st)
        static_tensor = torch.stack(static_tensors)

    # 处理LULC索引（如果存在）
    lulc_tensor = None
    if has_static4 and (lulc_idx is not None):
        lulc_tensor_list = []
        for li in lulc_idx:
            if isinstance(li, torch.Tensor):
                li_t = li.long()
            else:
                li_t = torch.tensor(li, dtype=torch.long)
            lulc_tensor_list.append(li_t)
        lulc_tensor = torch.stack(lulc_tensor_list)
    # 确保目标数据是张量
    targets_tensors = []
    for target in targets:
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(np.array(target)).float()
        targets_tensors.append(target)
    targets = torch.stack(targets_tensors)
    point_ids = list(point_ids)

    if (static_tensor is not None) and (lulc_tensor is not None):
        return images, climate, static_tensor, lulc_tensor, targets, point_ids
    elif static_tensor is not None:
        return images, climate, static_tensor, targets, point_ids
    else:
        return images, climate, targets, point_ids

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

def load_best_model_compatible(model_path, device='cpu', static_feature_dim=0, lulc_num_classes=0,
                               seq_len=None, lstm_n_features_override=None):
    """Compatible loading of best model using same architecture as training"""
    print(f"=== Loading Best Model (Compatible Version) ===")
    print(f"Model path: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        print(f"Model checkpoint info:")
        for key in checkpoint.keys():
            if key != 'state_dict':
                print(f"  {key}: {checkpoint[key]}")
        
        # 自动检测气候特征维度
        lstm_n_features = lstm_n_features_override
        # 方法1: 从checkpoint的state_dict中推断
        if lstm_n_features is None and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            if 'lstm.project_inp.weight' in state_dict:
                weight_shape = state_dict['lstm.project_inp.weight'].shape
                if len(weight_shape) >= 2:
                    lstm_n_features = weight_shape[1]  # 第二个维度是输入特征数
                    print(f"✓ Inferred lstm_n_features from checkpoint: {lstm_n_features}")
        
        # 方法2: 从config读取（如果前面没有找到）
        if lstm_n_features is None:
            climate_folder = config.climate_csv_folder_path
            if os.path.exists(climate_folder):
                csv_files = [f for f in os.listdir(climate_folder) if f.endswith('.csv')]
                lstm_n_features = len(csv_files)
                print(f"✓ Auto-detected lstm_n_features from config folder: {lstm_n_features}")
            else:
                # 默认值（fallback）
                lstm_n_features = 13  # 新的气候数据是13个变量
                print(f"⚠ Using default lstm_n_features: {lstm_n_features}")

        # 序列长度与训练保持一致
        inferred_seq_len = seq_len if seq_len is not None else 61
        if seq_len is None:
            print(f"⚠ seq_len not provided, fallback to default {inferred_seq_len}")
        else:
            print(f"✓ Inferred seq_len from test data: {inferred_seq_len}")
        
        # Create model instance with same architecture as training
        # If static_feature_dim > 0, use static-enabled model
        common_kwargs = dict(
            cnn_arch="ViT-CoMer",
            rnn_arch="Transformer",
            cnn_in_channels=14,  # 训练时包含SRTM，共14个通道
            regresor_input_from_cnn=384,  # 与训练一致
            lstm_n_features=lstm_n_features,
            lstm_n_layers=2,
            lstm_out=128,
            hidden_size=128,
            seq_len=inferred_seq_len,
            use_spectral_enhance=False,
            spectral_type='hybrid'
        )

        if static_feature_dim and static_feature_dim > 0:
            print(f"Using SoilNetLSTMWithStatic (static_feature_dim={static_feature_dim}, lulc_num_classes={lulc_num_classes})")
            model = SoilNetLSTMWithStatic(static_feature_dim=static_feature_dim, lulc_num_classes=lulc_num_classes, **common_kwargs)
        else:
            model = SoilNetLSTM(**common_kwargs)
        
        # 加载模型权重
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 过滤掉形状不匹配的权重，防止加载报错
        model_state = model.state_dict()
        filtered_state_dict = {}
        skipped_keys = []
        for key, tensor in state_dict.items():
            if key in model_state and model_state[key].shape == tensor.shape:
                filtered_state_dict[key] = tensor
            else:
                skipped_keys.append(key)
        if skipped_keys:
            print(f"⚠ Skipping {len(skipped_keys)} keys due to shape mismatch: {skipped_keys[:5]}")

        # 使用strict=False允许部分权重不匹配
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        
        print(f"Model loaded successfully")
        print(f"Missing keys count: {len(missing_keys)}")
        print(f"Unexpected keys count: {len(unexpected_keys)}")
        
        if missing_keys:
            print(f"Missing keys example: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Unexpected keys example: {unexpected_keys[:5]}...")
        
        model.eval()
        return model
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise e

def load_test_data():
    """加载测试数据"""
    print(f"\n=== Loading Test Data ===")
    
    # 导入训练时使用的数据预处理
    from dataset.dataset_loader import myNormalize, myToTensor
    from torchvision import transforms
    
    # 使用与训练时相同的数据预处理
    OC_MAX = 30.8  # 与训练时一致
    mynorm = myNormalize(img_bands_min_max=[[(0,7),(0,1)], [(7,12),(-1,1)], [(12), (-4,2963)], [(13), (0, 90)]], 
                        oc_min=0, oc_max=OC_MAX)
    my_to_tensor = myToTensor()
    test_transform = transforms.Compose([mynorm, my_to_tensor])
    
    # 根据是否配置静态特征选择数据集
    static_csv = getattr(config, 'static_csv_path', None)
    use_static = static_csv is not None and os.path.exists(static_csv)

    if use_static:
        test_dataset = ChinaSNDatasetClimateStatic(
            l8_dir=config.test_l8_folder_path,
            csv_dir=config.test_csv_path,
            climate_csv_folder=config.climate_csv_folder_path,
            static_csv_path=static_csv,
            l8_bands=config.bands,
            transform=test_transform,
            dates=None,
            climate_dtype=torch.float32,
            normalize_climate=True,
            return_point_id=True
        )
        # 新接口：数值静态维度与LULC类别数
        if hasattr(test_dataset, 'get_static_numeric_dim'):
            static_dim = test_dataset.get_static_numeric_dim()
        else:
            static_dim = test_dataset.get_static_feature_dim()
        lulc_classes = test_dataset.get_lulc_num_classes() if hasattr(test_dataset, 'get_lulc_num_classes') else 0
        print(f"Detected static features: numeric_dim={static_dim}, lulc_classes={lulc_classes}")
    else:
        test_dataset = ChinaSNDatasetClimate(
            l8_dir=config.test_l8_folder_path,
            csv_dir=config.test_csv_path,
            climate_csv_folder=config.climate_csv_folder_path,
            l8_bands=config.bands,
            transform=test_transform,  # 使用与训练时相同的预处理
            dates=None,
            climate_dtype=torch.float32,
            normalize_climate=True,
            return_point_id=True
        )
        static_dim = 0
        lulc_classes = 0
    
    # 过滤掉包含NaN的图像文件
    print("过滤包含NaN的图像文件...")
    valid_indices = []
    for i in range(len(test_dataset)):
        try:
            sample = test_dataset[i]
            # 兼容 Subset 包装后的返回
            if isinstance(sample, tuple) and len(sample) >= 2:
                data_tuple = sample[0]
                # 兼容 (img, clim) 或 (img, clim, static)
                if isinstance(data_tuple, tuple) and len(data_tuple) >= 2:
                    l8_img = data_tuple[0]
                else:
                    # 异常格式，跳过
                    print(f"跳过样本 {i}: unexpected data format")
                    continue
            else:
                print(f"跳过样本 {i}: unexpected sample format")
                continue

            # 检查图像是否包含NaN
            if isinstance(l8_img, np.ndarray):
                has_nan = np.isnan(l8_img).any()
            else:
                has_nan = torch.isnan(l8_img).any()

            if not has_nan:
                valid_indices.append(i)
        except Exception as e:
            print(f"跳过样本 {i}: {e}")
            continue
    
    print(f"原始样本数: {len(test_dataset)}")
    print(f"有效样本数: {len(valid_indices)}")
    
    # 创建子集
    test_dataset = torch.utils.data.Subset(test_dataset, valid_indices)
    
    print(f"Test samples count: {len(test_dataset)}")

    if len(test_dataset) == 0:
        raise ValueError("No valid samples found in test dataset after filtering NaNs.")

    # 推断气候序列长度与特征维度
    sample0 = test_dataset[0]
    if isinstance(sample0, tuple):
        data_tuple0 = sample0[0]
    else:
        data_tuple0 = sample0
    if not (isinstance(data_tuple0, tuple) and len(data_tuple0) >= 2):
        raise ValueError("Unexpected data format when inferring climate tensor shape.")
    climate_tensor = data_tuple0[1]
    if not isinstance(climate_tensor, torch.Tensor):
        climate_tensor = torch.tensor(climate_tensor)
    climate_seq_len = int(climate_tensor.shape[0])
    climate_feat_dim = int(climate_tensor.shape[1]) if climate_tensor.dim() > 1 else 1
    print(f"Inferred climate sequence length: {climate_seq_len}, feature dim: {climate_feat_dim}")
    
    # 创建数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=16, 
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate_fn
    )
    
    return test_loader, test_dataset, static_dim, lulc_classes, climate_seq_len, climate_feat_dim

def predict_with_model(model, test_loader, device='cpu'):
    """使用模型进行预测"""
    print(f"\n=== Model Prediction ===")
    
    model.to(device)
    all_predictions = []
    all_targets = []
    all_point_ids = []
    all_latitudes = []
    all_longitudes = []
    skipped_batches = 0
    skipped_samples_due_to_nan = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 兼容三种batch格式
            if len(batch) == 6:
                images, climate, static_feat, lulc_idx, targets, point_ids = batch
            elif len(batch) == 5:
                images, climate, static_feat, targets, point_ids = batch
                lulc_idx = None
            else:
                images, climate, targets, point_ids = batch
                static_feat = None
                lulc_idx = None
            images = images.to(device)
            climate = climate.to(device)
            targets = targets.to(device)
            if static_feat is not None:
                static_feat = static_feat.to(device)
            if lulc_idx is not None and isinstance(lulc_idx, torch.Tensor):
                lulc_idx = lulc_idx.to(device)
            
            try:
                # Debug: Check for NaN in model inputs
                if torch.isnan(images).any():
                    print(f"Warning: NaN detected in input images at batch {batch_idx+1}")
                if torch.isnan(climate).any():
                    print(f"Warning: NaN detected in input climate at batch {batch_idx+1}")
                
                # Check model parameters for NaN
                model_nan_params = []
                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        model_nan_params.append(name)
                if model_nan_params:
                    print(f"Warning: NaN detected in model parameters: {model_nan_params[:3]}...")
                
                # 模型预测（根据是否有静态特征选择输入）
                if (static_feat is not None) and (lulc_idx is not None):
                    outputs = model((images, climate, static_feat, lulc_idx))
                elif static_feat is not None:
                    outputs = model((images, climate, static_feat))
                else:
                    outputs = model((images, climate))

                outputs = outputs.squeeze()
                predictions = outputs.detach().cpu().numpy()
                targets_np = targets.detach().cpu().numpy()
                
                # Debug: Check model output range
                raw_outputs = predictions
                print(f"Debug - Raw model outputs range: [{raw_outputs.min():.4f}, {raw_outputs.max():.4f}]")
                print(f"Debug - Raw targets range: [{targets.cpu().numpy().min():.4f}, {targets.cpu().numpy().max():.4f}]")
                
                # 模型输出已经是归一化的，直接使用
                predictions = raw_outputs
                print("Using model outputs directly (already normalized)")
                
                targets_np = targets_np
                
                # 详细调试：检查模型输出和目标的量纲
                print(f"Raw model outputs: min={raw_outputs.min():.6f}, max={raw_outputs.max():.6f}")
                print(f"Raw targets: min={targets_np.min():.6f}, max={targets_np.max():.6f}")
                
                # 检查数据是否已经归一化
                if targets_np.max() <= 1.0 and targets_np.min() >= 0.0:
                    print("Targets are normalized (0-1 range), applying OC_MAX for denormalization")
                    OC_MAX = 30.8
                    predictions = predictions * OC_MAX
                    targets_np = targets_np * OC_MAX
                    print(f"Denormalized predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
                    print(f"Denormalized targets range: [{targets_np.min():.4f}, {targets_np.max():.4f}]")
                else:
                    print("Targets appear to be in real scale already, no denormalization needed")
                    print(f"Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
                    print(f"Targets range: [{targets_np.min():.4f}, {targets_np.max():.4f}]")
                
                print(f"Final predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
                print(f"Final targets range: [{targets_np.min():.4f}, {targets_np.max():.4f}]")

                # 逐样本过滤NaN，而不是整批跳过
                preds_arr = np.array(predictions).reshape(-1)
                targs_arr = np.array(targets_np).reshape(-1)
                pid_list = list(point_ids)
                valid_mask = np.isfinite(preds_arr) & np.isfinite(targs_arr)
                skipped_in_batch = int((~valid_mask).sum())
                if skipped_in_batch > 0:
                    skipped_samples_due_to_nan += skipped_in_batch
                    print(f"Batch {batch_idx+1}: filtered {skipped_in_batch} samples due to NaN/Inf")

                preds_arr = preds_arr[valid_mask]
                targs_arr = targs_arr[valid_mask]
                pid_list = [pid for i, pid in enumerate(pid_list) if valid_mask[i]]

                all_predictions.extend(preds_arr)
                all_targets.extend(targs_arr)
                all_point_ids.extend(pid_list)
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx+1}/{len(test_loader)} completed")
                    
            except Exception as e:
                print(f"  Batch {batch_idx+1} prediction failed: {e}")
                skipped_batches += 1
                continue
    
    # 获取经纬度信息（构建健壮映射）
    df_test = pd.read_csv(config.test_csv_path)
    # 自动识别Point列名（大小写/下划线兼容）
    pid_col_candidates = [c for c in df_test.columns if c.lower().replace('_','') in ['pointid','point_id','point','id']]
    if len(pid_col_candidates) == 0:
        raise ValueError("测试CSV中未找到 Point_id 列")
    pid_col = pid_col_candidates[0]

    # 构建映射字典，包含多种key形式
    pid_to_coord = {}
    for _, row in df_test.iterrows():
        pid_raw = row[pid_col]
        lat = row['Latitude'] if 'Latitude' in df_test.columns else row['latitude']
        lon = row['Longitude'] if 'Longitude' in df_test.columns else row['longitude']
        keys = set()
        try:
            keys.add(str(pid_raw).strip())
            keys.add(str(int(float(pid_raw))))
        except:
            pass
        for k in keys:
            pid_to_coord[k] = (lat, lon)

    for point_id in all_point_ids:
        key1 = str(point_id).strip()
        key2 = None
        try:
            key2 = str(int(float(point_id)))
        except:
            pass
        if key1 in pid_to_coord:
            lat, lon = pid_to_coord[key1]
        elif key2 and key2 in pid_to_coord:
            lat, lon = pid_to_coord[key2]
        else:
            print(f"Warning: Point_id {point_id} coordinates not found")
            lat, lon = np.nan, np.nan
        all_latitudes.append(lat)
        all_longitudes.append(lon)
    
    print(f"Prediction completed, {len(all_predictions)} samples")
    print(f"Skipped batches: {skipped_batches}, filtered samples due to NaN/Inf: {skipped_samples_due_to_nan}")
    
    return np.array(all_predictions), np.array(all_targets), all_point_ids, np.array(all_latitudes), np.array(all_longitudes)

def analyze_regional_performance(predictions, targets, latitudes, longitudes):
    """分析区域性能"""
    print(f"\n=== Regional Performance Analysis ===")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'prediction': predictions,
        'target': targets,
        'latitude': latitudes,
        'longitude': longitudes,
        'error': predictions - targets,
        'abs_error': np.abs(predictions - targets)
    })
    
    # 分配区域
    results_df['region'] = results_df.apply(
        lambda row: assign_china_geographic_region(row['latitude'], row['longitude']), 
        axis=1
    )
    
    # 移除无效数据
    results_df = results_df.dropna(subset=['latitude', 'longitude'])
    
    print(f"Valid samples count: {len(results_df)}")
    
    # 按区域分析性能
    regional_results = {}
    
    for region in results_df['region'].unique():
        region_data = results_df[results_df['region'] == region]
        
        if len(region_data) < 3:  # 样本太少跳过
            continue
        
        # 检查是否有NaN值
        valid_mask = ~(np.isnan(region_data['target']) | np.isnan(region_data['prediction']))
        if valid_mask.sum() < 3:  # 有效样本太少跳过
            continue
            
        valid_targets = region_data['target'][valid_mask]
        valid_predictions = region_data['prediction'][valid_mask]
        
        # 计算性能指标
        rmse = np.sqrt(mean_squared_error(valid_targets, valid_predictions))
        r2 = r2_score(valid_targets, valid_predictions)
        mae = mean_absolute_error(valid_targets, valid_predictions)
        
        # 计算相对误差
        valid_region_data = region_data[valid_mask]
        rel_error = valid_region_data['abs_error'] / (valid_region_data['target'] + 1e-8)
        mean_rel_error = rel_error.mean()
        
        regional_results[region] = {
            'n_samples': len(region_data),
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'mean_rel_error': mean_rel_error,
            'lat_mean': region_data['latitude'].mean(),
            'lon_mean': region_data['longitude'].mean(),
            'soc_mean': region_data['target'].mean(),
            'soc_std': region_data['target'].std()
        }
    
    return results_df, regional_results

def print_regional_summary(regional_results):
    """打印区域性能总结"""
    print(f"\n{'='*80}")
    print("Regional Performance Summary (Real Model Validation)")
    print(f"{'='*80}")
    
    # 按R²排序
    sorted_regions = sorted(regional_results.items(), key=lambda x: x[1]['r2'], reverse=True)
    
    print(f"{'Region':<12} {'Samples':<8} {'RMSE':<8} {'R2':<8} {'MAE':<8} {'Rel_Error':<8} {'SOC_Mean':<8}")
    print("-"*80)
    
    for region, results in sorted_regions:
        print(f"{region:<12} {results['n_samples']:<8} "
              f"{results['rmse']:<8.3f} {results['r2']:<8.3f} {results['mae']:<8.3f} "
              f"{results['mean_rel_error']:<8.3f} {results['soc_mean']:<8.3f}")
    
    # 找出最佳和最差区域
    if len(sorted_regions) > 0:
        best_region = sorted_regions[0]
        worst_region = sorted_regions[-1]
        
        print(f"\nBest prediction region: {best_region[0]} (R2 = {best_region[1]['r2']:.3f})")
        print(f"Worst prediction region: {worst_region[0]} (R2 = {worst_region[1]['r2']:.3f})")
        
        # 分析区域差异
        r2_values = [r[1]['r2'] for r in sorted_regions]
        r2_std = np.std(r2_values)
        print(f"Regional R2 std: {r2_std:.3f}")
        
        if r2_std > 0.1:
            print("Large regional performance differences, model may have regional bias")
        else:
            print("Regional performance is relatively balanced")

def create_regional_visualizations(results_df, regional_results, save_dir='regional_analysis'):
    """创建区域可视化图表"""
    print(f"\n=== Creating Regional Visualizations ===")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 区域性能热力图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 准备热力图数据
    regions = list(regional_results.keys())
    metrics = ['rmse', 'r2', 'mae', 'mean_rel_error']
    metric_names = ['RMSE', 'R2', 'MAE', '相对误差']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i//2, i%2]
        
        values = [regional_results[region][metric] for region in regions]
        
        # 创建热力图
        im = ax.imshow(np.array(values).reshape(1, -1), cmap='RdYlBu_r', aspect='auto')
        
        # 设置标签
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(regions, rotation=45)
        ax.set_yticks([0])
        ax.set_yticklabels([metric_name])
        
        # 添加数值标注
        for j, value in enumerate(values):
            ax.text(j, 0, f'{value:.3f}', ha='center', va='center', fontweight='bold')
        
        ax.set_title(f'各区域{metric_name}性能')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/regional_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 地理分布散点图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 绝对误差分布
    scatter1 = axes[0].scatter(results_df['longitude'], results_df['latitude'], 
                              c=results_df['abs_error'], cmap='Reds', alpha=0.6, s=20)
    axes[0].set_xlabel('经度 (°E)')
    axes[0].set_ylabel('纬度 (°N)')
    axes[0].set_title('预测绝对误差地理分布')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='绝对误差')
    
    # 相对误差分布
    rel_error = results_df['abs_error'] / (results_df['target'] + 1e-8)
    scatter2 = axes[1].scatter(results_df['longitude'], results_df['latitude'], 
                              c=rel_error, cmap='Blues', alpha=0.6, s=20)
    axes[1].set_xlabel('经度 (°E)')
    axes[1].set_ylabel('纬度 (°N)')
    axes[1].set_title('预测相对误差地理分布')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='相对误差')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/error_geographic_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization charts saved to: {save_dir}/")

def save_detailed_results(results_df, regional_results, save_path='regional_performance_results.csv'):
    """保存详细结果"""
    print(f"\n=== Saving Detailed Results ===")
    
    # 保存区域统计结果
    regional_df = pd.DataFrame(regional_results).T
    regional_df.to_csv(save_path)
    
    # 保存样本级结果
    sample_results_path = save_path.replace('.csv', '_samples.csv')
    results_df.to_csv(sample_results_path, index=False)
    
    print(f"Regional statistics saved to: {save_path}")
    print(f"Sample-level results saved to: {sample_results_path}")

def main():
    """主函数"""
    print("=== Best Model Regional Performance Validation (Compatible Version) ===")
    
    try:
        # 1. Load test data (and detect static dims)
        test_loader, test_dataset, static_dim, lulc_classes, climate_seq_len, climate_feat_dim = load_test_data()
        
        # 2. Try model files after knowing static_dim
        model_candidates = [
            "bestmodel/RUN_11.19_climate_new3.pth.tar"
        ]
        
        model = None
        model_path = None
        for candidate in model_candidates:
            if os.path.exists(candidate):
                try:
                    print(f"Trying to load model: {candidate}")
                    # 兼容创建支持LULC嵌入的模型
                    model = load_best_model_compatible(
                        candidate,
                        static_feature_dim=static_dim,
                        lulc_num_classes=lulc_classes,
                        seq_len=climate_seq_len,
                        lstm_n_features_override=climate_feat_dim
                    )
                    model_path = candidate
                    print(f"Successfully loaded model: {candidate}")
                    break
                except Exception as e:
                    print(f"Failed to load: {e}")
                    continue
        
        if model is None:
            print("All model files failed to load")
            return
        
        # 3. Model prediction
        predictions, targets, point_ids, latitudes, longitudes = predict_with_model(model, test_loader)
        
        # 4. Regional performance analysis
        results_df, regional_results = analyze_regional_performance(predictions, targets, latitudes, longitudes)
        
        # 5. Print summary
        print_regional_summary(regional_results)
        
        # 6. Create visualizations
        create_regional_visualizations(results_df, regional_results)
        
        # 7. Save results
        save_detailed_results(results_df, regional_results)
        
        print(f"\nRegional performance validation completed!")
        print(f"Please check the generated charts and CSV files for detailed results.")
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
