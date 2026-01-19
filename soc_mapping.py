"""
SOC制图脚本
使用训练好的模型对栅格数据进行批量预测，生成SOC分布图
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from soilnet.soil_net import SoilNetLSTM
from soilnet.soil_net_static import SoilNetLSTMWithStatic
from dataset.dataset_loader_china import ChinaSNDatasetClimate
from dataset.dataset_loader_china_static import ChinaSNDatasetClimateStatic

def load_model_for_mapping(checkpoint_path, device, use_static=False, 
                          cnn_arch="ViT-CoMer", rnn_arch="Transformer",
                          cnn_in_channels=14, regresor_input_from_cnn=768,
                          lstm_n_features=17, lstm_out=128, hidden_size=128,
                          seq_len=60, img_size=64, reg_version=1,
                          static_dim=0, lulc_classes=0):
    """
    加载训练好的模型用于制图
    
    Args:
        checkpoint_path: 模型检查点路径
        device: 计算设备
        use_static: 是否使用静态特征
        ... (其他模型参数)
    
    Returns:
        加载的模型
    """
    # 根据是否使用静态特征选择模型类
    if use_static and static_dim > 0:
        model = SoilNetLSTMWithStatic(
            cnn_arch=cnn_arch,
            rnn_arch=rnn_arch,
            cnn_in_channels=cnn_in_channels,
            regresor_input_from_cnn=regresor_input_from_cnn,
            lstm_n_features=lstm_n_features,
            lstm_out=lstm_out,
            hidden_size=hidden_size,
            seq_len=seq_len,
            img_size=img_size,
            reg_version=reg_version,
            static_dim=static_dim,
            lulc_classes=lulc_classes
        )
    else:
        model = SoilNetLSTM(
            cnn_arch=cnn_arch,
            rnn_arch=rnn_arch,
            cnn_in_channels=cnn_in_channels,
            regresor_input_from_cnn=regresor_input_from_cnn,
            lstm_n_features=lstm_n_features,
            lstm_out=lstm_out,
            hidden_size=hidden_size,
            seq_len=seq_len,
            img_size=img_size,
            reg_version=reg_version
        )
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理不同的检查点格式
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # 加载权重（允许部分匹配）
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print(f"模型已加载: {checkpoint_path}")
    return model


def extract_patch_from_raster(raster_path, center_x, center_y, patch_size=64, bands=None):
    """
    从栅格中提取指定位置的图像块
    
    Args:
        raster_path: 栅格文件路径
        center_x, center_y: 中心点坐标（像素坐标或地理坐标）
        patch_size: 图像块大小
        bands: 要提取的波段索引列表（None表示所有波段）
    
    Returns:
        图像块数组 (channels, height, width)
    """
    with rasterio.open(raster_path) as src:
        # 如果输入是地理坐标，转换为像素坐标
        if isinstance(center_x, (int, float)) and center_x > 1000:  # 假设是地理坐标
            row, col = src.index(center_x, center_y)
        else:
            row, col = int(center_y), int(center_x)
        
        half_size = patch_size // 2
        
        # 计算窗口
        window = rasterio.windows.Window(
            col_off=max(0, col - half_size),
            row_off=max(0, row - half_size),
            width=patch_size,
            height=patch_size
        )
        
        # 读取数据
        if bands is None:
            data = src.read(window=window)
        else:
            data = src.read(bands, window=window)
        
        # 如果窗口超出边界，进行填充
        if data.shape[1] < patch_size or data.shape[2] < patch_size:
            padded = np.zeros((data.shape[0], patch_size, patch_size), dtype=data.dtype)
            h, w = data.shape[1], data.shape[2]
            padded[:, :h, :w] = data
            data = padded
        
        return data


def get_climate_data_for_point(point_id, climate_csv_folder, dates=None):
    """
    获取指定点的气候时序数据
    
    Args:
        point_id: 点ID
        climate_csv_folder: 气候数据文件夹路径
        dates: 日期列表（如果为None，从CSV自动读取）
    
    Returns:
        气候数据数组 (seq_len, n_features)
    """
    # 读取所有气候变量CSV
    csv_files = [f for f in os.listdir(climate_csv_folder) 
                 if f.endswith('.csv')]
    
    climate_data = []
    for csv_file in sorted(csv_files):
        csv_path = os.path.join(climate_csv_folder, csv_file)
        df = pd.read_csv(csv_path)
        
        # 查找对应Point_ID的行
        row = df[df['Point_ID'] == int(point_id)]
        
        if len(row) == 0:
            # 如果找不到，返回NaN填充的数组
            if dates:
                return np.full((len(dates), 1), np.nan)
            else:
                # 从第一个文件推断日期列
                date_cols = [col for col in df.columns 
                            if str(col).isdigit() and len(str(col)) == 8]
                return np.full((len(date_cols), 1), np.nan)
        
        # 提取日期列数据
        if dates:
            date_cols = dates
        else:
            date_cols = [col for col in df.columns 
                        if str(col).isdigit() and len(str(col)) == 8]
            date_cols = sorted(date_cols)
        
        values = row[date_cols].values.squeeze()
        climate_data.append(values)
    
    # 堆叠为 (seq_len, n_features)
    climate_array = np.stack(climate_data, axis=1)
    return climate_array


def predict_soc_for_raster(model, raster_path, climate_csv_folder, 
                          output_path, patch_size=64, batch_size=32,
                          device='cuda', use_static=False, static_csv_path=None,
                          transform=None, dates=None):
    """
    对整幅栅格进行SOC预测
    
    Args:
        model: 训练好的模型
        raster_path: 输入栅格路径（多波段）
        climate_csv_folder: 气候数据文件夹
        output_path: 输出SOC栅格路径
        patch_size: 图像块大小
        batch_size: 批处理大小
        device: 计算设备
        use_static: 是否使用静态特征
        static_csv_path: 静态特征CSV路径
        transform: 数据变换函数
        dates: 气候数据日期列表
    """
    print(f"开始SOC制图...")
    print(f"输入栅格: {raster_path}")
    print(f"输出栅格: {output_path}")
    
    # 打开输入栅格
    with rasterio.open(raster_path) as src:
        height, width = src.height, src.width
        crs = src.crs
        transform = src.transform
        
        print(f"栅格尺寸: {width} x {height}")
        print(f"空间参考: {crs}")
        
        # 创建输出数组
        soc_map = np.full((height, width), np.nan, dtype=np.float32)
        
        # 批量处理像素
        # 注意：这里需要根据实际情况调整
        # 如果栅格很大，可能需要：
        # 1. 使用滑动窗口
        # 2. 或者先提取样点位置进行预测
        
        print("注意：当前实现需要为每个像素提供Point_ID和对应的气候数据")
        print("如果要对整幅栅格制图，需要：")
        print("1. 准备每个像素的Point_ID映射")
        print("2. 准备每个像素的气候时序数据")
        print("3. 或者使用空间插值方法获取每个像素的气候数据")
    
    print("制图完成！")


def predict_soc_for_points(model, points_csv, raster_path, climate_csv_folder,
                          output_csv, patch_size=64, batch_size=32,
                          device='cuda', use_static=False, static_csv_path=None,
                          dates=None):
    """
    对CSV中指定的点进行SOC预测
    
    Args:
        model: 训练好的模型
        points_csv: 包含点坐标和Point_ID的CSV文件
        raster_path: 输入栅格路径
        climate_csv_folder: 气候数据文件夹
        output_csv: 输出预测结果CSV路径
        ... (其他参数)
    """
    print(f"开始对点进行SOC预测...")
    
    # 读取点数据
    points_df = pd.read_csv(points_csv)
    
    # 检查必需的列
    required_cols = ['Point_ID']
    if not all(col in points_df.columns for col in required_cols):
        raise ValueError(f"CSV文件必须包含列: {required_cols}")
    
    predictions = []
    point_ids = []
    
    model.eval()
    with torch.no_grad():
        for idx, row in points_df.iterrows():
            point_id = int(row['Point_ID'])
            
            # 获取坐标（如果有）
            if 'Longitude' in row and 'Latitude' in row:
                lon, lat = row['Longitude'], row['Latitude']
            else:
                # 如果没有坐标，需要从其他数据源获取
                lon, lat = None, None
            
            # 提取图像块
            # 注意：需要根据实际情况调整坐标转换
            if lon is not None and lat is not None:
                patch = extract_patch_from_raster(
                    raster_path, lon, lat, patch_size=patch_size
                )
            else:
                # 如果无法获取坐标，跳过
                print(f"警告: Point_ID {point_id} 缺少坐标信息，跳过")
                continue
            
            # 获取气候数据
            climate_data = get_climate_data_for_point(
                point_id, climate_csv_folder, dates=dates
            )
            
            # 数据预处理
            if transform:
                patch, _ = transform((patch, 0))
            
            # 转换为张量
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).to(device)
            climate_tensor = torch.from_numpy(climate_data).float().unsqueeze(0).to(device)
            
            # 预测
            if use_static and static_csv_path:
                # 需要加载静态特征
                # 这里简化处理，实际需要从CSV读取
                static_numeric = torch.zeros(1, 0).to(device)  # 占位
                lulc_idx = torch.zeros(1, dtype=torch.long).to(device)  # 占位
                output = model((patch_tensor, climate_tensor, static_numeric, lulc_idx))
            else:
                output = model((patch_tensor, climate_tensor))
            
            pred_soc = output.cpu().item()
            predictions.append(pred_soc)
            point_ids.append(point_id)
            
            if (idx + 1) % 100 == 0:
                print(f"已处理 {idx + 1}/{len(points_df)} 个点")
    
    # 保存结果
    result_df = pd.DataFrame({
        'Point_ID': point_ids,
        'Predicted_SOC': predictions
    })
    result_df.to_csv(output_csv, index=False)
    print(f"预测结果已保存: {output_csv}")
    print(f"共预测 {len(predictions)} 个点")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SOC制图脚本')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--raster', type=str, required=True,
                       help='输入栅格路径')
    parser.add_argument('--points_csv', type=str, default=None,
                       help='点数据CSV（如果只预测特定点）')
    parser.add_argument('--output', type=str, required=True,
                       help='输出路径（CSV或栅格）')
    parser.add_argument('--climate_folder', type=str,
                       default=config.climate_csv_folder_path,
                       help='气候数据文件夹路径')
    parser.add_argument('--static_csv', type=str, default=None,
                       help='静态特征CSV路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批处理大小')
    parser.add_argument('--patch_size', type=int, default=64,
                       help='图像块大小')
    
    # 模型参数
    parser.add_argument('--cnn_arch', type=str, default='ViT-CoMer',
                       help='CNN架构')
    parser.add_argument('--rnn_arch', type=str, default='Transformer',
                       help='RNN架构')
    parser.add_argument('--cnn_in_channels', type=int, default=14,
                       help='输入波段数')
    parser.add_argument('--lstm_n_features', type=int, default=17,
                       help='气候变量数量')
    parser.add_argument('--seq_len', type=int, default=60,
                       help='时序长度')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    # 注意：需要根据实际检查点调整参数
    model = load_model_for_mapping(
        args.checkpoint, device,
        use_static=(args.static_csv is not None),
        cnn_arch=args.cnn_arch,
        rnn_arch=args.rnn_arch,
        cnn_in_channels=args.cnn_in_channels,
        lstm_n_features=args.lstm_n_features,
        seq_len=args.seq_len
    )
    
    # 执行预测
    if args.points_csv:
        # 对特定点进行预测
        predict_soc_for_points(
            model, args.points_csv, args.raster, args.climate_folder,
            args.output, patch_size=args.patch_size, batch_size=args.batch_size,
            device=device, use_static=(args.static_csv is not None),
            static_csv_path=args.static_csv
        )
    else:
        # 对整幅栅格进行预测（需要更多实现）
        print("整幅栅格制图功能需要进一步实现")
        print("建议先使用 --points_csv 参数对特定点进行预测")







