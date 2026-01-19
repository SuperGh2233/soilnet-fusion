"""
测试气候数据加载
"""
import sys
sys.path.append('.')
from dataset.dataset_loader_china import ChinaSNDatasetClimate
import config
import torch

print('Testing climate data loading...')
try:
    dataset = ChinaSNDatasetClimate(
        l8_dir=config.test_l8_folder_path,
        csv_dir=config.test_csv_path,
        climate_csv_folder=config.climate_csv_folder_path,
        l8_bands=config.bands,
        transform=None,
        dates=None,
        climate_dtype=torch.float32,
        normalize_climate=False,
        return_point_id=True
    )
    
    print(f'Dataset size: {len(dataset)}')
    sample = dataset[0]
    (l8_img, clim_arr), socd, point_id = sample
    
    print(f'Point ID: {point_id}')
    print(f'SOC: {socd}')
    print(f'L8 image shape: {l8_img.shape}')
    print(f'Climate data shape: {clim_arr.shape}')
    print(f'Climate data range: [{clim_arr.min():.3f}, {clim_arr.max():.3f}]')
    print(f'Climate data non-zero count: {(clim_arr != 0).sum()}')
    print('Climate data loading successful!')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()


