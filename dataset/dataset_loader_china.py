import torch
import numpy as np
from torch.utils.data import Dataset
from skimage import io
import os
from torchvision import datasets, transforms
import pandas as pd
import torch.nn.functional as F

# 直接定义需要的函数，避免导入问题
def reshape_tensor(tensor):
    """Takes in a pytorch tensor and reshapes it to (C,H,W) if it is not already in that shape."""
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 3 and tensor.shape[2] < tensor.shape[0]:
        tensor = tensor.permute((2,0,1))
    elif tensor.dim() == 3 and tensor.shape[2] > tensor.shape[0]:
        pass
    else:
        raise ValueError(f"Input tensor shape is unvalid: {tensor.shape}")
    return tensor

def reshape_array(array):
    """Takes in an array and reshapes it to (C,H,W) if it is not already in that shape."""
    if array.ndim == 2:
        array = np.expand_dims(array, axis=0)
    elif array.ndim == 3 and array.shape[2] < array.shape[0]:
        array = np.swapaxes(array, 0, 2)
        array = np.swapaxes(array, 1, 2)
    elif array.ndim == 3 and array.shape[2] > array.shape[0]:
        pass
    else:
        raise ValueError(f"Input array shape is invalid: {array.shape}")
    return array

def get_df_max_min(df, col):
    """Takes in a pandas dataframe and a column name and returns a tuple of the maximum and minimum values."""
    if isinstance(df, pd.DataFrame) and col in df.columns:
        max_val = float(df[col].max())
        min_val = float(df[col].min())
        return (min_val, max_val)
    else:
        raise ValueError("Invalid input. Please provide a pandas dataframe and a valid column name.")

def normalize(value, min_val, max_val):
    """Takes in a value, min and max of the data| returns the normalized value between 0 and 1."""
    return (value - min_val) / (max_val - min_val)

def log_transform(value):
    """Applies log transformation to the given value."""
    return np.log(value)

# Import the same transformation classes from the original file
try:
    from dataset.dataset_loader import myNormalize, myToTensor, Augmentations, NormalizeClimDF
except ImportError:
    from .dataset_loader import myNormalize, myToTensor, Augmentations, NormalizeClimDF

class ChinaSNDataset(Dataset):
  def __init__(self, l8_dir, csv_dir , l8_bands: list = None ,transform = None, return_point_id = False):
    # Declaring them becuase we nee them in __getitem__ function
    self.l8_dir = l8_dir
    self.csv_dir = csv_dir
    # List of the names in each path
    self.l8_names = [f for f in os.listdir(l8_dir) if f.endswith('.tif')] # reading only 
    self.l8_names.sort()
    # Declaring the l8 bands we want to use, if None all the bands will be used
    self.l8_bands = l8_bands if l8_bands else None
    # Declaring the transform function
    self.transform = transform
    # Reading the csv file in __init__ function to avoid reading it in every __getitem__ call
    self.df = pd.read_csv(self.csv_dir)
    self.return_point_id = return_point_id
  def __len__(self):
    return len(self.l8_names)
  def __getitem__(self, index):
    l8_img_name = self.l8_names[index] 
    l8_img_path = os.path.join(self.l8_dir,l8_img_name)

    point_id = l8_img_name.split('_')[0]

    # Robust Point_id matching
    df_pid_str = self.df['Point_id'].astype(str).str.replace('.0','', regex=False).str.strip()
    pid_str = str(point_id).replace('.0','').strip()
    row = self.df[df_pid_str == pid_str]
    if row.empty:
        # Try int match
        try:
            pid_int = int(float(point_id))
            row = self.df[self.df['Point_id'] == pid_int]
        except Exception:
            pass
    if row.empty:
        raise IndexError(f"No SOC row found for Point_id '{point_id}' from file '{l8_img_name}'. Check CSV '{self.csv_dir}' Point_id format.")
    socd = row['SOC'].values[0]
    

    l8_img = io.imread(l8_img_path)
    if self.l8_bands: l8_img = l8_img[self.l8_bands,:,:]

    if self.transform:
        l8_img,socd  = self.transform((l8_img,socd))
    
    # Returning the point_id as well if return_point_id is True | Used in the test phase
    if self.return_point_id:
        return l8_img,socd,point_id
    else:
        return l8_img,socd

class ChinaSNDatasetClimate(Dataset):
  def __init__(self, l8_dir, csv_dir , climate_csv_folder,
               l8_bands: list = None ,transform = None,
               dates = None,  # 使用 None，让代码自动从 CSV 文件读取日期
               climate_dtype = torch.float32, normalize_climate = True, return_point_id = False):
    
    # Declaring them becuase we nee them in __getitem__ function
    self.l8_dir = l8_dir
    self.csv_dir = csv_dir
    # List of the names in each path
    self.l8_names = [f for f in os.listdir(l8_dir) if f.endswith('.tif')] # reading only 
    self.l8_names.sort()
    # Declaring the l8 bands we want to use, if None all the bands will be used
    self.l8_bands = l8_bands if l8_bands else None
    # Declaring the transform function
    self.transform = transform
    # Reading the csv file in __init__ function to avoid reading it in every __getitem__ call
    self.df = pd.read_csv(self.csv_dir)
    
    # Reading Climate csv files
    # List all files in the directory and filter for .csv files
    csv_files = [f for f in os.listdir(climate_csv_folder) if os.path.isfile(os.path.join(climate_csv_folder, f)) and f.endswith('.csv')]
    self.clim_dfs =  [pd.read_csv(os.path.join(climate_csv_folder, f)) for f in csv_files]
    
    # 自动从第一个气候文件读取日期列
    if dates is None and len(self.clim_dfs) > 0:
        # 获取第一行作为日期列（跳过 Point_id 等非日期列）
        first_row = self.clim_dfs[0].iloc[0]
        # 过滤出日期格式的列（8位数字）
        self.dates = [col for col in first_row.index if str(col).isdigit() and len(str(col)) == 8]
    else:
        self.dates = dates or []
    
    if normalize_climate and self.dates:
        norm_clim = NormalizeClimDF(dates=self.dates)
        self.clim_dfs = [norm_clim(clim_df) for clim_df in self.clim_dfs]
    
    self.clim_dtype = climate_dtype
    
    self.return_point_id = return_point_id
    
  def __len__(self):
    return len(self.l8_names)
  
  def __getitem__(self, index):
    l8_img_name = self.l8_names[index] 
    l8_img_path = os.path.join(self.l8_dir,l8_img_name)

    point_id = l8_img_name.split('_')[0]

    # Robust Point_id matching for main CSV
    df_pid_str = self.df['Point_id'].astype(str).str.replace('.0','', regex=False).str.strip()
    pid_str = str(point_id).replace('.0','').strip()
    row = self.df[df_pid_str == pid_str]
    if row.empty:
        try:
            pid_int = int(float(point_id))
            row = self.df[self.df['Point_id'] == pid_int]
        except Exception:
            pass
    if row.empty:
        raise IndexError(f"No SOC row found for Point_id '{point_id}' from file '{l8_img_name}'. Check CSV '{self.csv_dir}' Point_id format.")
    socd = row['SOC'].values[0]
    
    self.clim_dfs_row = []
    for df in self.clim_dfs:
        # 尝试不同的Point_id列名
        point_id_col = None
        for col in df.columns:
            if col.lower() in ['point_id', 'pointid', 'pid']:
                point_id_col = col
                break
        
        if point_id_col is None:
            print(f"Warning: No Point_id column found in climate data")
            self.clim_dfs_row.append(np.full(len(self.dates), -999, dtype=np.float32))
            continue
            
        df_pid_str = df[point_id_col].astype(str).str.replace('.0','', regex=False).str.strip()
        row = df[df_pid_str == pid_str]
        if row.empty:
            try:
                pid_int = int(float(point_id))
                row = df[df[point_id_col] == pid_int]
            except Exception:
                pass
        if not row.empty:
            self.clim_dfs_row.append(row[self.dates].values.squeeze())
        else:
            self.clim_dfs_row.append(np.full(len(self.dates), -999, dtype=np.float32))
    clim_arr = np.stack(self.clim_dfs_row, axis=1)

    l8_img = io.imread(l8_img_path)
    if self.l8_bands: l8_img = l8_img[self.l8_bands,:,:]

    if self.transform:
        l8_img,socd  = self.transform((l8_img,socd))
        clim_arr = torch.tensor(clim_arr).to(dtype=self.clim_dtype)
      
    if self.return_point_id:
        return (l8_img,clim_arr),socd, point_id
    else:
        return (l8_img,clim_arr),socd 