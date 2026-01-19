import os
import sys
script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
script_dir = os.path.dirname(script_path) #i.e. /path/to/dir/
sys.path.append(script_dir) # add submodules to path so that CNNFlattener can import ChannelAttention from within CNNFlattener

# 导出主要函数
from .utils import reshape_tensor, reshape_array, get_df_max_min, normalize, log_transform, TextColors