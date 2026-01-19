import os
import shutil
import random

# ========== 配置参数 ==========
# 原始影像文件夹
src_folder = 'dataset/l8_images_CN'
# 划分后保存的文件夹
dst_root = 'dataset/l8_images_CN_split'
# 划分比例
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# ========== 创建目标文件夹 ==========
train_folder = os.path.join(dst_root, 'train')
val_folder = os.path.join(dst_root, 'val')
test_folder = os.path.join(dst_root, 'test')
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# ========== 获取所有影像文件 ==========
all_files = [f for f in os.listdir(src_folder) if f.endswith('.tif')]
random.shuffle(all_files)

total = len(all_files)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

train_files = all_files[:train_end]
val_files = all_files[train_end:val_end]
test_files = all_files[val_end:]

# ========== 复制文件 ==========
def copy_files(file_list, dst_folder):
    for f in file_list:
        src_path = os.path.join(src_folder, f)
        dst_path = os.path.join(dst_folder, f)
        shutil.copy2(src_path, dst_path)

print(f"总文件数: {total}")
print(f"训练集: {len(train_files)}，验证集: {len(val_files)}，测试集: {len(test_files)}")

copy_files(train_files, train_folder)
copy_files(val_files, val_folder)
copy_files(test_files, test_folder)

print("数据集划分完成！")
print(f"训练集路径: {train_folder}")
print(f"验证集路径: {val_folder}")
print(f"测试集路径: {test_folder}") 