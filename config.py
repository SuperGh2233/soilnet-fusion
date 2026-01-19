import os
# 影像数据路径 - 中国数据集3500
train_l8_folder_path = 'dataset/l8_images_CN_split/train'
test_l8_folder_path = 'dataset/l8_images_CN_split/test'
val_l8_folder_path = 'dataset/l8_images_CN_split/val'

# 中国数据集 CSV 路径
lucas_csv_path = 'dataset/CN-SOC-3500_new.csv'
test_csv_path = 'dataset/CN-SOC-3500_new.csv'  # 测试数据使用相同的CSV文件

# 气候数据路径 - 使用2011-2015年处理后的数据
climate_csv_folder_path = "dataset/Climate/climate_exports_indexed_2011_2015/output_filled_norm"
# 自动统计气候变量个数
climate_variable_count = len([f for f in os.listdir(climate_csv_folder_path) if f.endswith('.csv')])

# 静态特征 CSV 路径（可与主CSV相同，静态分支仅读取新增静态列）
static_csv_path = lucas_csv_path

# 影像波段配置 (使用整数索引，对应波段顺序)
bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # 对应 B2, B3, B4, B5, B6, B7, B8, B10, B11, NDVI, NDWI, SRTM, Slope (14个波段)

# SimCLR 模型路径（如果有预训练模型的话）
SIMCLR_PATH = "pre_model/RUN_LUCAS_SimCLR_bs64_D_2025_09_09_T_19_57_SelfSupervised.pth"

# ViT/ViT-CoMer 预训练权重路径（DeiT-Base，embed_dim=768）
VIT_PRETRAINED_PATH = "pre_model/deit_base_patch16_224-b5f2ef4d.pth"






# # 影像数据路径 - 中国数据集1300
# train_l8_folder_path = 'dataset/I8_images_CN1300/train'
# test_l8_folder_path = 'dataset/I8_images_CN1300/test'
# val_l8_folder_path = 'dataset/I8_images_CN1300/val'

# # 中国数据集 CSV 路径
# lucas_csv_path = 'dataset/China-SOCD-Dataset-20250415_filter0-30_resorted.CSV'

# # 气候数据路径 - 中国数据集
# climate_csv_folder_path = "dataset/Climate/CN/CN-1300/"

# # SimCLR 模型路径（如果有预训练模型的话）
# SIMCLR_PATH = "results/RUN_LUCAS_Self560_ViT_Trans_D_2024_08_19_T_16_13_SelfSupervised.pth"
