# 静态特征开关使用指南

## 概述

已添加 `-static` 命令行开关，用于显式控制是否使用静态特征分支。

## 使用方法

### 启用静态特征

```bash
python train.py \
    -e experiment_name \
    -static \
    --static_csv dataset/CN-SOC-3500_new.csv \
    -lstm \
    -cnn ViT-CoMer \
    -rnn Transformer \
    -ne 60 \
    -seed 1
```

### 禁用静态特征（默认）

```bash
python train.py \
    -e experiment_name \
    -lstm \
    -cnn ViT-CoMer \
    -rnn Transformer \
    -ne 60 \
    -seed 1
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-static` / `--use_static` | flag | False | 启用静态特征分支 |
| `--static_csv` | str | None | 静态特征CSV文件路径 |

## 工作逻辑

1. **启用静态特征** (`-static`):
   - 必须同时提供 `--static_csv` 路径
   - 如果路径不存在，会报错
   - 如果模块不可用，会报错

2. **禁用静态特征** (默认):
   - 即使提供了 `--static_csv`，也不会使用静态特征
   - 使用标准的 `SNDatasetClimate` 数据集

## 完整示例

### 示例1：启用静态特征

```bash
python train.py \
    -e baseline_with_static \
    -d CHINA \
    -w 6 \
    -cnn ViT-CoMer \
    -rnn Transformer \
    -lstm \
    -trbs 32 \
    -ne 60 \
    -lr 0.0001 \
    -ls step \
    -stm \
    -srtm \
    -static \
    --static_csv dataset/CN-SOC-3500_new.csv \
    --label_mode baseline_raw_mse \
    -seed 1
```

### 示例2：禁用静态特征（对比实验）

```bash
python train.py \
    -e baseline_no_static \
    -d CHINA \
    -w 6 \
    -cnn ViT-CoMer \
    -rnn Transformer \
    -lstm \
    -trbs 32 \
    -ne 60 \
    -lr 0.0001 \
    -ls step \
    -stm \
    -srtm \
    --label_mode baseline_raw_mse \
    -seed 1
```

## 错误处理

### 错误1：启用了静态特征但未提供CSV路径

```
警告: 已启用静态特征 (-static)，但未提供 --static_csv 路径
错误: 无法找到静态特征CSV路径，请使用 --static_csv 指定
```

**解决方法**: 添加 `--static_csv` 参数

### 错误2：CSV文件不存在

```
错误: 静态特征CSV文件不存在: /path/to/file.csv
```

**解决方法**: 检查文件路径是否正确

### 错误3：静态特征模块不可用

```
错误: 静态特征数据集模块不可用 (dataset_loader_china_static)
```

**解决方法**: 确保 `dataset/dataset_loader_china_static.py` 文件存在

## 与标签策略的配合使用

静态特征开关可以与标签策略开关一起使用：

```bash
python train.py \
    -e log1p_mse_with_static \
    -static \
    --static_csv dataset/CN-SOC-3500_new.csv \
    --label_mode log1p_mse \
    -lstm \
    -cnn ViT-CoMer \
    -rnn Transformer \
    -ne 60 \
    -seed 1
```

## 结果记录

启用静态特征后，JSON结果文件中会包含：

```json
{
    "USE_STATIC_FEATURES": true,
    "STATIC_CSV": "dataset/CN-SOC-3500_new.csv",
    ...
}
```

## 注意事项

1. **必须与 `-lstm` 一起使用**: 静态特征只在气候数据分支中可用
2. **仅支持 CHINA 数据集**: 当前实现仅支持中国数据集
3. **CSV格式要求**: CSV必须包含静态特征列（如 bulk_density, cec, sand, silt, clay, twi, tpi, lulc 等）

## 对比实验建议

可以运行以下对比实验：

1. **无静态特征**: 不添加 `-static`
2. **有静态特征**: 添加 `-static --static_csv <path>`

对比两者的性能差异，评估静态特征的有效性。






























