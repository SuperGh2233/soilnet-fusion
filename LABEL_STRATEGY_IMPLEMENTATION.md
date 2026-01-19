# 标签策略开关实现方案

## A) 修改点清单

### 1. dataset/dataset_loader.py
- **函数**: `myNormalize.__init__` 和 `myNormalize.__call__`
- **改动**: 
  - 新增参数 `normalize_oc: bool = True`, `clip_oc: bool = True`
  - 在 `__call__` 中根据参数决定是否归一化和clip标签

### 2. train.py
- **改动**:
  - 新增命令行参数：`--label_mode`, `--huber_beta`, `--tail_threshold`, `--tail_weight`
  - 在构建 `myNormalize` 时根据 `label_mode` 设置 `normalize_oc` 和 `clip_oc`
  - 将参数传递给训练函数

### 3. train_utils.py
- **函数**: `train_step`, `train`, `test_step_w_id`
- **改动**:
  - `train_step` 新增标签变换和加权损失逻辑
  - `train` 函数新增参数并传递给 `train_step`
  - `test_step_w_id` 输出原尺度预测值（y_real_raw, y_pred_raw）

### 4. train.py (后处理部分)
- **改动**:
  - 删除或条件化 `y * OC_MAX` 的反归一化逻辑
  - 直接使用 CSV 中的 `y_real_raw` 和 `y_pred_raw` 计算指标

## B) 关键代码补丁

见下方各文件的完整补丁代码

## C) 示例命令行

```bash
# (1) baseline_raw_mse: 原尺度 SOC + MSE
python train.py -e baseline_raw_mse --label_mode baseline_raw_mse -ne 100 -seed 1 2 3

# (2) log1p_mse: log1p(SOC) + MSE
python train.py -e log1p_mse --label_mode log1p_mse -ne 100 -seed 1 2 3

# (3) log1p_huber: log1p(SOC) + Huber
python train.py -e log1p_huber --label_mode log1p_huber --huber_beta 1.0 -ne 100 -seed 1 2 3

# (4) log1p_huber_w: log1p(SOC) + Huber + 高值加权
python train.py -e log1p_huber_w --label_mode log1p_huber_w --huber_beta 1.0 --tail_threshold 30.0 --tail_weight 2.0 -ne 100 -seed 1 2 3
```

## D) 结果区分方式

### JSON 文件中
每个实验的 JSON 文件会包含以下字段用于区分：
- `LABEL_MODE`: 标签策略模式（baseline_raw_mse / log1p_mse / log1p_huber / log1p_huber_w）
- `HUBER_BETA`: Huber损失的beta参数
- `TAIL_THRESHOLD`: 高值样本阈值
- `TAIL_WEIGHT`: 高值样本权重

### CSV 文件中
测试结果 CSV 包含以下列：
- `point_id`: 样本ID
- `y_real_raw`: 真实值（原尺度 SOC）
- `y_pred_raw`: 预测值（原尺度 SOC）
- `y_real_log`: 真实值（log空间，仅log1p_*模式）
- `y_pred_log`: 预测值（log空间，仅log1p_*模式）

### 文件命名
- 模型文件: `results/RUN_{EXP_NAME}_{run_name}_best.pth.tar`
- CSV文件: `results/RUN_{EXP_NAME}_{run_name}_best.csv`
- JSON文件: `results/RUN_{EXP_NAME}_{run_name}.json`

其中 `EXP_NAME` 应设置为不同的值（如 baseline_raw_mse, log1p_mse 等）以区分实验。

