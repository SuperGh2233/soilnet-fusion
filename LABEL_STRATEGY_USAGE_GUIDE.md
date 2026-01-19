# 标签策略开关使用指南

## 概述

本实现为 SoilNet 项目添加了标签策略开关功能，用于长尾 SOC 回归的消融实验。通过简单的命令行参数即可切换四种不同的训练策略。

## 四种实验模式

### 1. baseline_raw_mse
- **训练目标**: 原尺度 SOC
- **损失函数**: MSE
- **适用场景**: 基线实验，直接在原尺度上回归

### 2. log1p_mse
- **训练目标**: log1p(SOC)
- **损失函数**: MSE
- **适用场景**: 对数变换缓解长尾分布

### 3. log1p_huber
- **训练目标**: log1p(SOC)
- **损失函数**: Huber (SmoothL1Loss)
- **适用场景**: 对数变换 + 鲁棒损失函数

### 4. log1p_huber_w
- **训练目标**: log1p(SOC)
- **损失函数**: Huber + 高值加权
- **适用场景**: 对数变换 + 鲁棒损失 + 高值样本加权（SOC>30 权重2.0）

## 命令行示例

### 基础用法

```bash
# 实验1: baseline_raw_mse
python train.py \
    -e baseline_raw_mse \
    --label_mode baseline_raw_mse \
    -ne 100 \
    -seed 1 2 3 \
    -lstm \
    -cnn ViT-CoMer \
    -rnn Transformer

# 实验2: log1p_mse
python train.py \
    -e log1p_mse \
    --label_mode log1p_mse \
    -ne 100 \
    -seed 1 2 3 \
    -lstm \
    -cnn ViT-CoMer \
    -rnn Transformer

# 实验3: log1p_huber
python train.py \
    -e log1p_huber \
    --label_mode log1p_huber \
    --huber_beta 1.0 \
    -ne 100 \
    -seed 1 2 3 \
    -lstm \
    -cnn ViT-CoMer \
    -rnn Transformer

# 实验4: log1p_huber_w
python train.py \
    -e log1p_huber_w \
    --label_mode log1p_huber_w \
    --huber_beta 1.0 \
    --tail_threshold 30.0 \
    --tail_weight 2.0 \
    -ne 100 \
    -seed 1 2 3 \
    -lstm \
    -cnn ViT-CoMer \
    -rnn Transformer
```

### 高级参数调整

```bash
# 调整 Huber 损失的 beta 参数（控制对异常值的敏感度）
python train.py \
    -e log1p_huber_beta0.5 \
    --label_mode log1p_huber \
    --huber_beta 0.5 \
    -ne 100 -seed 1

# 调整高值样本的阈值和权重
python train.py \
    -e log1p_huber_w_custom \
    --label_mode log1p_huber_w \
    --huber_beta 1.0 \
    --tail_threshold 25.0 \
    --tail_weight 3.0 \
    -ne 100 -seed 1
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--label_mode` | str | None | 标签策略模式，可选值见上文 |
| `--huber_beta` | float | 1.0 | Huber损失的beta参数，值越小对异常值越敏感 |
| `--tail_threshold` | float | 30.0 | 高值样本阈值（g/kg），超过此值的样本会被加权 |
| `--tail_weight` | float | 2.0 | 高值样本的权重倍数 |

## 输出文件说明

### JSON 结果文件

文件名: `results/RUN_{EXP_NAME}_{run_name}.json`

关键字段：
```json
{
    "LABEL_MODE": "log1p_huber_w",
    "HUBER_BETA": 1.0,
    "TAIL_THRESHOLD": 30.0,
    "TAIL_WEIGHT": 2.0,
    "RMSE_MEAN": 2.45,
    "R2_MEAN": 0.85,
    "MAE_MEAN": 1.89,
    "best_dict": {
        "RMSE": 2.38,
        "R2": 0.87,
        "MAE": 1.82
    }
}
```

### CSV 预测结果文件

文件名: `results/RUN_{EXP_NAME}_{run_name}_best.csv`

#### 四种模式的CSV列：

**baseline_raw_mse 模式:**
```csv
point_id,y_real_raw,y_pred_raw,y_real,y_pred
123456,15.3,14.8,15.3,14.8
```

**log1p_* 模式:**
```csv
point_id,y_real_raw,y_pred_raw,y_real_log,y_pred_log,y_real,y_pred
123456,15.3,14.8,2.74,2.72,15.3,14.8
```

- `y_real_raw`: 真实值（原尺度 SOC，g/kg）
- `y_pred_raw`: 预测值（原尺度 SOC，g/kg）
- `y_real_log`: 真实值（log空间）
- `y_pred_log`: 预测值（log空间）
- `y_real`, `y_pred`: 兼容旧版的列名

**重要**: 所有指标（RMSE, MAE, R²等）都是在原尺度上计算的，即使用 `y_real_raw` 和 `y_pred_raw`。

## 验证实验正确性

### 1. 检查数据集标签范围

```python
import pandas as pd

# 读取训练数据CSV
df = pd.read_csv('dataset/CN-SOC-3500_new.csv')
print(f"OC 范围: {df['OC'].min():.2f} - {df['OC'].max():.2f}")
print(f"OC > 30 的样本数: {(df['OC'] > 30).sum()}")
print(f"OC > 30 的比例: {(df['OC'] > 30).mean()*100:.2f}%")
```

### 2. 检查预测结果CSV

```python
import pandas as pd
import numpy as np

# 读取预测结果
df = pd.read_csv('results/RUN_log1p_huber_w_D_2025_12_29_T_20_00_best.csv')

print(f"预测值范围: {df['y_pred_raw'].min():.2f} - {df['y_pred_raw'].max():.2f}")
print(f"真实值范围: {df['y_real_raw'].min():.2f} - {df['y_real_raw'].max():.2f}")

# 检查高值样本的预测效果
high_value_mask = df['y_real_raw'] > 30
print(f"\n高值样本 (SOC>30) 统计:")
print(f"  样本数: {high_value_mask.sum()}")
print(f"  RMSE: {np.sqrt(np.mean((df.loc[high_value_mask, 'y_pred_raw'] - df.loc[high_value_mask, 'y_real_raw'])**2)):.2f}")
print(f"  MAE: {np.mean(np.abs(df.loc[high_value_mask, 'y_pred_raw'] - df.loc[high_value_mask, 'y_real_raw'])):.2f}")
```

### 3. 对比四组实验

```python
import json
import pandas as pd

experiments = ['baseline_raw_mse', 'log1p_mse', 'log1p_huber', 'log1p_huber_w']
results = []

for exp in experiments:
    # 假设文件名格式为 RUN_{exp}_D_2025_12_29_T_XX_XX.json
    # 你需要根据实际文件名调整
    json_file = f'results/RUN_{exp}_D_2025_12_29_T_20_00.json'
    with open(json_file, 'r') as f:
        data = json.load(f)
    results.append({
        'Experiment': exp,
        'RMSE': data['best_dict']['RMSE'],
        'R2': data['best_dict']['R2'],
        'MAE': data['best_dict']['MAE']
    })

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
```

## 常见问题

### Q1: 为什么不能同时使用 `--label_mode` 和 `--log_loss`？

A: `--label_mode` 已经包含了对数变换的逻辑，不需要再使用 `--log_loss`。如果同时使用可能导致重复变换。

### Q2: 如何恢复旧版行为（归一化到[0,1]）？

A: 不传递 `--label_mode` 参数即可，此时会使用默认的归一化行为。

### Q3: 为什么 log1p_* 模式下训练loss很小？

A: 因为训练loss是在log空间计算的，log1p(30) ≈ 3.4，所以loss值会比原尺度小很多。但最终评估指标是在原尺度上计算的。

### Q4: 如何选择 huber_beta 参数？

A: 
- beta=1.0 (默认): 平衡鲁棒性和精度
- beta<1.0: 对异常值更鲁棒，但可能牺牲精度
- beta>1.0: 更接近MSE，精度更高但对异常值敏感

### Q5: 如何确定 tail_threshold 和 tail_weight？

A:
- `tail_threshold`: 根据数据分布选择，建议选择高分位数（如90%分位数）
- `tail_weight`: 建议从2.0开始，可以尝试1.5-3.0之间的值

## 实验建议

### 最小可行实验（快速验证）

```bash
# 使用较少的epoch和单个seed快速验证
python train.py -e test_baseline --label_mode baseline_raw_mse -ne 10 -seed 1 -lstm
python train.py -e test_log1p --label_mode log1p_mse -ne 10 -seed 1 -lstm
```

### 完整消融实验

```bash
# 使用完整配置和多个seed
for mode in baseline_raw_mse log1p_mse log1p_huber log1p_huber_w; do
    python train.py \
        -e ${mode}_full \
        --label_mode ${mode} \
        --huber_beta 1.0 \
        --tail_threshold 30.0 \
        --tail_weight 2.0 \
        -ne 100 \
        -seed 1 2 3 4 5 \
        -lstm \
        -cnn ViT-CoMer \
        -rnn Transformer \
        -lr 1e-4 \
        -wd 1e-5 \
        -trbs 32 \
        -tsbs 32
done
```

## 技术细节

### 标签变换流程

1. **数据集加载**: 
   - `myNormalize` 根据 `normalize_oc=False` 保持标签为原尺度
   - 不进行 clip 操作，保留所有高值样本

2. **训练阶段**:
   - `train_step` 根据 `label_mode` 对标签进行变换
   - 在变换后的空间计算损失
   - 反向传播更新模型

3. **评估阶段**:
   - `test_step_w_id` 将模型输出转换回原尺度
   - 在原尺度上计算所有评估指标
   - 保存原尺度和log空间的值到CSV

### 损失函数对比

| 损失函数 | 公式 | 特点 |
|---------|------|------|
| MSE | `(y_pred - y_true)²` | 对异常值敏感 |
| Huber | `0.5*x² if |x|<β else β*(|x|-0.5*β)` | 对异常值鲁棒 |
| Weighted Huber | `w * Huber(y_pred, y_true)` | 可调整样本权重 |

## 联系与支持

如有问题，请查看代码注释或联系项目维护者。

