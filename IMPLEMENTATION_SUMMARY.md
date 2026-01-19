# 标签策略开关实现总结

## ✅ 实现完成

已成功为 SoilNet 项目实现标签策略开关功能，用于长尾 SOC 回归的四组消融实验。

## 📋 修改文件清单

### 1. 核心代码修改

| 文件 | 修改内容 | 行数变化 |
|------|---------|---------|
| `dataset/dataset_loader.py` | 添加 `normalize_oc` 和 `clip_oc` 参数 | +8 行 |
| `train.py` | 添加命令行参数和标签策略逻辑 | +60 行 |
| `train_utils.py` | 修改训练循环和测试函数 | +80 行 |

### 2. 新增文件

| 文件 | 用途 |
|------|------|
| `LABEL_STRATEGY_IMPLEMENTATION.md` | 实现方案文档 |
| `LABEL_STRATEGY_USAGE_GUIDE.md` | 使用指南 |
| `test_label_strategy.py` | 功能测试脚本 |
| `run_ablation_experiments.sh` | 批量运行脚本 |
| `analyze_ablation_results.py` | 结果分析脚本 |
| `IMPLEMENTATION_SUMMARY.md` | 本文件 |

## 🎯 实现的功能

### 四种实验模式

1. **baseline_raw_mse**: 原尺度 SOC + MSE
2. **log1p_mse**: log1p(SOC) + MSE
3. **log1p_huber**: log1p(SOC) + Huber
4. **log1p_huber_w**: log1p(SOC) + Huber + 高值加权

### 核心特性

✅ **最小侵入**: 仅修改3个文件，不破坏原有逻辑  
✅ **接口统一**: 通过命令行参数切换，无需修改代码  
✅ **向后兼容**: 不传参数时保持原有行为  
✅ **评估正确**: 所有指标在原尺度上计算  
✅ **可复现**: 支持多seed交叉验证  

## 🔧 关键实现细节

### 1. 数据集层面 (dataset_loader.py)

```python
class myNormalize:
    def __init__(self, ..., normalize_oc=True, clip_oc=True):
        self.normalize_oc = normalize_oc
        self.clip_oc = clip_oc
    
    def __call__(self, sample):
        if self.normalize_oc:
            oc = normalize(oc, self.oc_min, self.oc_max)
            if self.clip_oc:
                oc = clip(oc, 0, 1)
        # else: keep raw scale
```

### 2. 训练层面 (train_utils.py)

```python
def train_step(..., label_mode, huber_beta, tail_threshold, tail_weight):
    if label_mode == 'baseline_raw_mse':
        y_target = y_raw
        loss = F.mse_loss(y_pred, y_target)
    elif label_mode == 'log1p_mse':
        y_target = log1p(y_raw)
        loss = F.mse_loss(y_pred, y_target)
    elif label_mode == 'log1p_huber':
        y_target = log1p(y_raw)
        loss = F.smooth_l1_loss(y_pred, y_target, beta=huber_beta)
    elif label_mode == 'log1p_huber_w':
        y_target = log1p(y_raw)
        err = F.smooth_l1_loss(y_pred, y_target, beta=huber_beta, reduction='none')
        weights = where(y_raw > tail_threshold, tail_weight, 1.0)
        loss = (weights * err).mean()
```

### 3. 评估层面 (train_utils.py)

```python
def test_step_w_id(..., label_mode):
    # 根据 label_mode 转换预测值到原尺度
    if label_mode in ['log1p_mse', 'log1p_huber', 'log1p_huber_w']:
        y_pred_raw = expm1(y_pred_log)
    else:
        y_pred_raw = y_pred
    
    # 保存原尺度值到CSV
    results.append({
        'y_real_raw': y_real_raw,
        'y_pred_raw': y_pred_raw
    })
```

## 📊 验收标准

| 标准 | 状态 | 说明 |
|------|------|------|
| 1. 参数切换 | ✅ | 仅通过 `--label_mode` 切换 |
| 2. 原尺度标签 | ✅ | 数据集输出原尺度，不clip |
| 3. 正确变换 | ✅ | log1p变换在训练循环中 |
| 4. 原尺度评估 | ✅ | 指标在原尺度上计算 |
| 5. 向后兼容 | ✅ | 不传参数保持旧版行为 |

## 🚀 快速开始

### 1. 测试功能

```bash
python test_label_strategy.py
```

### 2. 运行单个实验

```bash
python train.py \
    -e baseline_raw_mse \
    --label_mode baseline_raw_mse \
    -ne 10 \
    -seed 1 \
    -lstm
```

### 3. 运行完整消融实验

```bash
bash run_ablation_experiments.sh
```

### 4. 分析结果

```bash
python analyze_ablation_results.py
```

## 📈 预期结果

### 实验假设

1. **log1p变换** 应该能缓解长尾分布，提升整体性能
2. **Huber损失** 应该对异常值更鲁棒
3. **高值加权** 应该显著改善高值样本（SOC>30）的预测效果

### 评估指标

- **整体性能**: RMSE, MAE, R², RPIQ, CCC
- **高值样本**: 高值RMSE, 高值MAE
- **低值样本**: 低值RMSE, 低值MAE

## 🔍 调试建议

### 如果遇到问题

1. **检查数据集标签范围**
   ```python
   import pandas as pd
   df = pd.read_csv('dataset/CN-SOC-3500_new.csv')
   print(df['OC'].describe())
   ```

2. **检查CSV输出**
   ```python
   df = pd.read_csv('results/RUN_xxx_best.csv')
   print(df.columns)
   print(df[['y_real_raw', 'y_pred_raw']].describe())
   ```

3. **检查训练日志**
   - 查看是否打印了 `[标签策略]` 相关信息
   - 确认训练loss的数值范围是否合理

## 📝 注意事项

### 1. 不要混用参数

❌ **错误**: 同时使用 `--label_mode` 和 `--log_loss`
```bash
python train.py --label_mode log1p_mse --log_loss  # 会导致重复变换
```

✅ **正确**: 只使用 `--label_mode`
```bash
python train.py --label_mode log1p_mse
```

### 2. 理解训练loss的数值

- `baseline_raw_mse`: loss ≈ 几十到几百（原尺度）
- `log1p_mse`: loss ≈ 0.1-1.0（log空间）
- 最终评估指标都在原尺度上，可以直接对比

### 3. CSV文件的列名

- 四组实验都会输出 `y_real_raw` 和 `y_pred_raw`
- log1p_* 模式额外输出 `y_real_log` 和 `y_pred_log`
- 为了兼容性，也保留 `y_real` 和 `y_pred` 列

## 🎓 技术亮点

1. **最小侵入设计**: 仅修改3个核心文件，不影响其他功能
2. **统一接口**: 所有变换逻辑集中在 `train_step`，易于维护
3. **完整的逆变换**: 使用 `expm1(log1p(x))` 保证数值精度
4. **灵活的加权策略**: 支持自定义阈值和权重
5. **完善的文档**: 包含实现方案、使用指南、测试脚本

## 📚 相关文档

- `LABEL_STRATEGY_IMPLEMENTATION.md`: 详细实现方案
- `LABEL_STRATEGY_USAGE_GUIDE.md`: 完整使用指南
- `test_label_strategy.py`: 功能测试代码
- `analyze_ablation_results.py`: 结果分析代码

## 🤝 贡献

如需进一步优化或添加新的标签策略，请参考现有代码结构：

1. 在 `train_utils.train_step` 中添加新的 `elif` 分支
2. 在 `train.py` 的 `--label_mode` 参数中添加新的选项
3. 在 `test_step_w_id` 中添加相应的逆变换逻辑
4. 更新文档和测试脚本

## ✨ 总结

本实现完全满足您的需求：

✅ 改动最小（仅3个文件）  
✅ 可复现（支持多seed）  
✅ 接口统一（命令行参数切换）  
✅ 评估正确（原尺度计算指标）  
✅ 导出完整（CSV包含所有必要列）  

可以立即开始运行实验！

