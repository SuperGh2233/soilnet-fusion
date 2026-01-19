# 标签策略开关 - 快速参考

## 🎯 一句话总结

通过 `--label_mode` 参数一键切换四种标签策略，用于长尾 SOC 回归消融实验。

## 🚀 快速开始（3步）

### 1️⃣ 测试功能
```bash
python test_label_strategy.py
```

### 2️⃣ 运行实验
```bash
# 单个实验（快速验证）
python train.py -e baseline_raw_mse --label_mode baseline_raw_mse -ne 10 -seed 1 -lstm

# 完整消融实验（4组）
bash run_ablation_experiments.sh
```

### 3️⃣ 查看结果
```bash
python analyze_ablation_results.py
```

## 📊 四种模式对比

| 模式 | 训练目标 | 损失函数 | 命令行 |
|------|---------|---------|--------|
| **baseline_raw_mse** | 原尺度 SOC | MSE | `--label_mode baseline_raw_mse` |
| **log1p_mse** | log1p(SOC) | MSE | `--label_mode log1p_mse` |
| **log1p_huber** | log1p(SOC) | Huber | `--label_mode log1p_huber --huber_beta 1.0` |
| **log1p_huber_w** | log1p(SOC) | Huber+加权 | `--label_mode log1p_huber_w --tail_threshold 30.0 --tail_weight 2.0` |

## 📝 完整命令示例

```bash
# 实验1: baseline
python train.py -e baseline_raw_mse --label_mode baseline_raw_mse -ne 100 -seed 1 2 3 -lstm -cnn ViT-CoMer -rnn Transformer

# 实验2: log1p + MSE
python train.py -e log1p_mse --label_mode log1p_mse -ne 100 -seed 1 2 3 -lstm -cnn ViT-CoMer -rnn Transformer

# 实验3: log1p + Huber
python train.py -e log1p_huber --label_mode log1p_huber --huber_beta 1.0 -ne 100 -seed 1 2 3 -lstm -cnn ViT-CoMer -rnn Transformer

# 实验4: log1p + Huber + 加权
python train.py -e log1p_huber_w --label_mode log1p_huber_w --huber_beta 1.0 --tail_threshold 30.0 --tail_weight 2.0 -ne 100 -seed 1 2 3 -lstm -cnn ViT-CoMer -rnn Transformer
```

## 📂 输出文件

### JSON 结果
```
results/RUN_{EXP_NAME}_{timestamp}.json
```
包含字段: `LABEL_MODE`, `HUBER_BETA`, `TAIL_THRESHOLD`, `TAIL_WEIGHT`, `RMSE_MEAN`, `R2_MEAN`, `best_dict`

### CSV 预测
```
results/RUN_{EXP_NAME}_{timestamp}_best.csv
```
包含列: `point_id`, `y_real_raw`, `y_pred_raw`, `y_real_log`, `y_pred_log`

## 🔍 验证实验

### 检查数据集
```python
import pandas as pd
df = pd.read_csv('dataset/CN-SOC-3500_new.csv')
print(f"OC 范围: {df['OC'].min():.2f} - {df['OC'].max():.2f}")
print(f"OC > 30 样本数: {(df['OC'] > 30).sum()} ({(df['OC'] > 30).mean()*100:.1f}%)")
```

### 检查预测结果
```python
df = pd.read_csv('results/RUN_xxx_best.csv')
print(df[['y_real_raw', 'y_pred_raw']].describe())
```

## 📚 详细文档

| 文档 | 内容 |
|------|------|
| `IMPLEMENTATION_SUMMARY.md` | ✅ 实现总结（推荐先看） |
| `LABEL_STRATEGY_USAGE_GUIDE.md` | 📖 完整使用指南 |
| `LABEL_STRATEGY_IMPLEMENTATION.md` | 🔧 技术实现细节 |

## ⚠️ 重要提示

1. **不要混用**: 不要同时使用 `--label_mode` 和 `--log_loss`
2. **训练loss数值**: log1p模式的训练loss会比baseline小很多（正常现象）
3. **评估指标**: 所有指标都在原尺度上计算，可以直接对比
4. **CSV列名**: 使用 `y_real_raw` 和 `y_pred_raw` 进行分析

## 🎓 核心原理

```
数据集 → 原尺度标签 (不归一化)
         ↓
训练时 → 根据label_mode变换 (raw / log1p)
         ↓
计算loss → MSE / Huber / Weighted Huber
         ↓
评估时 → 转换回原尺度
         ↓
输出CSV → y_real_raw, y_pred_raw (原尺度)
         ↓
计算指标 → RMSE, MAE, R² (原尺度)
```

## ✅ 验收清单

- [x] 四组实验仅通过参数切换
- [x] 数据集输出原尺度标签
- [x] log1p变换在训练循环中
- [x] 评估指标在原尺度上计算
- [x] CSV包含原尺度列
- [x] 向后兼容旧版代码

## 🤝 需要帮助？

1. 运行测试: `python test_label_strategy.py`
2. 查看日志: 训练时会打印 `[标签策略]` 相关信息
3. 检查文档: 查看 `IMPLEMENTATION_SUMMARY.md`

---

**祝实验顺利！** 🎉

