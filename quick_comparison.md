# 实验结果快速对比

## 当前实验结果

| 实验 | RMSE | R² | MAE | 说明 |
|------|------|----|----|------|
| **log1p_mse** | 8.95 | 0.355 | 5.11 | 对数变换 + MSE |
| **log1p_huber** | 8.94 | 0.357 | 5.12 | 对数变换 + Huber |
| **baseline_raw_mse** | ? | ? | ? | **需要运行** |

## 观察

1. **log1p_mse 和 log1p_huber 几乎相同**
   - RMSE 差异 < 0.01
   - R² 差异 < 0.002
   - 说明 Huber 损失在这个数据集上没有带来明显改进

2. **缺少真正的基线对比**
   - 需要运行 `baseline_raw_mse` 来对比
   - 如果 baseline 更好，说明 log1p 变换可能不适合

## 下一步建议

### 选项1：运行基线实验（推荐）
```bash
python train.py -e baseline_raw_mse --label_mode baseline_raw_mse \
    -d CHINA -w 6 -cnn ViT-CoMer -rnn Transformer -lstm \
    -trbs 32 -ne 60 -lr 0.0001 -ls step -stm -srtm -seed 1
```

### 选项2：尝试加权版本
如果 baseline 效果也不好，可以尝试：
```bash
python train.py -e log1p_huber_w --label_mode log1p_huber_w \
    --huber_beta 1.0 --tail_threshold 30.0 --tail_weight 2.0 \
    -d CHINA -w 6 -cnn ViT-CoMer -rnn Transformer -lstm \
    -trbs 32 -ne 60 -lr 0.0001 -ls step -stm -srtm -seed 1
```

### 选项3：如果所有策略效果都不好
可能需要考虑：
1. **数据问题**：检查数据质量、分布
2. **模型架构**：可能需要调整模型结构
3. **超参数**：学习率、batch size 等
4. **特征工程**：可能需要更好的特征

## 分析建议

运行 baseline 后，对比：
- 如果 baseline > log1p：说明 log1p 不适合，回到原尺度
- 如果 baseline ≈ log1p：说明变换没有帮助，需要其他改进
- 如果 baseline < log1p：说明 log1p 有帮助，但需要进一步优化

































