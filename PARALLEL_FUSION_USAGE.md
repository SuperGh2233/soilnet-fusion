# 并行融合使用指南

## 概述

新增了并行版本的 S-CMRL 融合模块（`SemanticAlignedFusionParallel`），作为串行融合的替代方案。

## 串行 vs 并行融合

### 串行融合（Sequential，默认）
```
Climate + Visual → Result1
Result1 + Static → Final
```
- **特点**：Static 特征会基于已经融合了 Visual 的结果进行融合
- **可能的问题**：Static 和 Visual 之间存在依赖关系，梯度传播可能不够直接

### 并行融合（Parallel，新增）
```
Climate 查询 Visual → resid_visual
Climate 查询 Static → resid_static
Final = Climate + resid_visual + resid_static
```
- **特点**：Climate 分别独立查询 Visual 和 Static，然后将残差相加
- **优点**：
  1. Visual 和 Static 互不干扰，梯度传播更直接
  2. 物理意义更清晰：Climate 同时受到 Visual（当前地表）和 Static（固有环境）的修正
  3. 两个弱模态的贡献独立学习，可能更灵活

## 使用方法

### 命令行参数

```bash
python train.py \
    -e experiment_name \
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
    --use_scmrl_fusion \          # 启用 S-CMRL 融合
    --scmrl_parallel \             # 使用并行融合（不加此参数则使用串行融合）
    --scmrl_alpha_init 1.5 \
    --scmrl_lambda_align 0.1 \
    --scmrl_temperature 0.07 \
    -static \
    --static_csv dataset/CN-SOC-3500_new.csv \
    -seed 1
```

### 代码中使用

```python
from soilnet.submodules.semantic_aligned_fusion import SemanticAlignedFusionParallel

# 创建并行融合模块
fusion = SemanticAlignedFusionParallel(
    climate_dim=128,
    visual_dim=1024,
    static_dim=128,
    num_heads=8,
    alpha_init=1.5,
    learnable_alpha=True,
    dropout=0.1
)

# 前向传播
fused_feat = fusion(climate_feat, visual_feat, static_feat)
```

## 实现细节

### 并行融合公式

```python
# Step 1: 分别计算残差
resid_visual = alpha_v * Attention(Q=climate, K=visual, V=visual)
resid_static = alpha_s * Attention(Q=climate, K=static, V=static)

# Step 2: 统一相加
fused_feat = climate_feat + resid_visual + resid_static
```

### 关键区别

1. **残差计算**：两个残差都基于原始的 `climate_feat` 计算，而不是串行中的"先融合再融合"
2. **独立性**：Visual 和 Static 的注意力计算完全独立，互不影响
3. **梯度流**：梯度可以直接从 `fused_feat` 流向 `climate_feat`、`visual_feat` 和 `static_feat`

## 预期效果

### 理论优势

1. **更直接的梯度传播**：两个弱模态的梯度不会相互干扰
2. **更清晰的物理意义**：Climate 同时受到两个独立来源的修正
3. **更灵活的 Alpha 学习**：两个 Alpha 值可以独立调整

### 可能的效果

- **Alpha 值变化**：并行融合中，两个 Alpha 值可能会更快地收敛到最优值
- **训练稳定性**：梯度传播更直接，可能训练更稳定
- **性能提升**：如果 Visual 和 Static 确实是独立的，并行融合可能效果更好

## 对比实验建议

建议同时运行两个版本进行对比：

```bash
# 串行融合（基线）
python train.py ... --use_scmrl_fusion -e sequential_baseline

# 并行融合（实验）
python train.py ... --use_scmrl_fusion --scmrl_parallel -e parallel_experiment
```

对比指标：
- 训练损失和验证损失
- Alpha 值的变化趋势
- 最终性能（RMSE, MAE, R²）

## 注意事项

1. **兼容性**：并行融合与串行融合的接口完全兼容，可以无缝切换
2. **Alpha 值监控**：两种模式的 Alpha 值监控方式相同，都会在训练过程中打印
3. **模型保存**：两种模式的模型结构略有不同，不能直接互换使用

## 代码位置

- **并行融合实现**：`soilnet/submodules/semantic_aligned_fusion.py` - `SemanticAlignedFusionParallel` 类
- **串行融合实现**：`soilnet/submodules/semantic_aligned_fusion.py` - `SemanticAlignedFusion` 类
- **模型集成**：`soilnet/soil_net.py` - `SoilNetLSTM.__init__` 方法
- **命令行参数**：`train.py` - `--scmrl_parallel` 参数

