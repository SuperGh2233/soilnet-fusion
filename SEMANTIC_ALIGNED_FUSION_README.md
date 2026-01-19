# Semantic-Aligned Cross-Modal Residual Learning (S-CMRL) 融合模块

## 概述

基于 **S-CMRL** (Semantic-Alignment Cross-Modal Residual Learning) 的多模态融合模块，用于解决弱模态噪声干扰问题。

**参考**：
- GitHub 仓库: [Brain-Cog-Lab/S-CMRL](https://github.com/Brain-Cog-Lab/S-CMRL)
- 论文: "Enhancing Audio-Visual Spiking Neural Networks through Semantic-Alignment and Cross-Modal Residual Learning"

## 核心思想

### 1. 跨模态残差学习

**公式**：`F_final = F_climate + alpha * Attention(Q_cli, K_weak, V_weak)`

- **Query (Q)**: 来自 Climate 分支（强模态）
- **Key (K) & Value (V)**: 来自 Visual/Static 分支（弱模态）
- **alpha**: 可学习参数，初始化为 1.5

**为什么有效**：
- Climate 作为 Query，主动"检索"弱模态中有用的信息
- 残差连接保证强模态的主导地位
- 如果弱模态是噪声，alpha 会学习变小，自动屏蔽噪声

### 2. 语义对齐损失

基于对比学习（InfoNCE），拉近同一样本的 Climate-Visual 特征距离，推远不同样本间的距离。

**参数**：温度系数 `tau = 0.07`（参考 S-CMRL 仓库）

**为什么有效**：
- 迫使弱模态学习与强模态相关的语义，而不是随机特征
- 如果对齐损失下降，说明弱模态成功学习到了有用的语义

## 模块结构

### `CrossModalResidualBlock`

跨模态残差块，实现 Climate + Visual/Static 的融合。

```python
from soilnet.submodules.semantic_aligned_fusion import CrossModalResidualBlock

fusion_block = CrossModalResidualBlock(
    climate_dim=128,      # Climate 特征维度
    weak_dim=384,         # Visual/Static 特征维度
    num_heads=8,          # 注意力头数
    alpha_init=1.5,       # alpha 初始值
    learnable_alpha=True, # alpha 是否可学习
    dropout=0.1
)

fused_feat = fusion_block(climate_feat, visual_feat)
```

### `SemanticAlignmentLoss`

语义对齐损失函数。

```python
from soilnet.submodules.semantic_aligned_fusion import SemanticAlignmentLoss

alignment_loss_fn = SemanticAlignmentLoss(temperature=0.07)

loss = alignment_loss_fn(climate_feat, visual_feat)
```

### `SemanticAlignedFusion`

完整的融合模块，支持三个模态（Climate, Visual, Static）。

```python
from soilnet.submodules.semantic_aligned_fusion import SemanticAlignedFusion

fusion = SemanticAlignedFusion(
    climate_dim=128,      # Climate 特征维度
    visual_dim=384,       # Visual 特征维度
    static_dim=64,        # Static 特征维度（可选）
    num_heads=8,
    alpha_init=1.5,
    learnable_alpha=True
)

fused_feat = fusion(climate_feat, visual_feat, static_feat)
```

## 使用方法

### 方法 1: 在 SoilNetLSTM 中集成

```python
from soilnet.submodules.semantic_aligned_fusion import (
    SemanticAlignedFusion,
    SemanticAlignmentLoss
)

class SoilNetLSTMWithSCMRL(SoilNetLSTM):
    def __init__(self, ..., use_scmrl_fusion=True, **kwargs):
        super().__init__(...)
        
        if use_scmrl_fusion:
            # 创建 S-CMRL 融合模块
            self.fusion = SemanticAlignedFusion(
                climate_dim=lstm_out,      # 例如 128
                visual_dim=regresor_input_from_cnn,  # 例如 384 或 1024
                static_dim=static_dim,      # 如果有静态特征
                num_heads=8,
                alpha_init=1.5,
                learnable_alpha=True
            )
            
            # 语义对齐损失
            self.alignment_loss_fn = SemanticAlignmentLoss(temperature=0.07)
            
            # 回归头（输入维度与 Climate 相同）
            self.reg = nn.Linear(lstm_out, 1)
        else:
            # 使用原有的 MultiHeadRegressor
            self.reg = MultiHeadRegressor(...)
    
    def forward(self, input_raster_ts, ...):
        # 提取特征
        cnn_features = self.cnn(raster_stack)      # [B, visual_dim]
        climate_features = self.lstm(ts_features)   # [B, climate_dim]
        static_features = ...                       # [B, static_dim] (可选)
        
        if self.use_scmrl_fusion:
            # 使用 S-CMRL 融合
            fused_features = self.fusion(
                climate_features,
                cnn_features,
                static_features
            )  # [B, climate_dim]
            
            # 回归预测
            output = self.reg(fused_features)
        else:
            # 原有方法：简单 concat
            output = self.reg(cnn_features, climate_features, ...)
        
        return output
```

### 方法 2: 在训练循环中使用对齐损失

```python
# 在 train_utils.py 的 train_step 中：

def train_step(model, batch, optimizer, loss_fn, 
               alignment_loss_fn=None, lambda_align=0.1):
    # ... 前向传播 ...
    y_pred = model(inputs)
    
    # 主损失
    main_loss = loss_fn(y_pred, y_true)
    
    # 语义对齐损失（如果使用 S-CMRL）
    total_loss = main_loss
    if alignment_loss_fn is not None:
        # 获取中间特征（需要在模型中返回）
        climate_feat = model.get_climate_feat()
        visual_feat = model.get_visual_feat()
        align_loss = alignment_loss_fn(climate_feat, visual_feat)
        total_loss = main_loss + lambda_align * align_loss
    
    # 反向传播
    total_loss.backward()
    optimizer.step()
    
    return {
        'loss': main_loss.item(),
        'align_loss': align_loss.item() if alignment_loss_fn else 0.0,
        'total_loss': total_loss.item()
    }
```

### 方法 3: 监控 Alpha 值

```python
# 在训练过程中定期打印 alpha 值，观察弱模态的贡献：

if epoch % 10 == 0:
    alphas = model.fusion.get_alpha_values()
    print(f"Epoch {epoch} Alpha values:")
    for key, value in alphas.items():
        print(f"  {key}: {value:.4f}")
    
    # 如果 alpha 变得很小（接近 0），说明模型认为弱模态是噪声
    # 如果 alpha 稳定在正值，说明模型成功提取到了互补信息
```

## 验证修改是否有效

### 1. 观察 Alpha 值的变化

- **如果 alpha -> 0**：模型认为弱模态是噪声，自动屏蔽（成功）
- **如果 alpha 稳定在正值**：模型成功提取到互补信息（成功）

### 2. 观察 Alignment Loss 的下降

- **如果对齐损失下降**：弱模态学习到了与强模态相关的语义（成功）
- **如果对齐损失不下降**：可能需要调整温度参数或损失权重

### 3. 对比性能指标

- 使用 S-CMRL 融合 vs 简单 concat
- 如果 S-CMRL 性能更好或至少不掉点，说明有效

## 测试

运行测试脚本：

```bash
# 测试模块本身
python soilnet/submodules/semantic_aligned_fusion.py

# 测试集成
python test_semantic_aligned_fusion.py
```

## 文件结构

```
soilnet/submodules/
├── semantic_aligned_fusion.py  # S-CMRL 融合模块
└── ...

test_semantic_aligned_fusion.py  # 集成测试示例
SEMANTIC_ALIGNED_FUSION_README.md  # 本文档
```

## 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `alpha_init` | 1.5 | Alpha 参数的初始值（参考 S-CMRL 仓库） |
| `learnable_alpha` | True | Alpha 是否可学习 |
| `temperature` | 0.07 | 语义对齐损失的温度系数（参考 S-CMRL 仓库） |
| `num_heads` | 8 | 多头注意力的头数 |
| `dropout` | 0.1 | Dropout 概率 |
| `lambda_align` | 0.1 | 对齐损失的权重（在总损失中） |

## 原理总结

**为什么 S-CMRL 能防止弱模态变成噪声？**

1. **残差连接保证强模态主导**：
   - `F_final = F_climate + alpha * Attention(...)`
   - 即使弱模态是噪声，强模态仍然占主导地位

2. **可学习的 alpha 参数**：
   - 如果弱模态是噪声，模型会学习让 alpha -> 0，自动屏蔽噪声
   - 如果弱模态有用，alpha 会稳定在正值，提取互补信息

3. **语义对齐损失**：
   - 迫使弱模态学习与强模态相关的语义
   - 如果对齐损失下降，说明弱模态成功学习到了有用的特征

4. **跨模态注意力机制**：
   - Climate 作为 Query，主动"检索"弱模态中有用的信息
   - 只提取与 Climate 相关的特征，忽略无关噪声

## 参考

- [S-CMRL GitHub Repository](https://github.com/Brain-Cog-Lab/S-CMRL)
- 论文: "Enhancing Audio-Visual Spiking Neural Networks through Semantic-Alignment and Cross-Modal Residual Learning"








