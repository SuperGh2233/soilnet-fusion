# S-CMRL Fusion 实现说明文档

## 概述

S-CMRL (Semantic-Alignment Cross-Modal Residual Learning) 是一种多模态融合方法，用于融合 Climate（强模态）、Visual（弱模态）和 Static（弱模态）特征。

### 核心思想

1. **跨模态残差学习**：强模态（Climate）作为 Query），主动检索弱模态中的有用信息
2. **语义对齐损失**：通过对比学习拉近同一样本的不同模态特征，推远不同样本的特征
3. **可学习的 alpha 参数**：控制弱模态的贡献，如果弱模态是噪声，alpha 会自动变小

### 为什么能防止弱模态变成噪声

- **残差连接保证强模态的主导地位**：`F_final = F_climate + alpha * Attention(...)`
- **如果弱模态是噪声，Attention 机制会让模型学习忽略它**（alpha → 0）
- **语义对齐损失迫使弱模态学习与强模态相关的语义**，而不是随机特征

## 实现架构

### 1. 核心模块

#### 1.1 CrossModalResidualBlock（跨模态残差块）

**位置**：`soilnet/submodules/semantic_aligned_fusion.py`

**实现公式**：
```
F_final = F_climate + alpha * Attention(Q_cli, K_weak, V_weak)
```

**关键组件**：
- **Query (Q)**：来自 Climate 分支（强模态）
- **Key, Value (K, V)**：来自 Visual/Static 分支（弱模态）
- **Alpha**：可学习参数，初始化为 1.5

**代码结构**：
```python
class CrossModalResidualBlock(nn.Module):
    def __init__(self, climate_dim, weak_dim, num_heads=8, 
                 alpha_init=1.5, learnable_alpha=True, dropout=0.1):
        # 投影层：将弱模态特征投影到与 Climate 相同的维度
        self.weak_proj = nn.Linear(weak_dim, climate_dim)
        
        # 跨模态注意力：Climate 作为 Query，弱模态作为 Key/Value
        self.cross_attention = nn.MultiheadAttention(...)
        
        # Alpha 参数：控制弱模态的贡献
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
    
    def forward(self, climate_feat, weak_feat):
        # 投影弱模态特征
        weak_proj = self.weak_proj(weak_feat)
        
        # 跨模态注意力
        attn_out, _ = self.cross_attention(
            query=climate_feat,
            key=weak_proj,
            value=weak_proj
        )
        
        # 残差连接
        fused_feat = climate_feat + self.alpha * self.dropout(attn_out)
        return self.norm(fused_feat)
```

#### 1.2 SemanticAlignedFusion（语义对齐融合模块）

**位置**：`soilnet/submodules/semantic_aligned_fusion.py`

**融合策略**：
1. 先融合 Climate + Visual（通过 `CrossModalResidualBlock`）
2. 再融合结果 + Static（通过另一个 `CrossModalResidualBlock`）
3. 最终输出与 Climate 维度相同

**代码结构**：
```python
class SemanticAlignedFusion(nn.Module):
    def __init__(self, climate_dim, visual_dim, static_dim=None, ...):
        # Climate + Visual 融合
        self.climate_visual_fusion = CrossModalResidualBlock(...)
        
        # Climate-Visual 融合结果 + Static 融合
        if static_dim is not None:
            self.climate_static_fusion = CrossModalResidualBlock(...)
    
    def forward(self, climate_feat, visual_feat, static_feat=None):
        # Step 1: Climate + Visual 融合
        fused = self.climate_visual_fusion(climate_feat, visual_feat)
        
        # Step 2: 如果提供了 Static 特征，继续融合
        if static_feat is not None:
            fused = self.climate_static_fusion(fused, static_feat)
        
        return fused
```

#### 1.3 SemanticAlignmentLoss（语义对齐损失）

**位置**：`soilnet/submodules/semantic_aligned_fusion.py`

**原理**：基于对比学习（InfoNCE），拉近同一样本的 Climate-Visual 特征距离，推远不同样本间的距离。

**公式**：
```
Loss = -log(exp(sim(climate_i, visual_i) / T) / 
            sum_j(exp(sim(climate_i, visual_j) / T)))
```

其中：
- `sim(a, b)` 是余弦相似度
- `T` 是温度参数（默认 0.07）

### 2. 模型集成

#### 2.1 SoilNetLSTM 中的集成

**位置**：`soilnet/soil_net.py`

**初始化代码**：
```python
if self.use_scmrl_fusion:
    # 创建融合模块
    self.fusion = SemanticAlignedFusion(
        climate_dim=lstm_out,              # 例如 128
        visual_dim=regresor_input_from_cnn, # 例如 384 或 1024
        static_dim=static_dim,              # 如果有静态特征
        num_heads=8,
        alpha_init=scmrl_alpha_init,
        learnable_alpha=True,
        dropout=0.1
    )
    
    # 语义对齐损失
    self.alignment_loss_fn = SemanticAlignmentLoss(
        temperature=scmrl_temperature
    )
    
    # 回归头（输入维度与 Climate 相同）
    self.reg = nn.Linear(lstm_out, 1)
```

**前向传播**：
```python
def forward(self, input_raster_ts, region_ids=None):
    # 提取特征
    cnn_features = self.cnn(raster_stack)      # [B, visual_dim]
    climate_features = self.lstm(ts_features)  # [B, climate_dim]
    
    # 保存中间特征（用于对齐损失计算）
    self._last_climate_feat = climate_features
    self._last_visual_feat = cnn_features
    
    if self.use_scmrl_fusion:
        # 获取静态特征（如果有）
        static_feat = None
        if isinstance(input_raster_ts, (list, tuple)) and len(input_raster_ts) >= 3:
            static_feat = input_raster_ts[2]
        
        # 使用 S-CMRL 融合
        fused_feat = self.fusion(climate_features, cnn_features, static_feat)
        
        # 回归预测
        output = self.reg(fused_feat)
    else:
        # 原有方法：简单 concat
        output = self.reg(cnn_features, climate_features, ...)
    
    return output
```

#### 2.2 SoilNetLSTMWithStatic 中的集成

**位置**：`soilnet/soil_net_static.py`

**关键点**：
- 继承自 `SoilNetLSTM`，自动获得 SCMRL fusion 支持
- 在 `forward` 中正确处理静态特征
- 保存中间特征用于对齐损失计算

**前向传播**：
```python
def forward(self, input_raster_ts_static, region_ids=None):
    # 解包输入
    raster_stack = input_raster_ts_static[0]
    ts_features = input_raster_ts_static[1]
    static_features = input_raster_ts_static[2] if len(input_raster_ts_static) >= 3 else None
    
    # 提取特征
    cnn_features = self.cnn(raster_stack)
    climate_features = self.lstm(ts_features)
    
    # 保存中间特征（用于对齐损失计算）
    if hasattr(self, 'use_scmrl_fusion') and self.use_scmrl_fusion:
        self._last_climate_feat = climate_features
        self._last_visual_feat = cnn_features
    
    # S-CMRL 融合或原有融合方式
    if hasattr(self, 'use_scmrl_fusion') and self.use_scmrl_fusion:
        # 使用 S-CMRL 融合（静态特征会参与融合）
        fused_feat = self.fusion(climate_features, cnn_features, static_features)
        output = self.reg(fused_feat)
    else:
        # 旧版本：直接拼接
        reg_inputs = [cnn_features, climate_features, static_features]
        output = self.reg(*reg_inputs)
    
    return output
```

### 3. 训练集成

#### 3.1 损失计算

**位置**：`train_utils.py`

**训练步骤**：
```python
def train_step(model, batch, optimizer, loss_fn, 
               alignment_loss_fn=None, lambda_align=0.1):
    # 前向传播
    y_pred = model(inputs)
    
    # 主损失
    main_loss = loss_fn(y_pred, y_true)
    
    # 语义对齐损失（如果使用 S-CMRL）
    total_loss = main_loss
    if alignment_loss_fn is not None and model.use_scmrl_fusion:
        # 获取中间特征（在 forward 中已保存）
        climate_feat = model._last_climate_feat
        visual_feat = model._last_visual_feat
        
        align_loss = alignment_loss_fn(climate_feat, visual_feat)
        total_loss = main_loss + lambda_align * align_loss
    
    # 反向传播
    total_loss.backward()
    optimizer.step()
```

#### 3.2 Alpha 值监控

**位置**：`train_utils.py`

**每个 epoch 结束后自动打印**：
```python
# 打印 Alpha 值（如果使用 S-CMRL 融合）
if hasattr(model, 'use_scmrl_fusion') and model.use_scmrl_fusion and hasattr(model, 'fusion'):
    alphas = model.fusion.get_alpha_values()
    alpha_msg = "Alpha values: "
    alpha_items = [f"{k}={v:.4f}" for k, v in alphas.items()]
    alpha_msg += " | ".join(alpha_items)
    print(alpha_msg)
```

## 使用方法

### 1. 命令行参数

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
    --use_scmrl_fusion \              # 启用 S-CMRL 融合
    --scmrl_alpha_init 1.5 \          # Alpha 初始值（默认 1.5）
    --scmrl_lambda_align 0.1 \        # 对齐损失权重（默认 0.1）
    --scmrl_temperature 0.07 \         # 对齐损失温度（默认 0.07）
    -static \                          # 可选：启用静态特征
    --static_csv dataset/CN-SOC-3500_new.csv \
    -seed 1
```

### 2. 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_scmrl_fusion` | flag | False | 启用 S-CMRL 融合 |
| `--scmrl_alpha_init` | float | 1.5 | Alpha 参数初始值 |
| `--scmrl_lambda_align` | float | 0.1 | 对齐损失在总损失中的权重 |
| `--scmrl_temperature` | float | 0.07 | 对齐损失的温度参数 |

### 3. 与静态特征结合使用

当同时启用 `--use_scmrl_fusion` 和 `-static` 时：

1. **静态特征会参与 S-CMRL 融合**：
   - 先融合 Climate + Visual
   - 再融合结果 + Static

2. **Alpha 值监控**：
   - `climate_visual`：控制 Visual 模态的贡献
   - `climate_static`：控制 Static 模态的贡献

3. **回归器设置**：
   - 使用 SCMRL fusion：回归器为 `nn.Linear`，接收融合后的单一特征向量
   - 不使用 SCMRL fusion：回归器为 `MultiHeadRegressor`，接收多个输入

## 训练输出示例

### 1. 初始化信息

```
[Info] Using S-CMRL Fusion (Semantic-Alignment Cross-Modal Residual Learning)
       alpha_init=1.5, temperature=0.07
[Info] S-CMRL 对齐损失已启用: lambda=0.1, temperature=0.07
```

### 2. 训练过程输出

```
Epoch 1 Results: | train_loss: 0.123456 | val_loss: 0.234567 | align_loss: 1.234567
Alpha values: climate_visual=1.5000 | climate_static=1.5000

Epoch 2 Results: | train_loss: 0.112345 | val_loss: 0.223456 | align_loss: 1.123456
Alpha values: climate_visual=1.4523 | climate_static=1.4891

...
```

### 3. Alpha 值解读

- **Alpha 接近初始值（1.5）**：弱模态贡献正常
- **Alpha 变小（接近 0）**：模型认为弱模态是噪声，自动屏蔽
- **Alpha 稳定在正值**：模型成功提取到互补信息

## 复现步骤

### 1. 环境准备

确保已安装必要的依赖：
```bash
pip install torch torchvision
```

### 2. 数据准备

确保数据集已准备好：
- 影像数据
- 气候时间序列数据
- 静态特征数据（如果使用 `-static`）

### 3. 训练命令

#### 3.1 仅使用 S-CMRL 融合（无静态特征）

```bash
python train.py \
    -e scmrl_experiment \
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
    --use_scmrl_fusion \
    --scmrl_alpha_init 1.5 \
    --scmrl_lambda_align 0.1 \
    --scmrl_temperature 0.07 \
    -seed 1
```

#### 3.2 使用 S-CMRL 融合 + 静态特征

```bash
python train.py \
    -e scmrl_static_experiment \
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
    --use_scmrl_fusion \
    --scmrl_alpha_init 1.5 \
    --scmrl_lambda_align 0.1 \
    --scmrl_temperature 0.07 \
    -static \
    --static_csv dataset/CN-SOC-3500_new.csv \
    -seed 1
```

### 4. 监控训练

训练过程中会输出：
- **每个 epoch 的损失**：train_loss, val_loss, align_loss
- **Alpha 值**：climate_visual, climate_static（如果使用静态特征）

### 5. 查看保存的模型

训练结果会保存在 `results/` 目录下，JSON 文件中包含：
```json
{
    "USE_SCMRL_FUSION": true,
    "SCMRL_ALPHA_INIT": 1.5,
    "SCMRL_LAMBDA_ALIGN": 0.1,
    "SCMRL_TEMPERATURE": 0.07,
    ...
}
```

### 6. 检查 Alpha 值（可选）

使用 `check_alpha_values.py` 脚本查看已保存模型的 Alpha 值：

```bash
python check_alpha_values.py --model_path results/RUN_scmrl_experiment_best.pth.tar
```

## 关键代码文件

1. **核心实现**：
   - `soilnet/submodules/semantic_aligned_fusion.py`：SCMRL fusion 核心模块

2. **模型集成**：
   - `soilnet/soil_net.py`：`SoilNetLSTM` 中的集成
   - `soilnet/soil_net_static.py`：`SoilNetLSTMWithStatic` 中的集成

3. **训练集成**：
   - `train.py`：命令行参数和模型创建
   - `train_utils.py`：损失计算和 Alpha 值监控

4. **工具脚本**：
   - `check_alpha_values.py`：查看已保存模型的 Alpha 值

## 注意事项

1. **回归器维度**：
   - 使用 SCMRL fusion 时，回归器输入维度为 `lstm_out`（通常是 128）
   - 不使用 SCMRL fusion 时，回归器接收多个输入（cnn_dim, lstm_dim, static_dim）

2. **静态特征处理**：
   - 使用 SCMRL fusion 时：静态特征通过 `SemanticAlignedFusion` 融合
   - 不使用 SCMRL fusion 时：静态特征直接拼接（旧版本）

3. **中间特征保存**：
   - 模型在 `forward` 中会保存 `_last_climate_feat` 和 `_last_visual_feat`
   - 这些特征用于计算对齐损失

4. **Alpha 值监控**：
   - 每个 epoch 结束后自动打印
   - 可以通过 `model.fusion.get_alpha_values()` 获取

## 参考

- 原始实现参考：Brain-Cog-Lab/S-CMRL 仓库
- 论文："Enhancing Audio-Visual Spiking Neural Networks through Semantic-Alignment and Cross-Modal Residual Learning"

