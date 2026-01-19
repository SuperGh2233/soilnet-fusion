# 如何查看 Alpha 值

Alpha 值是 S-CMRL 融合模块中的可学习参数，用于控制弱模态（Visual/Static）的贡献。以下是查看 Alpha 值的几种方法：

## 方法 1: 训练过程中自动打印（推荐）

**已自动集成**：在训练过程中，每个 epoch 结束后会自动打印 Alpha 值。

运行训练命令：
```bash
python train.py -e newfusion -d CHINA -w 6 -cnn ViT-CoMer -rnn Transformer \
                -trbs 32 -ne 60 -lr 0.0001 -ls step -stm -srtm -static \
                --use_scmrl_fusion
```

**输出示例**：
```
Epoch 1 Results: | train_loss: 0.123456 | val_loss: 0.234567 | align_loss: 1.234567
Alpha values: climate_visual=1.5000 | climate_static=1.5000

Epoch 2 Results: | train_loss: 0.112345 | val_loss: 0.223456 | align_loss: 1.123456
Alpha values: climate_visual=1.4523 | climate_static=1.4891
...
```

## 方法 2: 从已保存的模型查看

使用 `check_alpha_values.py` 脚本：

```bash
# 查看最佳模型
python check_alpha_values.py --model_path results/RUN_newfusion_D_2026_01_15_T_XX_XX_best.pth.tar
```

**输出示例**：
```
======================================================================
Alpha 值（控制弱模态贡献）
======================================================================
  climate_visual      :   1.4523
    -> Visual 分支贡献正常
  climate_static      :   0.0891
    -> Static 分支贡献很小（接近噪声，已被屏蔽）
======================================================================
```

## 方法 3: 在 Python 代码中查看

### 3.1 训练过程中查看

在训练循环中添加：

```python
# 在 train.py 的训练循环中
if epoch % 10 == 0:  # 每 10 个 epoch 打印一次
    if hasattr(model, 'fusion'):
        alphas = model.fusion.get_alpha_values()
        print(f"\nEpoch {epoch} Alpha values:")
        for key, value in alphas.items():
            print(f"  {key}: {value:.4f}")
```

### 3.2 加载模型后查看

```python
import torch
from soilnet.soil_net import SoilNetLSTM
from check_alpha_values import print_alpha_values

# 加载模型
checkpoint = torch.load('results/RUN_xxx_best.pth.tar', map_location='cpu')
model = SoilNetLSTM(...)  # 使用相同的参数创建模型
model.load_state_dict(checkpoint['model_state_dict'])

# 查看 Alpha 值
print_alpha_values(model)
```

### 3.3 直接访问模型参数

```python
# 如果模型已加载
if hasattr(model, 'fusion'):
    # 方法 1: 使用 get_alpha_values() 方法（推荐）
    alphas = model.fusion.get_alpha_values()
    print(alphas)
    
    # 方法 2: 直接访问参数
    climate_visual_alpha = model.fusion.climate_visual_fusion.alpha.item()
    climate_static_alpha = model.fusion.climate_static_fusion.alpha.item()
    print(f"Climate-Visual Alpha: {climate_visual_alpha:.4f}")
    print(f"Climate-Static Alpha: {climate_static_alpha:.4f}")
```

## 方法 4: 从 JSON 结果文件查看

训练完成后，Alpha 的初始值会保存在 JSON 结果文件中：

```bash
# 查看 JSON 文件
cat results/RUN_newfusion_D_2026_01_15_T_XX_XX.json | grep -i alpha
```

**注意**：JSON 文件中只保存初始值，训练后的值需要从模型 checkpoint 中读取。

## Alpha 值的含义

| Alpha 值范围 | 含义 | 说明 |
|-------------|------|------|
| < 0.1 | 弱模态贡献很小 | 模型认为弱模态是噪声，已自动屏蔽 |
| 0.1 - 0.5 | 弱模态贡献较小 | 模型提取到少量有用信息 |
| 0.5 - 1.5 | 弱模态贡献正常 | 模型成功提取到互补信息 |
| > 1.5 | 弱模态贡献较大 | 弱模态提供了重要信息 |

## 如何判断 S-CMRL 是否有效？

### 1. Alpha 值的变化趋势

**成功的情况**：
- Alpha 值稳定在正值（0.5-2.0）：说明模型成功提取到互补信息
- Alpha 值逐渐变小但稳定：说明模型学会了筛选有用信息

**失败的情况**：
- Alpha 值一直很大（>3.0）：可能弱模态噪声太大，模型无法有效筛选
- Alpha 值震荡剧烈：可能需要调整学习率或损失权重

### 2. 对比实验

运行两个实验：
- **实验 A**：不使用 S-CMRL（`--use_scmrl_fusion` 不添加）
- **实验 B**：使用 S-CMRL（`--use_scmrl_fusion`）

对比性能指标（RMSE, MAE, R²），如果实验 B 更好或至少不掉点，说明 S-CMRL 有效。

## 快速测试

运行测试脚本验证 Alpha 值查看功能：

```bash
python check_alpha_values.py --model_path <你的模型路径>
```

或者运行集成测试：

```bash
python test_scmrl_integration.py
```

## 常见问题

**Q: 为什么看不到 Alpha 值？**
A: 确保：
1. 使用了 `--use_scmrl_fusion` 参数
2. 模型确实创建了 fusion 模块
3. 检查训练日志中是否有 "[Info] Using S-CMRL Fusion" 消息

**Q: Alpha 值一直是初始值 1.5？**
A: 可能的原因：
1. 训练轮数太少，参数还没更新
2. 学习率太小
3. 对齐损失权重太小（`lambda_align`）

**Q: 如何调整 Alpha 的初始值？**
A: 使用 `--scmrl_alpha_init` 参数：
```bash
python train.py ... --use_scmrl_fusion --scmrl_alpha_init 2.0
```

