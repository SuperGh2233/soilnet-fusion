# 静态特征未带来训练效果提升的可能原因分析

## 问题现象

从训练输出可以看到：
- Alpha值：`climate_static=1.4982`（接近初始值1.5）
- 说明模型可能没有学习到如何有效利用静态特征

## 可能的原因

### 1. 静态特征数据问题

#### 1.1 Point_ID 匹配失败
**症状**：数据集中很多样本的静态特征为0或缺失

**检查方法**：
```python
# 在训练开始前检查
from diagnose_static_features import check_point_id_matching
check_point_id_matching(train_ds, num_samples=100)
```

**可能原因**：
- Point_ID格式不一致（例如："123" vs "123.0" vs 123）
- 静态特征CSV中的Point_ID与数据集中的Point_ID不匹配

**解决方案**：
- 统一Point_ID格式
- 检查数据集加载器中的Point_ID匹配逻辑

#### 1.2 静态特征全为0或缺失值过多
**症状**：静态特征的值都是0或NaN

**检查方法**：
```python
from diagnose_static_features import check_data_distribution
check_data_distribution(train_ds, num_samples=100)
```

**可能原因**：
- 静态特征CSV中数据缺失
- 数据预处理时填充为0
- 特征列选择错误

**解决方案**：
- 检查静态特征CSV文件
- 使用合理的缺失值填充策略（如均值填充）
- 验证特征列是否正确

#### 1.3 静态特征方差为0（常数特征）
**症状**：某些静态特征在所有样本中值相同

**检查方法**：
```python
# 计算每个特征的方差
static_features = np.array([feat for feat in train_ds.static_numeric_features.values()])
variances = static_features.var(axis=0)
constant_features = np.where(variances < 1e-6)[0]
print(f"常数特征索引: {constant_features}")
```

**解决方案**：
- 移除常数特征
- 检查数据预处理是否正确

### 2. 模型实现问题

#### 2.1 静态特征未正确传递到模型
**症状**：`static_encoded`为None或维度不匹配

**检查方法**：
```python
# 在forward中添加调试输出
def forward(self, input_raster_ts_static, ...):
    static_features = input_raster_ts_static[2] if len(input_raster_ts_static) >= 3 else None
    print(f"DEBUG: static_features shape={static_features.shape if static_features is not None else None}")
    # ...
```

**可能原因**：
- 数据加载器返回的格式不正确
- 输入解包逻辑错误

**解决方案**：
- 检查数据加载器的`__getitem__`方法
- 确保返回格式为：`(影像, 气候, 静态数值特征, LULC索引)`

#### 2.2 StaticBranch输出维度不匹配
**症状**：SCMRL fusion的`static_dim`与StaticBranch输出维度不一致

**当前实现**：
- StaticBranch输出维度：`hidden_size` (128)
- SCMRL fusion `static_dim`：`hidden_size` (128)
- ✅ 应该匹配

**检查方法**：
```python
# 检查维度
print(f"StaticBranch hidden: {model.static_branch.hidden}")
print(f"SCMRL fusion static_dim: {model.fusion.static_dim}")
print(f"是否匹配: {model.static_branch.hidden == model.fusion.static_dim}")
```

#### 2.3 SCMRL Fusion未正确接收静态特征
**症状**：`static_encoded`为None时，SCMRL fusion不会使用静态特征

**当前代码逻辑**：
```python
static_encoded = None
if (static_features is not None) and self.static_feature_dim > 0:
    if (self.static_branch is not None):
        static_encoded = self.static_branch(static_features, lulc_indices)
    else:
        static_encoded = static_features

fused_feat = self.fusion(climate_features, cnn_features, static_encoded)
```

**问题**：如果`static_features`为None或`static_feature_dim`为0，`static_encoded`会是None

**检查方法**：
```python
# 在forward中添加调试
if static_encoded is None:
    print("⚠️ 警告: static_encoded 为 None，静态特征不会参与融合")
```

### 3. 训练问题

#### 3.1 Alpha值未学习
**症状**：Alpha值接近初始值1.5，没有变化

**可能原因**：
- 学习率太小
- 静态特征梯度为0
- 损失函数权重设置不当

**解决方案**：
- 检查Alpha参数是否可学习：`learnable_alpha=True`
- 增加对齐损失的权重：`--scmrl_lambda_align 0.2`（默认0.1）
- 检查梯度流：`static_encoded.grad is not None`

#### 3.2 静态特征梯度消失
**症状**：静态特征分支的梯度为0

**检查方法**：
```python
# 在训练循环中检查梯度
for name, param in model.named_parameters():
    if 'static' in name.lower():
        if param.grad is not None:
            print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
        else:
            print(f"{name}: grad=None")
```

**解决方案**：
- 检查StaticBranch的dropout是否过大
- 检查学习率设置
- 确保静态特征参与损失计算

### 4. 数据质量问题

#### 4.1 静态特征与SOC相关性低
**症状**：静态特征本身与目标变量（SOC）相关性不高

**检查方法**：
```python
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# 加载数据
static_df = pd.read_csv('dataset/CN-SOC-3500_new.csv')
soc = static_df['SOC'].values

# 计算相关性
for col in static_df.columns:
    if col not in ['Point_ID', 'SOC', 'Latitude', 'Longitude']:
        if static_df[col].dtype in [np.float64, np.int64]:
            corr, p_value = pearsonr(static_df[col].dropna(), soc[~static_df[col].isna()])
            print(f"{col}: corr={corr:.4f}, p={p_value:.4f}")
```

**解决方案**：
- 选择与SOC相关性高的静态特征
- 进行特征工程，创建更有意义的特征

#### 4.2 静态特征分布不合理
**症状**：静态特征分布偏斜或异常

**检查方法**：
```python
# 检查分布
static_df = pd.read_csv('dataset/CN-SOC-3500_new.csv')
for col in static_df.select_dtypes(include=[np.number]).columns:
    if col not in ['Point_ID', 'SOC']:
        print(f"\n{col}:")
        print(f"  均值: {static_df[col].mean():.4f}")
        print(f"  标准差: {static_df[col].std():.4f}")
        print(f"  最小值: {static_df[col].min():.4f}")
        print(f"  最大值: {static_df[col].max():.4f}")
        print(f"  缺失值: {static_df[col].isna().sum()}")
```

**解决方案**：
- 进行特征归一化
- 处理异常值
- 使用对数变换等

## 诊断步骤

### 步骤1：检查数据加载
```python
# 在train.py中添加
from diagnose_static_features import check_static_features_loading, check_point_id_matching

# 创建数据集后
check_static_features_loading(STATIC_CSV, train_ds)
check_point_id_matching(train_ds, num_samples=100)
```

### 步骤2：检查模型配置
```python
from diagnose_static_features import check_model_static_branch

# 创建模型后
check_model_static_branch(model)
```

### 步骤3：检查前向传播
```python
from diagnose_static_features import check_forward_pass

# 获取一个批次
sample_batch = next(iter(train_dl))
check_forward_pass(model, sample_batch[0])
```

### 步骤4：检查数据分布
```python
from diagnose_static_features import check_data_distribution

check_data_distribution(train_ds, num_samples=100)
```

### 步骤5：检查训练过程
在训练循环中添加：
```python
# 检查静态特征是否参与
if epoch % 10 == 0:
    # 获取一个批次
    sample_batch = next(iter(train_dl))
    inputs = sample_batch[0]
    
    # 检查静态特征
    if len(inputs) >= 3:
        static_feat = inputs[2]
        print(f"Epoch {epoch}: static_feat shape={static_feat.shape}, "
              f"mean={static_feat.mean().item():.4f}, "
              f"std={static_feat.std().item():.4f}, "
              f"is_all_zero={torch.allclose(static_feat, torch.zeros_like(static_feat))}")
    
    # 检查Alpha值
    if hasattr(model, 'fusion'):
        alphas = model.fusion.get_alpha_values()
        print(f"Epoch {epoch}: Alpha values: {alphas}")
```

## 常见问题排查清单

- [ ] Point_ID匹配率 > 90%
- [ ] 静态特征不全为0
- [ ] 静态特征方差 > 0.01（非常数特征）
- [ ] StaticBranch输出维度与SCMRL fusion static_dim匹配
- [ ] 前向传播中static_encoded不为None
- [ ] Alpha值在学习（远离初始值1.5）
- [ ] 静态特征分支有梯度
- [ ] 静态特征与SOC有相关性

## 快速修复建议

### 如果Point_ID匹配失败：
1. 检查数据加载器中的Point_ID匹配逻辑
2. 统一Point_ID格式（都转为字符串并去除".0"）

### 如果静态特征全为0：
1. 检查静态特征CSV文件
2. 检查数据加载逻辑
3. 验证特征列选择

### 如果Alpha值不学习：
1. 增加对齐损失权重：`--scmrl_lambda_align 0.2`
2. 检查学习率设置
3. 确保静态特征参与损失计算

### 如果静态特征维度不匹配：
1. 检查StaticBranch的hidden维度
2. 检查SCMRL fusion的static_dim参数
3. 确保两者一致

## 调试代码示例

在`train.py`的训练循环开始前添加：

```python
# 诊断静态特征
if USE_STATIC_FEATURES:
    print("\n" + "="*80)
    print("静态特征诊断")
    print("="*80)
    
    # 1. 检查数据加载
    from diagnose_static_features import (
        check_static_features_loading,
        check_model_static_branch,
        check_forward_pass,
        check_data_distribution,
        check_point_id_matching
    )
    
    check_static_features_loading(STATIC_CSV, train_ds)
    check_model_static_branch(model)
    check_data_distribution(train_ds, num_samples=100)
    check_point_id_matching(train_ds, num_samples=100)
    
    # 2. 检查一个批次
    sample_batch = next(iter(train_dl))
    check_forward_pass(model, sample_batch[0])
    
    print("="*80 + "\n")
```

