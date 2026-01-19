# 静态特征未归一化问题 - 修复方案

## 🔴 核心问题

**静态特征在数据加载时没有进行归一化，而气候特征有归一化！**

这导致：
1. **特征尺度不匹配**：静态特征的尺度可能与CNN/LSTM特征差异很大
2. **梯度不稳定**：不同尺度的特征导致梯度爆炸或消失
3. **Alpha值无法学习**：在SCMRL fusion中，静态特征作为弱模态，如果尺度不匹配，注意力机制无法有效工作
4. **模型难以学习**：神经网络对特征尺度非常敏感

## ✅ 为什么RF/XGBoost有效？

- **树模型对尺度不敏感**：RF和XGBoost基于决策树，只关心特征的分割点，不关心绝对数值
- **神经网络对尺度敏感**：需要归一化来保证梯度稳定和训练收敛

## 🔧 修复方案

### 方案1：在数据加载器中添加归一化（推荐）

修改 `dataset/dataset_loader_china_static.py`，在加载静态特征时进行归一化：

```python
from sklearn.preprocessing import StandardScaler

# 在 __init__ 方法中，加载静态特征后添加：
if static_csv_path and os.path.exists(static_csv_path):
    # ... 现有代码 ...
    
    # 归一化数值特征
    if len(numeric_cols) > 0:
        # 使用StandardScaler进行标准化（均值0，方差1）
        self.static_scaler = StandardScaler()
        numeric_values = feats_num[self.static_numeric_cols].values.astype(np.float32)
        numeric_values_normalized = self.static_scaler.fit_transform(numeric_values)
        
        # 更新查找表
        for idx, row in static_df.iterrows():
            point_id = str(row[pid_col]).replace('.0', '').strip()
            self.static_numeric_features[point_id] = numeric_values_normalized[idx]
```

### 方案2：在StaticBranch中添加BatchNorm（备选）

如果不想修改数据加载器，可以在`StaticBranch`中添加BatchNorm：

```python
class StaticBranch(nn.Module):
    def __init__(self, ...):
        # ... 现有代码 ...
        
        # 添加BatchNorm来归一化输入
        if numeric_dim > 0:
            self.input_norm = nn.BatchNorm1d(numeric_dim)
        else:
            self.input_norm = None
        
        # ... 现有代码 ...
    
    def forward(self, numeric_feats, lulc_idx=None):
        # 归一化输入
        if self.input_norm is not None:
            numeric_feats = self.input_norm(numeric_feats)
        
        # ... 现有代码 ...
```

### 方案3：预处理CSV文件（一次性修复）

使用 `prepare_static_features.py` 脚本预处理静态特征CSV：

```bash
python prepare_static_features.py \
    --input dataset/CN-SOC-3500_new.csv \
    --output dataset/CN-SOC-3500_new_normalized.csv
```

然后在训练时使用归一化后的CSV文件。

## 📊 验证修复效果

修复后，检查：

1. **特征统计**：
```python
# 检查归一化后的特征
for i in range(min(10, len(train_ds))):
    sample = train_ds[i]
    static_feat = sample[0][2]  # 静态特征
    print(f"样本{i}: mean={static_feat.mean():.4f}, std={static_feat.std():.4f}, "
          f"min={static_feat.min():.4f}, max={static_feat.max():.4f}")
```

2. **Alpha值变化**：
   - 修复前：Alpha值接近初始值1.5，不变化
   - 修复后：Alpha值应该会学习调整

3. **训练损失**：
   - 修复前：损失可能不稳定或下降缓慢
   - 修复后：损失应该更稳定，下降更快

## 🎯 其他可能的问题

即使修复了归一化，如果仍然无效，检查：

1. **对齐损失权重**：`--scmrl_lambda_align` 可能太小，尝试增加到0.2-0.5
2. **学习率**：静态特征分支可能需要更大的学习率
3. **特征相关性**：检查静态特征与SOC的相关性
4. **数据匹配**：确保Point_ID匹配率>90%

## 📝 实施步骤

1. **立即修复**：在数据加载器中添加归一化（方案1）
2. **重新训练**：使用归一化后的特征重新训练
3. **监控指标**：
   - Alpha值是否变化
   - 训练损失是否改善
   - 验证集性能是否提升
4. **对比实验**：对比修复前后的性能

