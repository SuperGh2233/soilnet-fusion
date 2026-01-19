# 静态特征集成方案

## 概述
针对SC地区SOC受土壤与地形因素影响更大的特点，建议添加静态特征：
- **土地利用类型** (Land Use Type)
- **土壤质地**：
  - 黏粒含量 (Clay Content)
  - CEC阳离子交换量 (Cation Exchange Capacity)
  - 其他土壤属性（可选：有机质、pH、容重等）

## 实施步骤

### 1. 准备静态特征数据

创建一个CSV文件，包含：
- `Point_ID`: 与现有数据集一致的点位ID
- `LandUse`: 土地利用类型（编码为数值，如1=农田，2=森林等）
- `Clay_Content`: 黏粒含量（%）
- `CEC`: 阳离子交换量（cmol/kg）
- 其他字段...

### 2. 数据预处理
运行 `prepare_static_features.py` 进行归一化和缺失值填充

### 3. 修改数据加载器
已创建 `dataset_loader_china_static.py` 支持静态特征

### 4. 修改模型架构
使用 `SoilNetLSTMWithStatic` 类（支持静态特征输入）

### 5. 训练
在训练命令中添加 `-static` 参数以启用静态特征

