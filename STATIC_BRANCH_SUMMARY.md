# Static 分支实现总结文档

## 1. 概述

Static 分支是 SoilNet 模型中用于处理静态特征（如土壤质地、土地利用类型、地形特征等）的独立模块。这些特征不随时间变化，但与土壤有机碳（SOC）预测密切相关。

### 1.1 设计目标

- **补充动态特征**：影像和气候数据是动态的，静态特征提供稳定的背景信息
- **提升区域性能**：特别是对SC（南方）地区，静态特征（如土壤质地）对SOC影响更大
- **灵活集成**：支持可选的静态特征分支，可通过命令行开关控制

### 1.2 支持的静态特征类型

1. **数值特征**：
   - 土壤质地：`bulk_density`（容重）、`cec`（阳离子交换量）、`sand`、`silt`、`clay`（砂粒、粉粒、黏粒含量）
   - 地形特征：`twi`（地形湿度指数）、`tpi`（地形位置指数）
   - 其他土壤属性：pH、有机质等

2. **类别特征**：
   - `lulc` / `CLCD`：土地利用类型（通过 Embedding 编码）

## 2. 架构设计

### 2.1 整体架构

```
输入数据流：
┌─────────────────┐
│  影像数据 (CNN)  │ ──┐
└─────────────────┘   │
┌─────────────────┐   │
│ 气候数据 (LSTM)  │ ──┼──> 融合模块 ──> 回归头 ──> SOC预测
└─────────────────┘   │
┌─────────────────┐   │
│ 静态特征 (Static)│ ──┘
└─────────────────┘
```

### 2.2 核心组件

#### 2.2.1 StaticBranch（独立编码器）

**位置**：`soilnet/static_branch.py`

**功能**：
- 对数值特征进行线性编码
- 对类别特征（LULC）进行 Embedding 编码
- 输出统一维度的特征向量

**关键代码**：
```12:56:soilnet/static_branch.py
    def __init__(
        self,
        numeric_dim: int,
        lulc_classes: int = 0,
        lulc_embed_dim: int = 16,
        hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.numeric_dim = numeric_dim
        self.lulc_classes = int(lulc_classes) if lulc_classes is not None else 0
        self.lulc_embed_dim = int(lulc_embed_dim)

        # LULC embedding（可选）
        if self.lulc_classes > 0:
            self.lulc_embedding = nn.Embedding(self.lulc_classes, self.lulc_embed_dim)
        else:
            self.lulc_embedding = None

        in_dim = numeric_dim + (self.lulc_embed_dim if self.lulc_embedding is not None else 0)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden),
        )

    def forward(self, numeric_feats: torch.Tensor, lulc_idx: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            numeric_feats: [B, numeric_dim]
            lulc_idx:     [B]，若无则传 None
        Returns:
            encoded: [B, hidden]
        """
        feats = numeric_feats
        if self.lulc_embedding is not None and lulc_idx is not None:
            # 安全处理无效索引（<0）
            safe_idx = torch.clamp(lulc_idx, min=0)
            emb = self.lulc_embedding(safe_idx)
            invalid_mask = (lulc_idx < 0).unsqueeze(-1).to(emb.dtype)
            emb = emb * (1.0 - invalid_mask)
            feats = torch.cat([feats, emb], dim=-1)
        encoded = self.encoder(feats)
        return encoded
```

**注意**：当前实现中，`StaticBranch` 类**已在 `SoilNetLSTMWithStatic` 中使用**。模型使用 `StaticBranch` 对静态特征进行编码，包括数值特征和 LULC Embedding。

#### 2.2.2 数据集加载器

**位置**：`dataset/dataset_loader_china_static.py`

**类名**：`ChinaSNDatasetClimateStatic`

**功能**：
- 继承自 `ChinaSNDatasetClimate`
- 从CSV文件加载静态特征
- 自动识别数值特征和类别特征
- 处理缺失值和特征编码

**关键实现**：
```47:121:dataset/dataset_loader_china_static.py
        if static_csv_path and os.path.exists(static_csv_path):
            print(f"加载静态特征: {static_csv_path}")
            static_df = pd.read_csv(static_csv_path)
            
            # 识别Point_ID列
            pid_col = None
            for col in static_df.columns:
                if col.lower() in ['point_id', 'pointid', 'pid']:
                    pid_col = col
                    break
            
            if pid_col is None:
                raise ValueError(f"静态特征CSV中未找到Point_ID列: {static_csv_path}")
            
            # 识别特征列（排除Point_ID、Latitude、Longitude等）
            exclude_cols = {pid_col, 'Latitude', 'Longitude', 'SOC', 'Year'}
            self.static_feature_cols = [col for col in static_df.columns 
                                       if col not in exclude_cols]
            
            if len(self.static_feature_cols) == 0:
                raise ValueError(f"静态特征CSV中未找到特征列: {static_csv_path}")
            
            # 将非数值型（如LULC/CLCD）保留为"索引"以供Embedding；数值列做均值填充
            self.static_category_maps = {}
            feats_df_raw = static_df[self.static_feature_cols].copy()
            categorical_cols = []
            numeric_cols = []
            for col in feats_df_raw.columns:
                if feats_df_raw[col].dtype == object or str(feats_df_raw[col].dtype).startswith('category'):
                    categorical_cols.append(col)
                else:
                    numeric_cols.append(col)

            # 只支持单一主类别列作为Embedding（如CLCD/LULC），如有多个，取第一个
            self.lulc_col = None
            if len(categorical_cols) > 0:
                self.lulc_col = categorical_cols[0]
            
            # 数值列：转换为数值并缺失填充为均值
            feats_num = pd.DataFrame(index=feats_df_raw.index)
            if len(numeric_cols) > 0:
                feats_num = feats_df_raw[numeric_cols].apply(pd.to_numeric, errors='coerce')
                for col in numeric_cols:
                    if feats_num[col].isna().any():
                        feats_num[col] = feats_num[col].fillna(feats_num[col].mean())

            # 保存数值列次序，用于模型输入维度
            self.static_numeric_cols = list(feats_num.columns)

            # 类别列：保存索引映射（用于Embedding），仅对选定的lulc_col生效
            self.lulc_index_map = {}
            self.num_lulc_classes = 0
            if self.lulc_col is not None:
                series = feats_df_raw[self.lulc_col].astype(str).str.strip()
                codes, uniques = pd.factorize(series)
                self.lulc_index_map = {str(v): int(i) for i, v in enumerate(uniques)}
                self.num_lulc_classes = len(uniques)
                lulc_codes = codes.astype(np.int64)
            else:
                # 如果没有类别列，使用-1占位
                lulc_codes = np.full((feats_df_raw.shape[0],), -1, dtype=np.int64)

            # 构建查找表
            self.static_numeric_features = {}
            self.lulc_indices = {}
            numeric_values = feats_num[self.static_numeric_cols].values.astype(np.float32) if len(self.static_numeric_cols) > 0 else np.zeros((feats_df_raw.shape[0], 0), dtype=np.float32)
            for idx, row in static_df.iterrows():
                point_id = str(row[pid_col]).replace('.0', '').strip()
                self.static_numeric_features[point_id] = numeric_values[idx]
                self.lulc_indices[point_id] = int(lulc_codes[idx])
            
            print(f"  静态数值特征维度: {len(self.static_numeric_cols)}")
            if self.lulc_col is not None:
                print(f"  类别特征（用于Embedding）: {self.lulc_col}（类别数={self.num_lulc_classes}）")
            print(f"  加载的静态特征样本数: {len(self.static_numeric_features)}")
```

**数据返回格式**：
```126:179:dataset/dataset_loader_china_static.py
    def __getitem__(self, index):
        """
        返回: (影像, 气候数据, 静态数值特征, LULC索引[可选]), SOC, (可选: Point_ID)
        """
        # 获取point_id（用于查找静态特征）
        l8_img_name = self.l8_names[index]
        point_id = l8_img_name.split('_')[0]
        
        # 获取基础数据（影像、气候、SOC）
        result = super().__getitem__(index)
        if self.return_point_id:
            (l8_img, clim_arr), socd, _ = result
        else:
            (l8_img, clim_arr), socd = result
        
        # 获取静态数值特征与LULC索引
        static_numeric = None
        lulc_idx = None
        if getattr(self, 'static_numeric_features', None) is not None:
            # 尝试不同的Point_ID格式匹配
            pid_key = str(point_id).replace('.0', '').strip()
            if pid_key in self.static_numeric_features:
                static_numeric = self.static_numeric_features[pid_key]
                lulc_idx = self.lulc_indices.get(pid_key, -1)
            else:
                # 尝试整数匹配
                try:
                    pid_int = int(float(point_id))
                    pid_key_int = str(pid_int)
                    if pid_key_int in self.static_numeric_features:
                        static_numeric = self.static_numeric_features[pid_key_int]
                        lulc_idx = self.lulc_indices.get(pid_key_int, -1)
                except:
                    pass
            
            if static_numeric is None:
                static_numeric = np.zeros(len(getattr(self, 'static_numeric_cols', [])), dtype=np.float32)
                lulc_idx = -1
            
            static_numeric = torch.from_numpy(static_numeric).to(self.clim_dtype)
            lulc_idx = torch.tensor(lulc_idx, dtype=torch.long)
        
        # 返回结果
        if static_numeric is not None:
            if self.return_point_id:
                return (l8_img, clim_arr, static_numeric, lulc_idx), socd, point_id
            else:
                return (l8_img, clim_arr, static_numeric, lulc_idx), socd
        else:
            # 如果没有静态特征，返回原有格式
            if self.return_point_id:
                return (l8_img, clim_arr), socd, point_id
            else:
                return (l8_img, clim_arr), socd
```

#### 2.2.3 模型集成

**位置**：`soilnet/soil_net_static.py`

**类名**：`SoilNetLSTMWithStatic`

**继承关系**：`SoilNetLSTMWithStatic` → `SoilNetLSTM` → `nn.Module`

**关键实现**：
```20:95:soilnet/soil_net_static.py
    def __init__(self, static_feature_dim=0, lulc_num_classes: int = 0, lulc_embed_dim: int = 16, **kwargs):
        """
        Args:
            static_feature_dim: 静态特征维度（土地利用、土壤质地等特征的数量）
            **kwargs: 传递给父类的其他参数
        """
        # 移除static_feature_dim，避免传递给父类
        self.static_feature_dim = static_feature_dim
        self.lulc_num_classes = int(lulc_num_classes) if lulc_num_classes is not None else 0
        self.lulc_embed_dim = int(lulc_embed_dim)
        
        # 如果使用SCMRL fusion，需要传入编码后的静态特征维度（与lstm_dim相同）
        # 因为StaticBranch会将静态特征编码为lstm_dim维度
        use_scmrl = kwargs.get('use_scmrl_fusion', False)
        lstm_dim = kwargs.get('lstm_out', 128)
        if use_scmrl and static_feature_dim > 0:
            # 编码后的静态特征维度 = lstm_dim
            kwargs['static_dim'] = lstm_dim
        
        super().__init__(**kwargs)
        
        # 重新定义回归器以支持静态特征（使用StaticBranch编码）
        # 如果static_feature_dim > 0，添加静态特征分支
        if static_feature_dim > 0:
            # 获取回归器输入维度
            cnn_dim = kwargs.get('regresor_input_from_cnn', 1024)
            lstm_dim = kwargs.get('lstm_out', 128)
            hidden_size = kwargs.get('hidden_size', 128)
            reg_version = kwargs.get('reg_version', 1)
            dropout_prob = kwargs.get('dropout_prob', 0.5)
            static_dropout = kwargs.get('static_dropout', 0.3)
            
            # 创建StaticBranch：数值特征 + LULC Embedding -> 固定维度编码
            # hidden维度与lstm_dim保持一致，便于融合
            static_hidden_dim = lstm_dim  # 使用与LSTM相同的维度
            self.static_branch = StaticBranch(
                numeric_dim=static_feature_dim,
                lulc_classes=self.lulc_num_classes,
                lulc_embed_dim=self.lulc_embed_dim,
                hidden=static_hidden_dim,
                dropout=static_dropout
            )
            
            # 检查是否使用SCMRL fusion（父类已经初始化了use_scmrl_fusion属性）
            use_scmrl = getattr(self, 'use_scmrl_fusion', False)
            
            if use_scmrl:
                # 如果使用SCMRL fusion，回归器接收融合后的单一特征向量（维度为lstm_out）
                # StaticBranch输出维度为static_hidden_dim（=lstm_dim），可以直接参与融合
                reg_input_dim = lstm_dim
                if self.use_regional_adaptation and getattr(self, "region_embedding", None) is not None:
                    reg_input_dim = lstm_dim + self.region_embed_dim
                self.reg = nn.Linear(reg_input_dim, 1)
                print(f"[OK] 已启用静态特征支持（使用StaticBranch编码，参与SCMRL融合）")
                print(f"    静态数值特征维度: {static_feature_dim}, LULC类别数: {self.lulc_num_classes}, 编码后维度: {static_hidden_dim}")
            else:
                # 如果不使用SCMRL fusion，使用MultiHeadRegressor接收多个输入
                # StaticBranch输出维度为static_hidden_dim
                reg_input_dims = [cnn_dim, lstm_dim, static_hidden_dim]
                if self.use_regional_adaptation and getattr(self, "region_embedding", None) is not None:
                    reg_input_dims.append(self.region_embed_dim)
                self.reg = MultiHeadRegressor(
                    *reg_input_dims,
                    hidden_size=hidden_size,
                    activation="sigmoid",
                    version=reg_version,
                    dropout_prob=dropout_prob
                )
                print(f"[OK] 已启用静态特征支持（使用StaticBranch编码，直接拼接）")
                print(f"    静态数值特征维度: {static_feature_dim}, LULC类别数: {self.lulc_num_classes}, 编码后维度: {static_hidden_dim}")
```

**前向传播**：
```78:142:soilnet/soil_net_static.py
    def forward(self, input_raster_ts_static: Tuple, region_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with static features
        
        Args:
            input_raster_ts_static: Tuple containing:
                - raster_stack: 影像数据 [B, C, H, W]
                - ts_features: 气候时间序列 [B, T, F]
                - static_features: 静态特征 [B, S] (可选)
        
        Returns:
            output: 预测的SOC值 [B, 1]
        """
        # 解包输入（兼容：两路、三路（旧：直接静态向量）、四路（新：数值静态+LULC索引））
        raster_stack = input_raster_ts_static[0]
        ts_features = input_raster_ts_static[1]
        static_features = None
        lulc_indices = None
        if len(input_raster_ts_static) >= 3:
            static_features = input_raster_ts_static[2]
        if len(input_raster_ts_static) >= 4:
            lulc_indices = input_raster_ts_static[3]
        
        # CNN特征提取
        if self.use_spectral_enhance:
            raster_stack = self.spectral(raster_stack)
        cnn_features = self.cnn(raster_stack)  # [B, cnn_dim]
        
        # 气候特征提取
        climate_features = self.lstm(ts_features)  # [B, lstm_dim]
        
        # 保存中间特征（用于对齐损失计算，如果使用SCMRL）
        if hasattr(self, 'use_scmrl_fusion') and self.use_scmrl_fusion:
            self._last_climate_feat = climate_features
            self._last_visual_feat = cnn_features
        
        # S-CMRL 融合或原有融合方式
        if hasattr(self, 'use_scmrl_fusion') and self.use_scmrl_fusion:
            # 使用 S-CMRL 融合（静态特征会参与融合）
            fused_feat = self.fusion(climate_features, cnn_features, static_features)
            
            # 如果有区域自适应，需要额外处理
            if self.use_regional_adaptation and getattr(self, "region_embedding", None) is not None and region_ids is not None:
                region_ids = region_ids.to(fused_feat.device).long()
                region_features = self.region_embedding(region_ids)
                fused_feat = torch.cat([fused_feat, region_features], dim=1)
            
            output = self.reg(fused_feat)
        else:
            # 旧版本：直接将静态特征送入回归头（不使用SCMRL fusion）
            if (static_features is not None) and self.static_feature_dim > 0:
                # 旧版本：直接将静态向量送入回归头，不经过StaticBranch编码
                reg_inputs = [cnn_features, climate_features, static_features]
            else:
                # 两输入回归器（原有逻辑）
                reg_inputs = [cnn_features, climate_features]

            if self.use_regional_adaptation and getattr(self, "region_embedding", None) is not None and region_ids is not None:
                region_ids = region_ids.to(cnn_features.device).long()
                region_features = self.region_embedding(region_ids)
                reg_inputs.append(region_features)

            output = self.reg(*reg_inputs)
        
        return output
```

## 3. 融合机制

### 3.1 两种融合方式

#### 方式1：直接拼接（旧版本，默认）

**特点**：
- 静态特征直接送入 `MultiHeadRegressor`
- 不经过 `StaticBranch` 编码
- 与 CNN 和 LSTM 特征并行输入回归头

**流程**：
```
CNN特征 [B, 1024] ──┐
LSTM特征 [B, 128]  ──┼──> MultiHeadRegressor ──> SOC预测
静态特征 [B, S]    ──┘
```

#### 方式2：SCMRL 融合（新版本，可选）

**特点**：
- 使用 `SemanticAlignedFusion` 模块
- 静态特征参与跨模态残差学习
- 通过可学习的 alpha 参数控制贡献

**融合流程**：
```202:282:soilnet/submodules/semantic_aligned_fusion.py
class SemanticAlignedFusion(nn.Module):
    """
    语义对齐跨模态残差融合模块
    
    输入三个模态的特征：
    - Climate（强模态）：性能最好
    - Visual（弱模态）：可能包含噪声
    - Static（弱模态）：可能包含噪声
    
    输出：融合后的特征，形状与 Climate 特征保持一致
    
    融合策略：
    1. 先融合 Climate + Visual（通过 CrossModalResidualBlock）
    2. 再融合结果 + Static（通过另一个 CrossModalResidualBlock）
    3. 最终输出与 Climate 维度相同
    """
    
    def __init__(
        self,
        climate_dim: int,
        visual_dim: int,
        static_dim: Optional[int] = None,
        num_heads: int = 8,
        alpha_init: float = 1.5,
        learnable_alpha: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.climate_dim = climate_dim
        self.visual_dim = visual_dim
        self.static_dim = static_dim
        
        # Climate + Visual 融合
        self.climate_visual_fusion = CrossModalResidualBlock(
            climate_dim=climate_dim,
            weak_dim=visual_dim,
            num_heads=num_heads,
            alpha_init=alpha_init,
            learnable_alpha=learnable_alpha,
            dropout=dropout
        )
        
        # Climate-Visual 融合结果 + Static 融合
        if static_dim is not None:
            self.climate_static_fusion = CrossModalResidualBlock(
                climate_dim=climate_dim,
                weak_dim=static_dim,
                num_heads=num_heads,
                alpha_init=alpha_init,
                learnable_alpha=learnable_alpha,
                dropout=dropout
            )
        else:
            self.climate_static_fusion = None
    
    def forward(
        self,
        climate_feat: torch.Tensor,
        visual_feat: torch.Tensor,
        static_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            climate_feat: Climate 特征 [B, climate_dim] 或 [B, seq_len, climate_dim]
            visual_feat: Visual 特征 [B, visual_dim] 或 [B, seq_len, visual_dim]
            static_feat: Static 特征 [B, static_dim]（可选）
        
        Returns:
            fused_feat: 融合后的特征，形状与 climate_feat 相同
        """
        # Step 1: Climate + Visual 融合
        fused = self.climate_visual_fusion(climate_feat, visual_feat)
        
        # Step 2: 如果提供了 Static 特征，继续融合
        if static_feat is not None and self.climate_static_fusion is not None:
            fused = self.climate_static_fusion(fused, static_feat)
        
        return fused
```

**SCMRL 融合的优势**：
- **防止弱模态噪声**：如果静态特征是噪声，alpha 参数会自动变小
- **语义对齐**：通过对比学习拉近同一样本的不同模态特征
- **残差连接**：保证强模态（Climate）的主导地位

## 4. 数据预处理

### 4.1 预处理脚本

**位置**：`prepare_static_features.py`

**功能**：
- 缺失值处理（KNN填充）
- 数值特征归一化（Min-Max Scaling）
- 类别特征编码（Label Encoding）

**使用示例**：
```bash
python prepare_static_features.py dataset/CN-SOC-3500_new.csv
```

**输出**：
- `dataset/CN-SOC-3500_new_processed.csv`：处理后的CSV文件
- `dataset/CN-SOC-3500_new_processed_metadata.pkl`：元数据（编码器、缩放器等）

### 4.2 CSV 格式要求

**必需列**：
- `Point_ID`：点位ID（用于匹配）

**可选列**：
- `Latitude`、`Longitude`：地理坐标
- `SOC`：标签（如果存在会被排除）
- 其他静态特征列：如 `bulk_density`、`cec`、`sand`、`silt`、`clay`、`twi`、`tpi`、`lulc` 等

## 5. 训练配置

### 5.1 命令行参数

**启用静态特征**：
```bash
python train.py \
    -e experiment_name \
    -static \                                    # 启用静态特征
    --static_csv dataset/CN-SOC-3500_new.csv \  # 静态特征CSV路径
    -lstm \                                      # 必须启用LSTM分支
    -cnn ViT-CoMer \
    -rnn Transformer \
    -ne 60 \
    -seed 1
```

**禁用静态特征**（默认）：
```bash
python train.py \
    -e experiment_name \
    -lstm \
    -cnn ViT-CoMer \
    -rnn Transformer \
    -ne 60 \
    -seed 1
```

### 5.2 配置参数

**位置**：`train.py`

**关键代码**：
```180:199:train.py
	USE_STATIC_FEATURES = args.use_static  # 静态特征开关
	
	# 静态特征CSV路径：优先命令行，其次 config.static_csv_path，最后使用默认路径
	if USE_STATIC_FEATURES:
		if args.static_csv and args.static_csv not in [None, ""]:
			STATIC_CSV = args.static_csv
		elif hasattr(config, 'static_csv_path') and config.static_csv_path:
			STATIC_CSV = config.static_csv_path
		else:
			# 默认使用主CSV文件（通常静态特征和主数据在同一个CSV中）
			STATIC_CSV = lucas_csv_path
			print(f"\033[93m[静态特征] 未指定 --static_csv，使用默认路径: {STATIC_CSV}\033[0m")
		
		# 验证文件是否存在
		if not os.path.exists(STATIC_CSV):
			print(f"\033[91m错误: 静态特征CSV文件不存在: {STATIC_CSV}\033[0m")
			raise FileNotFoundError(f"Static CSV file not found: {STATIC_CSV}")
	else:
		STATIC_CSV = None
```

**数据集初始化**：
```309:333:train.py
		if USE_STATIC_FEATURES and ('HAS_STATIC_DS' in locals()) and HAS_STATIC_DS and (STATIC_CSV is not None) and (STATIC_CSV != '') and os.path.exists(STATIC_CSV):
			train_ds = ChinaSNDatasetClimateStatic(train_l8_folder_path,
												lucas_csv_path,
												climate_csv_folder_path,
												static_csv_path=STATIC_CSV,
												l8_bands=bands, transform=train_transform)

			test_ds = ChinaSNDatasetClimateStatic(test_l8_folder_path,
											lucas_csv_path,
											climate_csv_folder_path,
											static_csv_path=STATIC_CSV,
											l8_bands=bands, transform=test_transform)
			
			val_ds = ChinaSNDatasetClimateStatic(val_l8_folder_path,
											lucas_csv_path,
											climate_csv_folder_path,
											static_csv_path=STATIC_CSV,
											l8_bands=bands, transform=test_transform)
			
			test_ds_w_id = ChinaSNDatasetClimateStatic(test_l8_folder_path,
											lucas_csv_path,
											climate_csv_folder_path,
											static_csv_path=STATIC_CSV,
											l8_bands=bands, transform=test_transform, return_point_id=True)
			USING_STATIC_FEATURES = True
```

## 6. 关键文件清单

| 文件路径 | 功能 | 说明 |
|---------|------|------|
| `soilnet/static_branch.py` | 静态特征编码器 | 定义 `StaticBranch` 类（已在模型中使用） |
| `dataset/dataset_loader_china_static.py` | 静态特征数据加载器 | `ChinaSNDatasetClimateStatic` 类 |
| `soilnet/soil_net_static.py` | 支持静态特征的模型 | `SoilNetLSTMWithStatic` 类 |
| `prepare_static_features.py` | 数据预处理脚本 | 缺失值填充、归一化、编码 |
| `train.py` | 训练脚本 | 静态特征开关和数据集初始化 |
| `soilnet/submodules/semantic_aligned_fusion.py` | SCMRL融合模块 | 支持静态特征的跨模态融合 |

## 7. 当前实现状态

### 7.1 已实现功能

✅ **数据加载**：
- 支持从CSV加载静态特征
- 自动识别数值特征和类别特征
- 处理缺失值和特征编码

✅ **模型集成**：
- 支持静态特征输入
- 两种融合方式（直接拼接 / SCMRL融合）
- 兼容区域自适应

✅ **训练流程**：
- 命令行开关控制
- 自动数据集选择
- 结果记录

### 7.2 已启用功能

✅ **StaticBranch 类**：
- 已在模型中使用，对静态特征进行编码
- 数值特征直接输入，LULC类别通过Embedding编码
- 拼接后通过MLP编码为固定维度（与lstm_dim相同）

### 7.3 限制

- **仅支持 CHINA 数据集**：当前实现仅支持中国数据集
- **必须启用 LSTM 分支**：静态特征只在气候数据分支中可用
- **单一类别特征**：仅支持一个类别列（LULC/CLCD）进行 Embedding

## 8. 使用建议

### 8.1 数据准备

1. **准备CSV文件**：
   - 包含 `Point_ID` 列
   - 包含静态特征列（数值或类别）
   - 确保 Point_ID 与主数据集匹配

2. **运行预处理**（可选）：
   ```bash
   python prepare_static_features.py dataset/your_static_features.csv
   ```

3. **验证数据**：
   - 检查缺失值比例
   - 确认特征维度合理

### 8.2 训练实验

**对比实验**：
- **基线**：不使用静态特征
- **实验1**：使用静态特征（直接拼接）
- **实验2**：使用静态特征 + SCMRL融合

**评估指标**：
- 整体性能（R²、RMSE）
- 区域性能（特别是SC地区）
- Alpha 参数值（如果使用SCMRL）

### 8.3 调试技巧

1. **检查特征加载**：
   - 查看数据集初始化时的打印信息
   - 确认静态特征维度正确

2. **监控 Alpha 值**（SCMRL）：
   - 如果 `alpha` 接近 0，说明静态特征可能是噪声
   - 如果 `alpha` 较大，说明静态特征有贡献

3. **验证数据匹配**：
   - 检查 Point_ID 匹配率
   - 确认没有大量样本缺失静态特征

## 9. 实现状态更新

### 9.1 LULC Embedding 已实现

✅ **当前实现**（2025-01-XX）：
- ✅ 使用 `StaticBranch` 进行特征编码
- ✅ LULC 类别通过 Embedding 编码（维度：16）
- ✅ 数值特征与 LULC Embedding 拼接
- ✅ 通过 MLP 编码为固定维度（与 LSTM 输出维度相同：128）
- ✅ 支持 SCMRL 融合和直接拼接两种方式

**实现流程**：
1. 数据加载：从 CSV 读取数值特征和 LULC 类别索引
2. 特征编码：
   - 数值特征直接输入 `StaticBranch`
   - LULC 索引通过 `nn.Embedding` 编码为 16 维向量
   - 拼接数值特征和 LULC Embedding
   - 通过 MLP（Linear → ReLU → Dropout → Linear）编码为 128 维
3. 模型融合：编码后的静态特征参与 SCMRL 融合或直接拼接

## 10. 未来改进方向

1. **多类别特征支持**：
   - 支持多个类别特征列
   - 每个类别特征独立 Embedding

3. **特征重要性分析**：
   - 分析哪些静态特征对SOC预测最重要
   - 进行特征选择

4. **跨数据集支持**：
   - 扩展到 LUCAS、RaCA 数据集
   - 统一静态特征接口

---

**文档版本**：v1.0  
**最后更新**：2025-01-XX  
**维护者**：SoilNet 团队

