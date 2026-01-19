# 代码补丁 - 可直接复制粘贴

本文件包含所有需要修改的代码的完整补丁，可以直接复制粘贴。

---

## 补丁 1: dataset/dataset_loader.py

### 位置 1: 修改 `myNormalize.__init__` 方法

**查找:**
```python
class myNormalize:
  """Normalize the image and the target value"""
  def __init__(self, img_bands_min_max =[[(0,7),(0,1)], [(7,12),(-1,1)], [(12), (-4,2963)], [(13), (0, 90)]], oc_min = 0, oc_max = 200):
    """
      A class to normalize image and target value arrays.
      
      Args:
      - `img_bands_min_max` (list): A list of tuples defining the bands to normalize and the corresponding minimum and maximum values.
            * Default is `[(0,7),(0,1)], [(7,12),(-1,1), [(12), (-4,2963)],[(13), (0, 90)]]`, where the first 7 bands are Landsat SR bands and the rest are indices. band 12 is SRTM and band 13 is slope
      
             `[(from_band, to_band),(min_of_bands , max_of_bands)]`
      - `oc_min` (int or float): The minimum value of the target array. Default is 0.
      - `oc_max` (int or float): The maximum value of the target array. Default is 1000.
      
      Returns:
      - A tuple containing the normalized image and target value arrays.
    """
    self.img_bands_min_max = img_bands_min_max
    self.oc_min = oc_min
    self.oc_max = oc_max
```

**替换为:**
```python
class myNormalize:
  """Normalize the image and the target value"""
  def __init__(self, img_bands_min_max =[[(0,7),(0,1)], [(7,12),(-1,1)], [(12), (-4,2963)], [(13), (0, 90)]], oc_min = 0, oc_max = 200, normalize_oc = True, clip_oc = True):
    """
      A class to normalize image and target value arrays.
      
      Args:
      - `img_bands_min_max` (list): A list of tuples defining the bands to normalize and the corresponding minimum and maximum values.
            * Default is `[(0,7),(0,1)], [(7,12),(-1,1), [(12), (-4,2963)],[(13), (0, 90)]]`, where the first 7 bands are Landsat SR bands and the rest are indices. band 12 is SRTM and band 13 is slope
      
             `[(from_band, to_band),(min_of_bands , max_of_bands)]`
      - `oc_min` (int or float): The minimum value of the target array. Default is 0.
      - `oc_max` (int or float): The maximum value of the target array. Default is 1000.
      - `normalize_oc` (bool): Whether to normalize the OC value to [0,1]. Default is True.
      - `clip_oc` (bool): Whether to clip the OC value to [0,1] after normalization. Default is True.
      
      Returns:
      - A tuple containing the normalized image and target value arrays.
    """
    self.img_bands_min_max = img_bands_min_max
    self.oc_min = oc_min
    self.oc_max = oc_max
    self.normalize_oc = normalize_oc
    self.clip_oc = clip_oc
```

### 位置 2: 修改 `myNormalize.__call__` 方法中的标签归一化部分

**查找:**
```python
    # Normalize the target value (0,1)
    oc = normalize(oc, self.oc_min, self.oc_max)

    # Cutting out of range Vlaues
    img[img > 1] = 1
    img[img < 0] = 0

    # Modify data based the normalization process (no need for log transformation)
    oc = oc if oc < 1 else 1
    oc = oc if oc > 0 else 0

    # # log transformation instead of normalization 
    # oc = log_transform(oc)

    return img, oc
```

**替换为:**
```python
    # Normalize the target value (0,1) - only if normalize_oc is True
    if self.normalize_oc:
        oc = normalize(oc, self.oc_min, self.oc_max)
        # Clip OC value only if clip_oc is True
        if self.clip_oc:
            oc = oc if oc < 1 else 1
            oc = oc if oc > 0 else 0
    # else: keep oc in raw scale (no normalization, no clipping)

    # Cutting out of range Values for image bands
    img[img > 1] = 1
    img[img < 0] = 0

    # # log transformation instead of normalization 
    # oc = log_transform(oc)

    return img, oc
```

---

## 补丁 2: train.py

### 位置 1: 在 `parse_arguments()` 函数末尾添加参数

**查找:**
```python
	# 区域自适应参数
	parser.add_argument('-ra', '--use_regional_adaptation', action='store_true', default=False, help='Use regional adaptation')
	parser.add_argument('-nr', '--num_regions', type=int, default=6, help='Number of regions (6 for China natural regions)')
	parser.add_argument('-rs', '--region_strategy', type=str, default='geographic', choices=['grid', 'geographic'], help='Region assignment strategy')

	args = parser.parse_args()
	return args
```

**替换为:**
```python
	# 区域自适应参数
	parser.add_argument('-ra', '--use_regional_adaptation', action='store_true', default=False, help='Use regional adaptation')
	parser.add_argument('-nr', '--num_regions', type=int, default=6, help='Number of regions (6 for China natural regions)')
	parser.add_argument('-rs', '--region_strategy', type=str, default='geographic', choices=['grid', 'geographic'], help='Region assignment strategy')
	
	# 标签策略开关参数（长尾SOC回归消融实验）
	parser.add_argument('--label_mode', type=str, default=None, 
						choices=['baseline_raw_mse', 'log1p_mse', 'log1p_huber', 'log1p_huber_w'],
						help='Label strategy mode for long-tail SOC regression ablation experiments')
	parser.add_argument('--huber_beta', type=float, default=1.0, 
						help='Beta parameter for Huber loss (SmoothL1Loss)')
	parser.add_argument('--tail_threshold', type=float, default=30.0, 
						help='Threshold for high-value samples in weighted loss')
	parser.add_argument('--tail_weight', type=float, default=2.0, 
						help='Weight for high-value samples (SOC > tail_threshold)')

	args = parser.parse_args()
	return args
```

### 位置 2: 修改 myNormalize 构建部分

**查找:**
```python
	if USE_SRTM:
		mynorm = myNormalize(img_bands_min_max =[[(0,7),(0,1)], [(7,12),(-1,1)], [(12), (-4,2963)], [(13), (0, 90)]], oc_min = 0, oc_max = OC_MAX)
	else:
		mynorm = myNormalize(img_bands_min_max =[[(0,7),(0,1)], [(7,12),(-1,1)]], oc_min = 0, oc_max = OC_MAX)
```

**替换为:**
```python
	# 根据 label_mode 决定是否归一化标签
	# 如果使用四组消融实验之一，则不归一化标签（保持原尺度）
	LABEL_MODE = args.label_mode
	HUBER_BETA = args.huber_beta
	TAIL_THRESHOLD = args.tail_threshold
	TAIL_WEIGHT = args.tail_weight
	
	if LABEL_MODE in ['baseline_raw_mse', 'log1p_mse', 'log1p_huber', 'log1p_huber_w']:
		# 四组消融实验：标签保持原尺度，不归一化，不clip
		normalize_oc_flag = False
		clip_oc_flag = False
		print(f"\n{'='*80}")
		print(f"[标签策略] 模式: {LABEL_MODE}")
		print(f"[标签策略] 标签将保持原尺度 SOC (不归一化, 不clip)")
		if LABEL_MODE.startswith('log1p'):
			print(f"[标签策略] 训练目标: log1p(SOC)")
		else:
			print(f"[标签策略] 训练目标: 原尺度 SOC")
		if 'huber' in LABEL_MODE:
			print(f"[标签策略] 损失函数: Huber (beta={HUBER_BETA})")
		else:
			print(f"[标签策略] 损失函数: MSE")
		if LABEL_MODE == 'log1p_huber_w':
			print(f"[标签策略] 高值加权: SOC>{TAIL_THRESHOLD} 权重={TAIL_WEIGHT}")
		print(f"{'='*80}\n")
	else:
		# 默认行为：归一化标签到 [0,1]
		normalize_oc_flag = True
		clip_oc_flag = True
		print(f"\n[标签策略] 使用默认归一化模式 (标签归一化到 [0,1])\n")
	
	if USE_SRTM:
		mynorm = myNormalize(img_bands_min_max =[[(0,7),(0,1)], [(7,12),(-1,1)], [(12), (-4,2963)], [(13), (0, 90)]], 
							 oc_min = 0, oc_max = OC_MAX, normalize_oc=normalize_oc_flag, clip_oc=clip_oc_flag)
	else:
		mynorm = myNormalize(img_bands_min_max =[[(0,7),(0,1)], [(7,12),(-1,1)]], 
							 oc_min = 0, oc_max = OC_MAX, normalize_oc=normalize_oc_flag, clip_oc=clip_oc_flag)
```

### 位置 3: 修改 train() 函数调用

**查找:**
```python
		else:
			results = train(model, train_dl, test_dl, val_dl,
							optimizer,
							loss_instance, epochs=NUM_EPOCHS, lr_scheduler=LR_SCHEDULER,
							save_model_path= save_model_path,
							save_model_if_mae_lower_than= best_rmse,
							save_train_data_metrics=SAVE_TRAIN_DATA_METRICS
							)
```

**替换为:**
```python
		else:
			results = train(model, train_dl, test_dl, val_dl,
							optimizer,
							loss_instance, epochs=NUM_EPOCHS, lr_scheduler=LR_SCHEDULER,
							save_model_path= save_model_path,
							save_model_if_mae_lower_than= best_rmse,
							save_train_data_metrics=SAVE_TRAIN_DATA_METRICS,
							label_mode=LABEL_MODE,
							huber_beta=HUBER_BETA,
							tail_threshold=TAIL_THRESHOLD,
							tail_weight=TAIL_WEIGHT
							)
```

### 位置 4: 修改 JSON 保存部分

**查找:**
```python
	# 增强功能参数（新增）
	cv_results_full['USE_ENHANCED_CLIMATE'] = USE_ENHANCED_CLIMATE
	cv_results_full['USE_CROSS_MODAL_FUSION'] = USE_CROSS_MODAL_FUSION
	cv_results_full['FUSION_TYPE'] = FUSION_TYPE
	cv_results_full['PRETRAINED_MODEL'] = PRETRAINED_MODEL
	cv_results_full['FREEZE_PRETRAINED'] = FREEZE_PRETRAINED
```

**替换为:**
```python
	# 增强功能参数（新增）
	cv_results_full['USE_ENHANCED_CLIMATE'] = USE_ENHANCED_CLIMATE
	cv_results_full['USE_CROSS_MODAL_FUSION'] = USE_CROSS_MODAL_FUSION
	cv_results_full['FUSION_TYPE'] = FUSION_TYPE
	cv_results_full['PRETRAINED_MODEL'] = PRETRAINED_MODEL
	cv_results_full['FREEZE_PRETRAINED'] = FREEZE_PRETRAINED
	
	# 标签策略参数（消融实验）
	cv_results_full['LABEL_MODE'] = LABEL_MODE
	cv_results_full['HUBER_BETA'] = HUBER_BETA
	cv_results_full['TAIL_THRESHOLD'] = TAIL_THRESHOLD
	cv_results_full['TAIL_WEIGHT'] = TAIL_WEIGHT
```

### 位置 5: 修改 test_step_w_id 调用（两处）

**第一处 - 查找:**
```python
	test_dl_w_id = DataLoader(test_ds_w_id, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
	# Pass region_ids if regional adaptation is enabled
	test_region_ids_for_eval = test_region_ids if args.use_regional_adaptation else None
	test_step_w_id(model=model, data_loader=test_dl_w_id, loss_fn=nn.L1Loss(), verbose=False, csv_file=f"results/RUN_{EXP_NAME}_{run_name}_best.csv", region_ids=test_region_ids_for_eval)
	print(f"Best model saved to results/RUN_{EXP_NAME}_{run_name}_best.csv")
```

**替换为:**
```python
	test_dl_w_id = DataLoader(test_ds_w_id, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
	# Pass region_ids if regional adaptation is enabled
	test_region_ids_for_eval = test_region_ids if args.use_regional_adaptation else None
	test_step_w_id(model=model, data_loader=test_dl_w_id, loss_fn=nn.L1Loss(), verbose=False, 
				   csv_file=f"results/RUN_{EXP_NAME}_{run_name}_best.csv", 
				   region_ids=test_region_ids_for_eval, label_mode=LABEL_MODE)
	print(f"Best model saved to results/RUN_{EXP_NAME}_{run_name}_best.csv")
```

**第二处 - 查找:**
```python
			load_checkpoint(model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY), filename=worst_model_path)
			model.eval()
			print("Worst Model loaded for metrics calculation")
			test_step_w_id(model=model, data_loader=test_dl_w_id, loss_fn=nn.L1Loss(), verbose=False, csv_file=worst_temp_csv, region_ids=test_region_ids_for_eval)
```

**替换为:**
```python
			load_checkpoint(model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY), filename=worst_model_path)
			model.eval()
			print("Worst Model loaded for metrics calculation")
			test_step_w_id(model=model, data_loader=test_dl_w_id, loss_fn=nn.L1Loss(), verbose=False, 
						   csv_file=worst_temp_csv, region_ids=test_region_ids_for_eval, label_mode=LABEL_MODE)
```

### 位置 6: 修改后处理部分（两处）

**第一处 - 查找:**
```python
	# Augment saved CSV with denormalized columns so downstream analysis uses real-scale values
	best_csv_path = f"results/RUN_{EXP_NAME}_{run_name}_best.csv"
	try:
		df = pd.read_csv(best_csv_path)
		if 'y_real_denorm' not in df.columns or 'y_pred_denorm' not in df.columns:
			df['y_real_denorm'] = df['y_real'] * OC_MAX
			df['y_pred_denorm'] = df['y_pred'] * OC_MAX
			denorm_best_csv_path = f"results/RUN_{EXP_NAME}_{run_name}_best_denorm.csv"
			df.to_csv(denorm_best_csv_path, index=False)
			print(f"Augmented denormalized CSV saved: {denorm_best_csv_path}")
	except Exception as e:
		print(f"Warning: failed to write denormalized best CSV: {e}")

	# Compute metrics on real-scale values for best model
	y_true = df['y_real']* OC_MAX
	y_pred = df['y_pred']* OC_MAX
```

**替换为:**
```python
	# Augment saved CSV with denormalized columns so downstream analysis uses real-scale values
	best_csv_path = f"results/RUN_{EXP_NAME}_{run_name}_best.csv"
	try:
		df = pd.read_csv(best_csv_path)
		
		# 根据 label_mode 决定如何获取原尺度值
		if LABEL_MODE in ['baseline_raw_mse', 'log1p_mse', 'log1p_huber', 'log1p_huber_w']:
			# 四组消融实验：CSV中已经包含原尺度值 (y_real_raw, y_pred_raw)
			if 'y_real_raw' in df.columns and 'y_pred_raw' in df.columns:
				print(f"[标签策略] CSV已包含原尺度列 (y_real_raw, y_pred_raw)，直接使用")
				y_true = df['y_real_raw']
				y_pred = df['y_pred_raw']
			else:
				# 兼容旧版：如果没有 _raw 列，假设 y_real/y_pred 已经是原尺度
				print(f"[标签策略] 警告: CSV中未找到 y_real_raw/y_pred_raw 列，假设 y_real/y_pred 为原尺度")
				y_true = df['y_real']
				y_pred = df['y_pred']
		else:
			# 默认模式：需要反归一化
			if 'y_real_denorm' not in df.columns or 'y_pred_denorm' not in df.columns:
				df['y_real_denorm'] = df['y_real'] * OC_MAX
				df['y_pred_denorm'] = df['y_pred'] * OC_MAX
				denorm_best_csv_path = f"results/RUN_{EXP_NAME}_{run_name}_best_denorm.csv"
				df.to_csv(denorm_best_csv_path, index=False)
				print(f"Augmented denormalized CSV saved: {denorm_best_csv_path}")
			y_true = df['y_real'] * OC_MAX
			y_pred = df['y_pred'] * OC_MAX
	except Exception as e:
		print(f"Warning: failed to process best CSV: {e}")
		# 回退到默认行为
		y_true = df['y_real'] * OC_MAX
		y_pred = df['y_pred'] * OC_MAX
```

**第二处 - 查找:**
```python
			# Compute metrics for worst model
			df_worst = pd.read_csv(worst_temp_csv)
			y_true_worst = df_worst['y_real']* OC_MAX
			y_pred_worst = df_worst['y_pred']* OC_MAX
```

**替换为:**
```python
			# Compute metrics for worst model
			df_worst = pd.read_csv(worst_temp_csv)
			
			# 根据 label_mode 决定如何获取原尺度值
			if LABEL_MODE in ['baseline_raw_mse', 'log1p_mse', 'log1p_huber', 'log1p_huber_w']:
				if 'y_real_raw' in df_worst.columns and 'y_pred_raw' in df_worst.columns:
					y_true_worst = df_worst['y_real_raw']
					y_pred_worst = df_worst['y_pred_raw']
				else:
					y_true_worst = df_worst['y_real']
					y_pred_worst = df_worst['y_pred']
			else:
				y_true_worst = df_worst['y_real'] * OC_MAX
				y_pred_worst = df_worst['y_pred'] * OC_MAX
```

---

## 补丁 3: train_utils.py

### 位置 1: 修改 train_step 函数签名和实现

**查找整个 train_step 函数并替换为:**

```python
def train_step(model:nn.Module, data_loader:DataLoader, loss_fn:nn.Module, optimizer:torch.optim.Optimizer,
               label_mode=None, huber_beta=1.0, tail_threshold=30.0, tail_weight=2.0):
    """
    训练步骤，支持标签策略开关
    
    Args:
        label_mode: 标签策略模式 (baseline_raw_mse / log1p_mse / log1p_huber / log1p_huber_w / None)
        huber_beta: Huber损失的beta参数
        tail_threshold: 高值样本阈值
        tail_weight: 高值样本权重
    """
    model.train()
    # Setup train loss and train accuracy values
    train_loss = 0
    loop = tqdm(data_loader, leave=True)
    for batch, (X, y) in enumerate(loop):
        # Send data to target device
        if isinstance(X, tuple) or isinstance(X, list): # if its a tuple it has the climate data in it
            X = [tensor.to(device) for tensor in list(X)]
            y = y.to(device)
        elif isinstance(X, torch.Tensor): # if its a tensor then its only the Image data
            X, y = X.to(device), y.to(device)
        else:
            raise ValueError(f"Input of the netowrk must be either a Tensor or a Tuple of Tensors but it is: {type(X)}")
        
        # 1. Forward pass
        y_pred = model(X)  # shape: [B, 1]

        # 2. 根据 label_mode 计算训练目标和损失
        if label_mode == 'baseline_raw_mse':
            # 原尺度 SOC + MSE
            y_target = y.unsqueeze(1)  # [B, 1]
            loss = torch.nn.functional.mse_loss(y_pred, y_target)
            
        elif label_mode == 'log1p_mse':
            # log1p(SOC) + MSE
            y_target = torch.log1p(torch.clamp(y, min=0)).unsqueeze(1)  # [B, 1]
            loss = torch.nn.functional.mse_loss(y_pred, y_target)
            
        elif label_mode == 'log1p_huber':
            # log1p(SOC) + Huber
            y_target = torch.log1p(torch.clamp(y, min=0)).unsqueeze(1)  # [B, 1]
            loss = torch.nn.functional.smooth_l1_loss(y_pred, y_target, beta=huber_beta)
            
        elif label_mode == 'log1p_huber_w':
            # log1p(SOC) + Huber + 高值加权
            y_target = torch.log1p(torch.clamp(y, min=0)).unsqueeze(1)  # [B, 1]
            err = torch.nn.functional.smooth_l1_loss(y_pred, y_target, beta=huber_beta, reduction='none')  # [B, 1]
            # 计算权重：SOC > tail_threshold 的样本权重为 tail_weight，否则为 1.0
            weights = torch.where(y.unsqueeze(1) > tail_threshold, 
                                 torch.tensor(tail_weight, device=device, dtype=y.dtype), 
                                 torch.tensor(1.0, device=device, dtype=y.dtype))  # [B, 1]
            loss = (weights * err).mean()
            
        else:
            # 默认行为（兼容旧版）：使用传入的 loss_fn
            loss = loss_fn(y_pred, y.unsqueeze(1))
        
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0 or batch == len(data_loader) - 1:
            loss_val = loss.item()
            loop.set_postfix(Train_Loss=train_loss / (batch+1))
            
    train_loss = train_loss / len(data_loader)
    return train_loss
```

### 位置 2: 修改 train 函数签名

**查找:**
```python
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = RMSELoss(),
          epochs: int = 5,
          lr_scheduler: bool = None,
          save_model_path = None,
          save_model_if_mae_lower_than = None,
          save_train_data_metrics = False
          ):
```

**替换为:**
```python
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = RMSELoss(),
          epochs: int = 5,
          lr_scheduler: bool = None,
          save_model_path = None,
          save_model_if_mae_lower_than = None,
          save_train_data_metrics = False,
          label_mode = None,
          huber_beta = 1.0,
          tail_threshold = 30.0,
          tail_weight = 2.0
          ):
```

### 位置 3: 修改 train 函数中调用 train_step 的部分

**查找:**
```python
        # 训练步骤
        train_loss = train_step(model=model,
                                           data_loader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
```

**替换为:**
```python
        # 训练步骤
        train_loss = train_step(model=model,
                                           data_loader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           label_mode=label_mode,
                                           huber_beta=huber_beta,
                                           tail_threshold=tail_threshold,
                                           tail_weight=tail_weight)
```

### 位置 4: 修改 test_step_w_id 函数签名和实现

**查找整个 test_step_w_id 函数并替换为:**

```python
def test_step_w_id(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, csv_file: str = "test.csv", 
                   verbose: bool = False, region_ids: np.ndarray = None, label_mode: str = None):
    """
    测试步骤，输出预测结果到CSV
    
    Args:
        label_mode: 标签策略模式，用于决定如何转换预测值到原尺度
    """
    size = len(data_loader.dataset)
    model.eval()
    test_loss = 0
    results = []  # Store results for CSV
    
    # Check if model supports regional adaptation
    use_regional = hasattr(model, 'use_regional_adaptation') and model.use_regional_adaptation
    if use_regional and region_ids is not None:
        region_ids = np.asarray(region_ids, dtype=np.int64)
        test_ptr = 0

    with torch.inference_mode():
        for batch, (X, y, point_id) in enumerate(data_loader):
            # Send data to target device
            if isinstance(X, tuple) or isinstance(X, list): # if it's a tuple, it has the climate data in it
                X = [tensor.to(device) for tensor in list(X)]
                y = y.to(device)
            elif isinstance(X, torch.Tensor): # if it's a tensor, it's only the Image data
                X, y = X.to(device), y.to(device)
            else:
                raise ValueError(f"Input of the network must be either a Tensor or a Tuple of Tensors but it is: {type(X)}")

            # Handle regional adaptation
            if use_regional and region_ids is not None:
                current_batch_size = y.shape[0] if hasattr(y, 'shape') else len(y)
                end_ptr = min(test_ptr + current_batch_size, len(region_ids))
                batch_region_slice = region_ids[test_ptr:end_ptr]
                test_ptr = end_ptr
                batch_region_ids = torch.as_tensor(batch_region_slice, dtype=torch.long, device=device)
                y_pred = model(X, batch_region_ids)
            else:
                y_pred = model(X)
            
            loss = loss_fn(y_pred, y.unsqueeze(1))
            test_loss += loss.item()

            # Save results for CSV
            if csv_file:
                y_pred = y_pred.squeeze(1)  # Remove the extra dimension from y_pred [B,1] -> [B]
                
                for i in range(len(point_id)):
                    # y 是原尺度 SOC（如果使用标签策略开关）
                    y_real_raw = y[i].item()
                    
                    # 根据 label_mode 转换预测值到原尺度
                    if label_mode in ['log1p_mse', 'log1p_huber', 'log1p_huber_w']:
                        # 模型输出是 log1p(SOC)，需要转换回原尺度
                        y_pred_log = y_pred[i].item()
                        y_pred_raw = np.expm1(y_pred_log)  # expm1(log1p(x)) = x
                        y_real_log = np.log1p(max(0, y_real_raw))
                        
                        results.append({
                            'point_id': point_id[i], 
                            'y_real_raw': y_real_raw,
                            'y_pred_raw': y_pred_raw,
                            'y_real_log': y_real_log,
                            'y_pred_log': y_pred_log,
                            'y_real': y_real_raw,  # 兼容旧版
                            'y_pred': y_pred_raw   # 兼容旧版
                        })
                    elif label_mode == 'baseline_raw_mse':
                        # 模型输出已经是原尺度
                        y_pred_raw = y_pred[i].item()
                        results.append({
                            'point_id': point_id[i], 
                            'y_real_raw': y_real_raw,
                            'y_pred_raw': y_pred_raw,
                            'y_real': y_real_raw,  # 兼容旧版
                            'y_pred': y_pred_raw   # 兼容旧版
                        })
                    else:
                        # 默认行为（兼容旧版）：假设是归一化值
                        results.append({
                            'point_id': point_id[i], 
                            'y_real': y[i].item(), 
                            'y_pred': y_pred[i].item()
                        })

    test_loss /= len(data_loader)
    if verbose:
        print(f"Test Loss: {test_loss:>8f}%")
        print(y_pred.shape, y.shape)

    # Save CSV
    if csv_file:
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False)

    #return test_loss
```

---

## 完成！

所有代码补丁已提供。请按顺序应用这些补丁，然后运行测试脚本验证功能。

