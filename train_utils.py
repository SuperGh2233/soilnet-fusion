import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
from dataset.utils.utils import TextColors as tc
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import shutil
import tempfile
from torchmetrics import R2Score

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def _compute_metrics_on_raw_scale(model, data_loader, label_mode, device):
    """
    在原尺度上计算 MAE, RMSE, R2 指标
    
    Returns:
        mae, rmse, r2 (都在原尺度上)
    """
    model.eval()
    all_y_true_raw = []
    all_y_pred_raw = []
    
    with torch.no_grad():
        for batch, (X, y) in enumerate(data_loader):
            # 移动到设备
            if isinstance(X, (tuple, list)):
                X = [tensor.to(device) for tensor in list(X)]
                y = y.to(device)
            elif isinstance(X, torch.Tensor):
                X, y = X.to(device), y.to(device)
            else:
                raise ValueError(f"Input must be Tensor or Tuple/List, got {type(X)}")
            
            y_pred = model(X)
            
            # 使用统一函数获取原尺度预测
            _, y_true_raw, y_pred_raw = _build_targets_and_pred_raw(
                y_raw=y, y_pred=y_pred, label_mode=label_mode
            )
            
            all_y_true_raw.append(y_true_raw.cpu().numpy())
            all_y_pred_raw.append(y_pred_raw.cpu().numpy())
    
    # 合并所有batch
    y_true_all = np.concatenate(all_y_true_raw)
    y_pred_all = np.concatenate(all_y_pred_raw)
    
    # 计算指标（原尺度）
    mae = np.mean(np.abs(y_true_all - y_pred_all))
    rmse = np.sqrt(np.mean((y_true_all - y_pred_all) ** 2))
    
    # 计算 R2
    ss_res = np.sum((y_true_all - y_pred_all) ** 2)
    ss_tot = np.sum((y_true_all - np.mean(y_true_all)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return float(mae), float(rmse), float(r2)


def _build_targets_and_pred_raw(y_raw: torch.Tensor, y_pred: torch.Tensor, label_mode: str,
                                huber_beta: float = 1.0, tail_threshold: float = 30.0, tail_weight: float = 2.0):
    """
    统一的标签变换和损失计算函数
    
    返回 (loss, y_true_raw, y_pred_raw)：
    - loss: 与 label_mode 一致的训练/验证损失
    - y_true_raw, y_pred_raw: 原尺度，用于 metrics
    
    Args:
        y_raw: 原尺度标签 [B] or [B,1]
        y_pred: 模型输出 [B] or [B,1]
        label_mode: 标签策略模式
        huber_beta: Huber损失的beta参数
        tail_threshold: 高值样本阈值
        tail_weight: 高值样本权重
    """
    y_raw = y_raw.view(-1)             # [B]
    y_pred = y_pred.view(-1)           # [B]

    if label_mode == 'baseline_raw_mse':
        # 原尺度 SOC + MSE
        # 建议：确保预测非负，避免负SOC
        y_pred_raw = F.softplus(y_pred)
        y_true_raw = y_raw
        loss = F.mse_loss(y_pred_raw, y_true_raw)

    elif label_mode == 'log1p_mse':
        # log1p(SOC) + MSE
        y_true_log = torch.log1p(torch.clamp(y_raw, min=0))
        y_pred_log = y_pred
        loss = F.mse_loss(y_pred_log, y_true_log)
        y_true_raw = y_raw
        y_pred_raw = torch.expm1(y_pred_log)

    elif label_mode == 'log1p_huber':
        # log1p(SOC) + Huber
        y_true_log = torch.log1p(torch.clamp(y_raw, min=0))
        y_pred_log = y_pred
        loss = F.smooth_l1_loss(y_pred_log, y_true_log, beta=huber_beta)
        y_true_raw = y_raw
        y_pred_raw = torch.expm1(y_pred_log)

    elif label_mode == 'log1p_huber_w':
        # log1p(SOC) + Huber + 高值加权
        y_true_log = torch.log1p(torch.clamp(y_raw, min=0))
        y_pred_log = y_pred
        err = F.smooth_l1_loss(y_pred_log, y_true_log, beta=huber_beta, reduction='none')  # [B]
        w = torch.where(y_raw > tail_threshold,
                        torch.tensor(tail_weight, device=y_raw.device, dtype=y_raw.dtype),
                        torch.tensor(1.0, device=y_raw.device, dtype=y_raw.dtype))
        loss = (w * err).mean()
        y_true_raw = y_raw
        y_pred_raw = torch.expm1(y_pred_log)

    else:
        # 兼容旧版：认为 y 是 [0,1] 的归一化标签
        # 注意：这里无法恢复原尺度，metrics 只能按旧流程外部处理
        y_true_raw = y_raw
        y_pred_raw = y_pred
        loss = None

    return loss, y_true_raw, y_pred_raw


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))
    
class R2Loss(nn.Module):
    """
    Calculates the R2 loss for regression problems.

    The R2 loss measures the proportion of variance in the dependent variable that can be explained by the independent
    variable. It is also known as the coefficient of determination.

    Args:
        None

    Shape:
        - Input: (batch_size, *)
        - Target: (batch_size, *)
        - Output: scalar value

    Attributes:
        mse (nn.MSELoss): Mean squared error loss

    Examples::
        >>> loss = R2Loss()
        >>> yhat = torch.tensor([1, 2, 3, 4])
        >>> y = torch.tensor([2, 4, 6, 8])
        >>> r2 = loss(yhat, y)
    """

    def __init__(self):
        """
        Initializes the R2Loss module.
        """
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        """
        Calculates the R2 loss for the given predictions and targets.

        Args:
            yhat (torch.Tensor): Predictions tensor of shape (batch_size, *)
            y (torch.Tensor): Targets tensor of shape (batch_size, *)

        Returns:
            torch.Tensor: Scalar tensor representing the R2 loss
        """
        ones = torch.ones_like(y)
        return 1 - (self.mse(yhat,y)/self.mse(y,ones*y.mean()))
    

class RMSLELoss(nn.Module):
    def __init__(self):
        super(RMSLELoss, self).__init__()

    def forward(self, predictions, actuals):
        """
        Compute the Root Mean Squared Logarithmic Error.
        
        Args:
            predictions (torch.Tensor): The predicted values.
            actuals (torch.Tensor): The actual values.
        
        Returns:
            torch.Tensor: The computed RMSLE value.
        """
        # Ensure predictions are greater than -1, as log(0) and negative values are undefined
        predictions = torch.clamp(predictions, min=-1 + 1e-9)
        actuals = torch.clamp(actuals, min=-1 + 1e-9)

        # Calculate the log loss
        log_diff = torch.log(predictions + 1) - torch.log(actuals + 1)
        squared_log_diff = torch.square(log_diff)

        # Return the square root of the mean of squared log differences
        return torch.sqrt(torch.mean(squared_log_diff))


    

def train_step(model:nn.Module, data_loader:DataLoader, loss_fn:nn.Module, optimizer:torch.optim.Optimizer,
               label_mode=None, huber_beta=1.0, tail_threshold=30.0, tail_weight=2.0,
               alignment_loss_fn=None, lambda_align=0.1):
    """
    训练步骤，支持标签策略开关（使用统一的变换函数）和 S-CMRL 对齐损失
    
    Args:
        label_mode: 标签策略模式 (baseline_raw_mse / log1p_mse / log1p_huber / log1p_huber_w / None)
        huber_beta: Huber损失的beta参数
        tail_threshold: 高值样本阈值
        tail_weight: 高值样本权重
        alignment_loss_fn: 语义对齐损失函数（S-CMRL），如果提供则计算对齐损失
        lambda_align: 对齐损失的权重
    """
    model.train()
    train_loss = 0.0
    align_loss_total = 0.0
    loop = tqdm(data_loader, leave=True)
    
    for batch, (X, y) in enumerate(loop):
        # Send data to target device
        if isinstance(X, (tuple, list)):
            X = [tensor.to(device) for tensor in list(X)]
            y = y.to(device)
        elif isinstance(X, torch.Tensor):
            X, y = X.to(device), y.to(device)
        else:
            raise ValueError(f"Input of the network must be either a Tensor or a Tuple of Tensors but it is: {type(X)}")
        
        # Forward pass
        y_pred = model(X)  # [B,1] or [B]

        # 根据 label_mode 计算主损失（使用统一函数）
        if label_mode in ['baseline_raw_mse', 'log1p_mse', 'log1p_huber', 'log1p_huber_w']:
            loss, _, _ = _build_targets_and_pred_raw(
                y_raw=y, y_pred=y_pred, label_mode=label_mode,
                huber_beta=huber_beta, tail_threshold=tail_threshold, tail_weight=tail_weight
            )
        else:
            # 默认行为（兼容旧版）：使用传入的 loss_fn
            loss = loss_fn(y_pred, y.unsqueeze(1))
        
        # S-CMRL 对齐损失（如果启用）
        total_loss = loss
        align_loss = None
        if alignment_loss_fn is not None and hasattr(model, 'use_scmrl_fusion') and model.use_scmrl_fusion:
            # 获取中间特征（在 forward 中已保存）
            if hasattr(model, '_last_climate_feat') and hasattr(model, '_last_visual_feat'):
                climate_feat = model._last_climate_feat
                visual_feat = model._last_visual_feat
                align_loss = alignment_loss_fn(climate_feat, visual_feat)
                total_loss = loss + lambda_align * align_loss
                align_loss_total += align_loss.item()

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # 更新进度条
        postfix = {'Train_Loss': train_loss / (batch+1)}
        if align_loss is not None:
            postfix['Align_Loss'] = align_loss_total / (batch+1)
        
        if batch % 10 == 0 or batch == len(data_loader) - 1:
            loop.set_postfix(**postfix)
            
    train_loss = train_loss / max(1, len(data_loader))
    align_loss_avg = align_loss_total / max(1, len(data_loader)) if alignment_loss_fn is not None else 0.0
    
    return train_loss, align_loss_avg


# Test step function
def test_step(model:nn.Module, data_loader:DataLoader, loss_fn:nn.Module, verbose = False,
              label_mode=None, huber_beta=1.0, tail_threshold=30.0, tail_weight=2.0):
    """
    测试/验证步骤，支持标签策略开关
    
    Args:
        label_mode: 标签策略模式，如果提供则使用统一变换函数
        huber_beta, tail_threshold, tail_weight: 标签策略参数
    """
    size = len(data_loader.dataset)
    # 针对 R2Score 特殊处理
    if isinstance(loss_fn, R2Score):
        if size < 2:
            if verbose:
                print("Not enough samples to calculate R2 score. Returning None.")
            return None
        model.eval()
        metric = R2Score().to(next(model.parameters()).device)
        with torch.inference_mode():
            for batch, (X, y) in enumerate(data_loader):
                if isinstance(X, (tuple, list)):
                    X = [tensor.to(next(model.parameters()).device) for tensor in list(X)]
                    y = y.to(next(model.parameters()).device)
                elif isinstance(X, torch.Tensor):
                    X, y = X.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
                else:
                    raise ValueError(f"Input of the network must be either a Tensor or a Tuple of Tensors but it is: {type(X)}")
                y_pred = model(X)
                metric.update(y_pred, y.unsqueeze(1))
        result = metric.compute().item()
        if verbose:
            print(f"R2 Score: {result}")
        return result
    
    # 其他 loss_fn：根据 label_mode 决定计算方式
    model.eval()
    test_loss = 0.0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            if isinstance(X, (tuple, list)):
                X = [tensor.to(next(model.parameters()).device) for tensor in list(X)]
                y = y.to(next(model.parameters()).device)
            elif isinstance(X, torch.Tensor):
                X, y = X.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
            else:
                raise ValueError(f"Input of the network must be either a Tensor or a Tuple of Tensors but it is: {type(X)}")
            y_pred = model(X)
            
            # 如果使用标签策略开关，使用统一函数计算损失
            if label_mode in ['baseline_raw_mse', 'log1p_mse', 'log1p_huber', 'log1p_huber_w']:
                loss, _, _ = _build_targets_and_pred_raw(
                    y_raw=y, y_pred=y_pred, label_mode=label_mode,
                    huber_beta=huber_beta, tail_threshold=tail_threshold, tail_weight=tail_weight
                )
            else:
                # 默认行为（兼容旧版）：使用传入的 loss_fn
                loss = loss_fn(y_pred, y.unsqueeze(1))
            
            test_loss += loss.item()
    test_loss /= len(data_loader)
    if verbose:
        print(f"Test Loss: {test_loss:>8f}")
        print(y_pred.shape, y.shape)
    return test_loss



import pandas as pd

def test_step_w_id(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, csv_file: str = "test.csv", 
                   verbose: bool = False, region_ids: np.ndarray = None, label_mode: str = None):
    """
    测试步骤，输出预测结果到CSV（使用统一的变换函数）
    
    Args:
        label_mode: 标签策略模式，用于决定如何转换预测值到原尺度
    
    注意：不再计算 test_loss，因为不同 label_mode 的 loss 不可比
    """
    model.eval()
    results = []
    
    # Check if model supports regional adaptation
    use_regional = hasattr(model, 'use_regional_adaptation') and model.use_regional_adaptation
    if use_regional and region_ids is not None:
        region_ids = np.asarray(region_ids, dtype=np.int64)
        ptr = 0

    with torch.no_grad():
        for batch, (X, y, point_id) in enumerate(data_loader):
            # Send data to target device
            if isinstance(X, (tuple, list)):
                X = [tensor.to(device) for tensor in list(X)]
                y = y.to(device)
            elif isinstance(X, torch.Tensor):
                X, y = X.to(device), y.to(device)
            else:
                raise ValueError(f"Input of the network must be either a Tensor or a Tuple of Tensors but it is: {type(X)}")

            # Handle regional adaptation
            if use_regional and region_ids is not None:
                current_batch_size = y.shape[0] if hasattr(y, 'shape') else len(y)
                batch_region = torch.as_tensor(region_ids[ptr:ptr+current_batch_size], dtype=torch.long, device=device)
                ptr += current_batch_size
                y_pred = model(X, batch_region)
            else:
                y_pred = model(X)

            # 统一得到原尺度预测（使用统一函数）
            if label_mode in ['baseline_raw_mse', 'log1p_mse', 'log1p_huber', 'log1p_huber_w']:
                _, y_true_raw, y_pred_raw = _build_targets_and_pred_raw(
                    y_raw=y, y_pred=y_pred, label_mode=label_mode
                )
                y_true_raw_np = y_true_raw.detach().cpu().numpy()
                y_pred_raw_np = y_pred_raw.detach().cpu().numpy()
                
                # 同时导出 log 便于分析（可选）
                y_true_log_np = np.log1p(np.clip(y_true_raw_np, 0, None))
                if label_mode.startswith('log1p'):
                    y_pred_log_np = y_pred.view(-1).detach().cpu().numpy()
                else:
                    y_pred_log_np = np.log1p(np.clip(y_pred_raw_np, 0, None))
                
                # 保存结果
                for i in range(len(point_id)):
                    row = {
                        'point_id': point_id[i],
                        'y_real_raw': float(y_true_raw_np[i]),
                        'y_pred_raw': float(y_pred_raw_np[i]),
                        'y_real_log': float(y_true_log_np[i]),
                        'y_pred_log': float(y_pred_log_np[i]),
                        'y_real': float(y_true_raw_np[i]),  # 兼容旧版
                        'y_pred': float(y_pred_raw_np[i])   # 兼容旧版
                    }
                    results.append(row)
            else:
                # 默认行为（兼容旧版）：假设是归一化值
                y_np = y.detach().cpu().numpy()
                y_pred_np = y_pred.view(-1).detach().cpu().numpy()
                
                for i in range(len(point_id)):
                    results.append({
                        'point_id': point_id[i], 
                        'y_real': float(y_np[i]), 
                        'y_pred': float(y_pred_np[i])
                    })

    # Save CSV
    if csv_file:
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False)
        if verbose:
            print(f"Saved {len(results)} predictions to {csv_file}")

    #return test_loss  # 不再返回 loss


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("Saving checkpoint=> ", end="")
    temp_filename = None
    try:
        # 确保目录存在
        dirname = os.path.dirname(filename) if os.path.dirname(filename) else '.'
        os.makedirs(dirname, exist_ok=True)
        
        # 将模型和优化器状态移到CPU，避免GPU序列化问题
        model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        optimizer_state_dict = {}
        for k, v in optimizer.state_dict().items():
            if isinstance(v, dict):
                # 处理optimizer state_dict中嵌套的字典（如state, param_groups等）
                optimizer_state_dict[k] = {k2: v2.cpu() if isinstance(v2, torch.Tensor) else v2 
                                          for k2, v2 in v.items()}
            elif isinstance(v, torch.Tensor):
                optimizer_state_dict[k] = v.cpu()
            else:
                optimizer_state_dict[k] = v
        
        checkpoint = {
            "state_dict": model_state_dict,
            "optimizer": optimizer_state_dict,
        }
        
        # 使用临时文件先保存，然后原子性地重命名（避免写入中断导致的损坏）
        temp_filename = filename + ".tmp"
        
        # 如果临时文件已存在，先删除
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        
        torch.save(checkpoint, temp_filename)
        
        # 验证临时文件确实被创建
        if not os.path.exists(temp_filename):
            raise RuntimeError(f"Temporary file {temp_filename} was not created successfully")
        
        # 原子性地移动到最终位置
        try:
            if os.path.exists(filename):
                os.replace(temp_filename, filename)
            else:
                # 使用os.rename在Windows上更可靠
                os.rename(temp_filename, filename)
        except OSError:
            # 如果重命名失败，尝试使用shutil.move
            shutil.move(temp_filename, filename)
        
        print("Done!")
    except (RuntimeError, OSError, FileNotFoundError) as e:
        print(f"Failed! Error: {e}")
        # 清理临时文件
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass
        
        # 检查是否是磁盘空间或IO问题
        if "file write failed" in str(e) or "pos" in str(e) or "No such file" in str(e):
            print(f"Possible causes: disk space full, disk I/O error, or file system issue.")
            print(f"Attempting direct save (without temporary file)...")
            # 尝试直接保存（不使用临时文件）
            try:
                model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                optimizer_state_dict = {}
                for k, v in optimizer.state_dict().items():
                    if isinstance(v, dict):
                        optimizer_state_dict[k] = {k2: v2.cpu() if isinstance(v2, torch.Tensor) else v2 
                                                  for k2, v2 in v.items()}
                    elif isinstance(v, torch.Tensor):
                        optimizer_state_dict[k] = v.cpu()
                    else:
                        optimizer_state_dict[k] = v
                checkpoint = {
                    "state_dict": model_state_dict,
                    "optimizer": optimizer_state_dict,
                }
                torch.save(checkpoint, filename)
                print(f"Checkpoint saved directly to {filename}")
            except Exception as e2:
                print(f"Direct save also failed: {e2}")
                raise
        else:
            raise
    except Exception as e:
        print(f"Failed! Unexpected error: {e}")
        # 清理临时文件
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass
        raise
    
def load_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("Loading checkpoint=> ", end="")
    checkpoint = torch.load(filename)
    # 处理不同的checkpoint格式
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    elif "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    print("Done!")

# 1. Take in various parameters required for training and test steps
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
          tail_weight = 2.0,
          alignment_loss_fn = None,
          lambda_align = 0.1,
          select_best_on_val = False
          ):
    """ Train the model and test it on the test set
    Note: If you don't have diffrent validation and test sets, just pass the same dataloader for both test and val

    Args:
        model (torch.nn.Module): Pytorch model
        train_dataloader (torch.utils.data.DataLoader): train dataloader
        test_dataloader (torch.utils.data.DataLoader): test dataloader
        val_dataloader (torch.utils.data.DataLoader): validation dataloader
        optimizer (torch.optim.Optimizer): optimizer
        loss_fn (torch.nn.Module, optional): Loss funciton. Defaults to RMSELoss().
        epochs (int, optional): Number of Epochs. Defaults to 5.
        lr_scheduler (bool, optional): Use LR scheduler. Defaults to None, Options are "plateau" or "step" . Defaults to None. / plateau or step
        save_model_path (str, optional): If given, saves the model with the given name and path. Defaults to None | Example: "my_checkpoint.pth.tar".

    Returns:
        _type_: _description_
    """
    if lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    elif lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2, verbose=True)
    else:
        pass
    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "val_loss": [],
               "MAE": [],
               "RMSE": [],
               "R2": [],
               "train_MAE": [],
                "train_RMSE": [],
                "train_R2": [],
                "best_epoch": None,
                "best_val_loss": None,
    }
    best_val_loss = float('inf')
    best_epoch = None
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in range(1, epochs+1):
        print(tc.OKGREEN,f"Epoch {epoch}\n-------------------------------",tc.ENDC)
        
        # 训练步骤
        train_loss, align_loss = train_step(model=model,
                                           data_loader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           label_mode=label_mode,
                                           huber_beta=huber_beta,
                                           tail_threshold=tail_threshold,
                                           tail_weight=tail_weight,
                                           alignment_loss_fn=alignment_loss_fn,
                                           lambda_align=lambda_align)
        
        # 确保模型处于评估模式进行验证
        model.eval()
        val_loss = test_step(model=model,
            data_loader=val_dataloader,
            loss_fn=loss_fn,
            label_mode=label_mode,
            huber_beta=huber_beta,
            tail_threshold=tail_threshold,
            tail_weight=tail_weight)

        if select_best_on_val and val_loss < best_val_loss:
            if not save_model_path:
                raise ValueError('select_best_on_val requires save_model_path')
            best_val_loss = float(val_loss)
            best_epoch = epoch
            save_checkpoint(model, optimizer, filename=save_model_path)
        # 验证后恢复训练模式
        model.train()
        
        # 4. Print out what's happening
        print_msg = f"Epoch {epoch} Results: | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}"
        if alignment_loss_fn is not None and align_loss > 0:
            print_msg += f" | align_loss: {align_loss:.6f}"
        print(tc.OKCYAN, print_msg, tc.ENDC)
        
        # 打印 Alpha 值（如果使用 S-CMRL 融合）
        if hasattr(model, 'use_scmrl_fusion') and model.use_scmrl_fusion and hasattr(model, 'fusion'):
            alphas = model.fusion.get_alpha_values()
            alpha_msg = "Alpha values: "
            alpha_items = [f"{k}={v:.4f}" for k, v in alphas.items()]
            alpha_msg += " | ".join(alpha_items)
            print(tc.OKCYAN, alpha_msg, tc.ENDC)
        
        print("")

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        if lr_scheduler == "step":
            scheduler.step()
        elif lr_scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            pass
    if select_best_on_val:
        if best_epoch is None:
            raise RuntimeError('No validation checkpoint was selected')
        load_checkpoint(model, optimizer, filename=save_model_path)
        results["best_epoch"] = best_epoch
        results["best_val_loss"] = best_val_loss

    # 计算测试集指标（必须在原尺度上计算）
    if label_mode in ['baseline_raw_mse', 'log1p_mse', 'log1p_huber', 'log1p_huber_w']:
        # label_mode 模式：在原尺度上计算指标
        test_mae, test_rmse, test_r2 = _compute_metrics_on_raw_scale(
            model, test_dataloader, label_mode, device
        )
        results["MAE"].append(test_mae)
        results["RMSE"].append(test_rmse)
        results["R2"].append(test_r2)
        
        if save_train_data_metrics:
            train_mae, train_rmse, train_r2 = _compute_metrics_on_raw_scale(
                model, train_dataloader, label_mode, device
            )
            results["train_MAE"].append([train_mae])
            results["train_RMSE"].append([train_rmse])
            results["train_R2"].append([train_r2])
    else:
        # 默认模式：使用原有的 test_step
        results["MAE"].append(test_step(model=model, data_loader=test_dataloader, loss_fn=nn.L1Loss(), verbose=False))
        results["RMSE"].append(test_step(model=model, data_loader=test_dataloader, loss_fn=RMSELoss(), verbose=False))
        results["R2"].append(test_step(model=model, data_loader=test_dataloader, loss_fn=R2Score().to(device), verbose=False)) 
        if save_train_data_metrics:
            results["train_MAE"].append([test_step(model=model, data_loader=train_dataloader, loss_fn=nn.L1Loss(), verbose=False)])
            results["train_RMSE"].append([test_step(model=model, data_loader=train_dataloader, loss_fn=RMSELoss(), verbose=False)])
            results["train_R2"].append([test_step(model=model, data_loader=train_dataloader, loss_fn=R2Score().to(device), verbose=False)])
    # Save the model
    if save_model_path and not select_best_on_val:
        if save_model_if_mae_lower_than:
            if results["MAE"][-1] < save_model_if_mae_lower_than:
                save_checkpoint(model, optimizer, filename=save_model_path)
        else:
            save_checkpoint(model, optimizer, filename=save_model_path)
    # 6. Return the filled results at the end of the epochs
    return results




def plot_losses(loss_dict):
    train_losses = loss_dict["train_loss"]
    val_losses = loss_dict["val_loss"]
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    
class BatchLoader(torch.utils.data.Dataset): 
    """ Takes in a Pytorch DataLoader and returns any batch using index
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __len__(self):
        return len(self.dataloader)

    def __call__(self, index):
        for i, batch in enumerate(self.dataloader):
            if i == index:
                return batch
        raise IndexError("Index out of range")
    



def evaluate_regression_metrics(y_true, y_pred):
    """Calculate multiple regression evaluation metrics."""
    # y_true = y_true * 87  # Multiply y_true by 87
    # y_pred = y_pred * 87  # Multiply y_pred by 87
    
    # Calculate RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate R2 (R-squared)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate RPIQ (Relative Prediction Interval Quality)

    y_std = np.std(y_true)
    rpiq = 1 - (rmse / y_std)
    
    # Calculate MAE (Mean Absolute Error)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate MEC (Mean Error Correction)
    mec = np.mean(y_true - y_pred)


    def rpiq_metric(y_real, y_pred):
     # Calculate quartiles Q1 and Q3 (both should use y_real for standard RPIQ definition)
     q1 = np.percentile(y_real, 25)
     q3 = np.percentile(y_real, 75)  # Fixed: should use y_real, not y_pred

     # Calculate RMSE
     rmse = np.sqrt(mean_squared_error(y_real, y_pred))

     # Calculate the ratio of the difference between Q3 and Q1 to RMSE
     ratio = (q3 - q1) / rmse

     return ratio
    
    rpiq = rpiq_metric(y_true, y_pred)

    
    # Calculate CCC (Concordance Correlation Coefficient)
    def concordance_correlation_coefficient(y_real, y_pred):
        # Raw data
        dct = {
            'y_real': y_real,
            'y_pred': y_pred
        }
        df = pd.DataFrame(dct)
        # Remove NaNs
        df = df.dropna()
        # Pearson product-moment correlation coefficients
        y_real = df['y_real']
        y_pred = df['y_pred']
        cor = np.corrcoef(y_real, y_pred)[0][1]
        # Means
        mean_real = np.mean(y_real)
        mean_pred = np.mean(y_pred)
        # Population variances
        var_real = np.var(y_real)
        var_pred = np.var(y_pred)
        # Population standard deviations
        sd_real = np.std(y_real)
        sd_pred = np.std(y_pred)
        # Calculate CCC
        numerator = 2 * cor * sd_real * sd_pred
        denominator = var_real + var_pred + (mean_real - mean_pred)**2

        return numerator / denominator
    
    ccc = concordance_correlation_coefficient(y_true, y_pred)
    
    return rmse, r2, rpiq, mae, mec, ccc    

#Physics-aware loss function design
# loss_lower = torch.mean(torch.max((1 - self.q) * errors, torch.zeros_like(errors)))
# loss_upper = torch.mean(torch.max(self.q * errors, torch.zeros_like(errors)))
class PhysicsPinballLoss(nn.Module):
    """
    Calculates quantile (pinball) loss function + two penalty terms for predictions
    that are lower than the lower bound and more than the upper bound.

    Args:
     q: your desired lower quantile (e.g., 0.1)
     beta: scaling factor for penalty term
    """

    def __init__(self, q, beta):
        super(PhysicsPinballLoss, self).__init__()
        self.q = q
        self.beta = beta

    def forward(self, y_pred, y_true):
        if self.q >= 0.5:
            raise ValueError('The input quantile should be lower than 0.5')
        else:
            e = y_true - y_pred
            loss_lower = torch.mean(torch.max(self.q * e, (self.q - 1) * e))
            loss_upper = torch.mean(torch.max((1 - self.q) * e, ((1 - self.q) - 1) * e))

            lower_bound = y_pred - loss_upper
            upper_bound = y_pred - loss_lower

            # Penalty terms based on conditions
            penalty_lower = torch.where(y_true < lower_bound, self.beta * (lower_bound - y_true), torch.tensor(0.0, device=device))
            penalty_upper = torch.where(y_true > upper_bound, self.beta * (y_true - upper_bound), torch.tensor(0.0, device=device))

            return torch.mean(loss_lower * (1 + penalty_lower) + loss_upper * (1 + penalty_upper))


class HybridR2Loss(nn.Module):
    """
    混合损失函数：结合R²损失和MSE损失
    在训练初期使用MSE，后期逐渐增加R²损失的权重
    """
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha  # MSE损失权重
        self.beta = beta    # R²损失权重
        self.mse = nn.MSELoss()
        
    def forward(self, yhat, y):
        mse_loss = self.mse(yhat, y)
        
        # R²损失计算
        ones = torch.ones_like(y)
        r2_loss = 1 - (self.mse(yhat, y) / self.mse(y, ones * y.mean()))
        
        # 混合损失
        total_loss = self.alpha * mse_loss + self.beta * r2_loss
        return total_loss

class FocalR2Loss(nn.Module):
    """
    焦点R²损失：关注难预测的样本
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
    def forward(self, yhat, y):
        # 计算预测误差
        error = torch.abs(yhat - y)
        
        # 计算焦点权重
        focal_weight = (1 - torch.exp(-error)) ** self.gamma
        
        # R²损失
        ones = torch.ones_like(y)
        r2_loss = 1 - (self.mse(yhat, y) / self.mse(y, ones * y.mean()))
        
        # 应用焦点权重
        focal_loss = focal_weight * r2_loss
        
        return focal_loss.mean()


def apply_mixup(x1, x2, y1, y2, alpha=0.2):
    """
    Mixup数据增强：混合两个样本
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size()[0]
    index = torch.randperm(batch_size).to(x1.device)

    mixed_x = lam * x1 + (1 - lam) * x2[index, :]
    mixed_y = lam * y1 + (1 - lam) * y2[index, :]
    
    return mixed_x, mixed_y

def apply_cutmix(x1, x2, y1, y2, alpha=1.0):
    """
    CutMix数据增强：裁剪混合两个样本
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size()[0]
    index = torch.randperm(batch_size).to(x1.device)

    y_a, y_b = y1, y2[index]
    
    # 生成裁剪框
    W, H = x1.size()[2], x1.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x1[:, :, bbx1:bbx2, bby1:bby2] = x2[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x1, y_a * lam + y_b * (1 - lam)

class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失：减少过拟合
    """
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))



def create_adaptive_scheduler(optimizer, epochs, initial_lr, scheduler_type="cosine_warmup"):
    """
    创建自适应学习率调度器
    """
    if scheduler_type == "cosine_warmup":
        # 余弦退火 + 预热
        warmup_epochs = max(1, epochs // 10)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=initial_lr,
            epochs=epochs,
            steps_per_epoch=1,
            pct_start=warmup_epochs/epochs,
            anneal_strategy='cos'
        )
    elif scheduler_type == "exponential":
        # 指数衰减
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=0.95
        )
    elif scheduler_type == "cyclic":
        # 循环学习率
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=initial_lr/100,
            max_lr=initial_lr,
            step_size_up=epochs//4,
            mode='triangular2'
        )
    else:
        scheduler = None
    
    return scheduler

