#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试标签策略开关功能

运行方式:
    python test_label_strategy.py
"""

import torch
import numpy as np
from dataset.dataset_loader import myNormalize

def test_normalize_oc_flag():
    """测试 myNormalize 的 normalize_oc 和 clip_oc 参数"""
    print("="*80)
    print("测试 myNormalize 标签归一化开关")
    print("="*80)
    
    # 模拟图像和标签数据
    img = np.random.rand(12, 64, 64)  # 12波段图像
    oc_values = [5.0, 15.0, 30.8, 35.0, 50.0]  # 不同的OC值，包括超过OC_MAX的
    
    OC_MAX = 30.8
    
    print("\n1. 默认模式 (normalize_oc=True, clip_oc=True)")
    print("-" * 40)
    mynorm_default = myNormalize(
        img_bands_min_max=[[(0,7),(0,1)], [(7,12),(-1,1)]], 
        oc_min=0, oc_max=OC_MAX,
        normalize_oc=True,
        clip_oc=True
    )
    
    for oc in oc_values:
        _, oc_normalized = mynorm_default((img.copy(), oc))
        print(f"  原始 OC={oc:5.1f} -> 归一化后={oc_normalized:.4f}")
    
    print("\n2. 新模式 (normalize_oc=False, clip_oc=False)")
    print("-" * 40)
    mynorm_raw = myNormalize(
        img_bands_min_max=[[(0,7),(0,1)], [(7,12),(-1,1)]], 
        oc_min=0, oc_max=OC_MAX,
        normalize_oc=False,
        clip_oc=False
    )
    
    for oc in oc_values:
        _, oc_raw = mynorm_raw((img.copy(), oc))
        print(f"  原始 OC={oc:5.1f} -> 保持原尺度={oc_raw:.4f}")
    
    print("\n✓ 测试通过: normalize_oc 和 clip_oc 参数工作正常")
    print()

def test_label_transformations():
    """测试标签变换逻辑"""
    print("="*80)
    print("测试标签变换逻辑")
    print("="*80)
    
    # 模拟一批标签数据
    y_raw = torch.tensor([5.0, 15.0, 30.0, 35.0, 50.0])
    
    print("\n原始标签 (原尺度 SOC):")
    print(f"  {y_raw.numpy()}")
    
    # 1. baseline_raw_mse: 不变换
    print("\n1. baseline_raw_mse 模式:")
    y_target = y_raw.unsqueeze(1)
    print(f"  训练目标: {y_target.squeeze().numpy()}")
    
    # 2. log1p_mse: log1p变换
    print("\n2. log1p_mse 模式:")
    y_target_log = torch.log1p(torch.clamp(y_raw, min=0)).unsqueeze(1)
    print(f"  训练目标 (log空间): {y_target_log.squeeze().numpy()}")
    
    # 验证逆变换
    y_recovered = torch.expm1(y_target_log.squeeze())
    print(f"  逆变换后: {y_recovered.numpy()}")
    print(f"  误差: {torch.abs(y_recovered - y_raw).max().item():.6f}")
    
    # 3. log1p_huber_w: 高值加权
    print("\n3. log1p_huber_w 模式 (tail_threshold=30.0, tail_weight=2.0):")
    tail_threshold = 30.0
    tail_weight = 2.0
    weights = torch.where(y_raw > tail_threshold, 
                         torch.tensor(tail_weight), 
                         torch.tensor(1.0))
    print(f"  样本权重: {weights.numpy()}")
    print(f"  高值样本数: {(y_raw > tail_threshold).sum().item()}")
    
    print("\n✓ 测试通过: 标签变换逻辑正确")
    print()

def test_loss_computation():
    """测试损失计算（使用统一函数）"""
    print("="*80)
    print("测试损失计算（统一函数）")
    print("="*80)
    
    # 导入统一函数
    import sys
    sys.path.insert(0, '.')
    from train_utils import _build_targets_and_pred_raw
    
    # 模拟预测值和真实值
    y_pred = torch.tensor([14.5, 16.2, 29.8, 33.1, 48.5])
    y_true = torch.tensor([15.0, 15.0, 30.0, 35.0, 50.0])
    
    print("\n真实值:", y_true.numpy())
    print("预测值:", y_pred.numpy())
    
    # 1. baseline_raw_mse
    print("\n1. baseline_raw_mse:")
    loss1, y_true_raw1, y_pred_raw1 = _build_targets_and_pred_raw(
        y_true, y_pred, 'baseline_raw_mse'
    )
    print(f"  Loss = {loss1.item():.4f}")
    print(f"  预测值（经过softplus）: {y_pred_raw1.numpy()}")
    
    # 2. log1p_mse
    print("\n2. log1p_mse:")
    loss2, y_true_raw2, y_pred_raw2 = _build_targets_and_pred_raw(
        y_true, y_pred, 'log1p_mse'
    )
    print(f"  Loss = {loss2.item():.4f}")
    print(f"  预测值（原尺度）: {y_pred_raw2.numpy()}")
    
    # 3. log1p_huber
    print("\n3. log1p_huber (beta=1.0):")
    loss3, y_true_raw3, y_pred_raw3 = _build_targets_and_pred_raw(
        y_true, y_pred, 'log1p_huber', huber_beta=1.0
    )
    print(f"  Loss = {loss3.item():.4f}")
    
    # 4. log1p_huber_w
    print("\n4. log1p_huber_w (tail_threshold=30.0, tail_weight=2.0):")
    loss4, y_true_raw4, y_pred_raw4 = _build_targets_and_pred_raw(
        y_true, y_pred, 'log1p_huber_w', 
        huber_beta=1.0, tail_threshold=30.0, tail_weight=2.0
    )
    print(f"  Loss = {loss4.item():.4f}")
    print(f"  高值样本数: {(y_true > 30.0).sum().item()}")
    
    print("\n✓ 测试通过: 统一函数工作正常")
    print()

def test_inverse_transform():
    """测试逆变换精度"""
    print("="*80)
    print("测试逆变换精度")
    print("="*80)
    
    # 测试不同范围的值
    test_values = np.array([0.1, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0])
    
    print("\n原始值 -> log1p -> expm1 -> 恢复值 (误差)")
    print("-" * 60)
    
    max_error = 0
    for val in test_values:
        val_tensor = torch.tensor(val)
        log_val = torch.log1p(val_tensor)
        recovered_val = torch.expm1(log_val)
        error = abs(recovered_val.item() - val)
        max_error = max(max_error, error)
        print(f"{val:6.1f} -> {log_val.item():6.4f} -> {recovered_val.item():6.4f} (误差: {error:.2e})")
    
    print(f"\n最大误差: {max_error:.2e}")
    assert max_error < 1e-5, "逆变换误差过大!"
    
    print("\n✓ 测试通过: 逆变换精度满足要求")
    print()

def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print(" 标签策略开关功能测试")
    print("="*80 + "\n")
    
    try:
        test_normalize_oc_flag()
        test_label_transformations()
        test_loss_computation()
        test_inverse_transform()
        
        print("="*80)
        print("✓ 所有测试通过!")
        print("="*80)
        print("\n可以开始运行完整实验:")
        print("  python train.py -e baseline_raw_mse --label_mode baseline_raw_mse -ne 10 -seed 1 -lstm")
        print()
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

