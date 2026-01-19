"""
诊断静态特征为什么没有帮助训练效果
检查可能的问题：
1. 静态特征是否正确加载
2. 静态特征是否全为0或缺失
3. 静态特征维度是否匹配
4. SCMRL fusion是否正确接收静态特征
5. Alpha值是否在学习
"""

import torch
import numpy as np
import pandas as pd
import os
import sys

def check_static_features_loading(static_csv_path, dataset_loader):
    """检查静态特征是否正确加载"""
    print("=" * 80)
    print("1. 检查静态特征加载")
    print("=" * 80)
    
    if not os.path.exists(static_csv_path):
        print(f"❌ 静态特征CSV文件不存在: {static_csv_path}")
        return False
    
    # 检查数据集中的静态特征
    if not hasattr(dataset_loader, 'static_numeric_features'):
        print("❌ 数据集中没有 static_numeric_features 属性")
        return False
    
    static_features = getattr(dataset_loader, 'static_numeric_features', {})
    lulc_indices = getattr(dataset_loader, 'lulc_indices', {})
    
    print(f"✓ 静态特征字典大小: {len(static_features)}")
    print(f"✓ LULC索引字典大小: {len(lulc_indices)}")
    
    # 检查几个样本
    sample_count = 0
    zero_count = 0
    for pid, feat in list(static_features.items())[:10]:
        sample_count += 1
        if isinstance(feat, np.ndarray):
            if np.allclose(feat, 0):
                zero_count += 1
            print(f"  样本 {pid}: 特征维度={feat.shape}, 是否全0={np.allclose(feat, 0)}, "
                  f"均值={feat.mean():.4f}, 标准差={feat.std():.4f}")
        else:
            print(f"  样本 {pid}: 特征类型={type(feat)}")
    
    if zero_count == sample_count:
        print(f"⚠️  警告: 检查的样本中，{zero_count}/{sample_count} 的静态特征全为0")
    
    return True


def check_model_static_branch(model):
    """检查模型中的静态分支"""
    print("\n" + "=" * 80)
    print("2. 检查模型静态分支")
    print("=" * 80)
    
    if not hasattr(model, 'static_branch'):
        print("❌ 模型没有 static_branch 属性")
        return False
    
    if model.static_branch is None:
        print("❌ 模型的 static_branch 为 None")
        return False
    
    print(f"✓ StaticBranch 存在")
    print(f"  - numeric_dim: {model.static_branch.numeric_dim}")
    print(f"  - lulc_classes: {model.static_branch.lulc_classes}")
    print(f"  - lulc_embed_dim: {model.static_branch.lulc_embed_dim}")
    print(f"  - hidden: {model.static_branch.hidden}")
    print(f"  - lulc_embedding: {model.static_branch.lulc_embedding is not None}")
    
    # 检查SCMRL fusion
    if hasattr(model, 'use_scmrl_fusion') and model.use_scmrl_fusion:
        print(f"\n✓ 使用 SCMRL Fusion")
        if hasattr(model, 'fusion'):
            print(f"  - fusion 模块存在")
            if hasattr(model.fusion, 'climate_static_fusion'):
                if model.fusion.climate_static_fusion is not None:
                    print(f"  - climate_static_fusion 存在")
                    print(f"    - climate_dim: {model.fusion.climate_static_fusion.climate_dim}")
                    print(f"    - weak_dim: {model.fusion.climate_static_fusion.weak_dim}")
                else:
                    print(f"  ❌ climate_static_fusion 为 None")
            else:
                print(f"  ❌ fusion 没有 climate_static_fusion 属性")
        else:
            print(f"  ❌ fusion 模块不存在")
    else:
        print(f"⚠️  未使用 SCMRL Fusion")
    
    return True


def check_forward_pass(model, sample_batch):
    """检查前向传播中静态特征的处理"""
    print("\n" + "=" * 80)
    print("3. 检查前向传播")
    print("=" * 80)
    
    model.eval()
    with torch.no_grad():
        # 解包输入
        if len(sample_batch) >= 4:
            raster_stack = sample_batch[0]
            ts_features = sample_batch[1]
            static_features = sample_batch[2]
            lulc_indices = sample_batch[3] if len(sample_batch) >= 4 else None
            
            print(f"✓ 输入解包成功")
            print(f"  - raster_stack: {raster_stack.shape}")
            print(f"  - ts_features: {ts_features.shape}")
            print(f"  - static_features: {static_features.shape if static_features is not None else None}")
            print(f"  - lulc_indices: {lulc_indices.shape if lulc_indices is not None else None}")
            
            # 检查静态特征是否全为0
            if static_features is not None:
                static_sum = static_features.sum().item()
                static_mean = static_features.mean().item()
                static_std = static_features.std().item()
                is_all_zero = torch.allclose(static_features, torch.zeros_like(static_features))
                
                print(f"\n  静态特征统计:")
                print(f"  - 总和: {static_sum:.4f}")
                print(f"  - 均值: {static_mean:.4f}")
                print(f"  - 标准差: {static_std:.4f}")
                print(f"  - 是否全为0: {is_all_zero}")
                
                if is_all_zero:
                    print(f"  ❌ 警告: 静态特征全为0！")
                
                # 检查LULC索引
                if lulc_indices is not None:
                    unique_lulc = torch.unique(lulc_indices)
                    print(f"\n  LULC索引统计:")
                    print(f"  - 唯一值: {unique_lulc.tolist()}")
                    print(f"  - 唯一值数量: {len(unique_lulc)}")
                    if torch.all(lulc_indices == -1):
                        print(f"  ❌ 警告: 所有LULC索引都是-1（无效）！")
            
            # 检查StaticBranch输出
            if hasattr(model, 'static_branch') and model.static_branch is not None and static_features is not None:
                static_encoded = model.static_branch(static_features, lulc_indices)
                print(f"\n  StaticBranch 输出:")
                print(f"  - 形状: {static_encoded.shape}")
                print(f"  - 均值: {static_encoded.mean().item():.4f}")
                print(f"  - 标准差: {static_encoded.std().item():.4f}")
                print(f"  - 是否全为0: {torch.allclose(static_encoded, torch.zeros_like(static_encoded))}")
            
            # 检查SCMRL fusion
            if hasattr(model, 'use_scmrl_fusion') and model.use_scmrl_fusion:
                if hasattr(model, 'fusion') and hasattr(model.fusion, 'climate_static_fusion'):
                    if model.fusion.climate_static_fusion is not None:
                        # 获取alpha值
                        alpha_static = model.fusion.climate_static_fusion.alpha.item()
                        print(f"\n  SCMRL Fusion Alpha值:")
                        print(f"  - climate_static alpha: {alpha_static:.4f}")
                        if abs(alpha_static - 1.5) < 0.01:
                            print(f"  ⚠️  警告: Alpha值接近初始值1.5，可能没有学习")
                        elif alpha_static < 0.1:
                            print(f"  ⚠️  警告: Alpha值很小（{alpha_static:.4f}），模型认为静态特征是噪声")
                        else:
                            print(f"  ✓ Alpha值正常，模型在学习使用静态特征")
        else:
            print(f"❌ 输入批次格式不正确，期望至少4个元素，实际: {len(sample_batch)}")


def check_data_distribution(dataset_loader, num_samples=100):
    """检查静态特征的数据分布"""
    print("\n" + "=" * 80)
    print("4. 检查静态特征数据分布")
    print("=" * 80)
    
    if not hasattr(dataset_loader, 'static_numeric_features'):
        print("❌ 数据集中没有静态特征")
        return
    
    static_features = getattr(dataset_loader, 'static_numeric_features', {})
    lulc_indices = getattr(dataset_loader, 'lulc_indices', {})
    
    if len(static_features) == 0:
        print("❌ 静态特征字典为空")
        return
    
    # 收集样本
    samples = []
    lulc_samples = []
    count = 0
    for pid, feat in static_features.items():
        if count >= num_samples:
            break
        if isinstance(feat, np.ndarray):
            samples.append(feat)
            lulc_samples.append(lulc_indices.get(pid, -1))
            count += 1
    
    if len(samples) == 0:
        print("❌ 没有有效的静态特征样本")
        return
    
    samples_array = np.array(samples)
    lulc_array = np.array(lulc_samples)
    
    print(f"✓ 收集了 {len(samples)} 个样本")
    print(f"\n  数值特征统计:")
    print(f"  - 特征维度: {samples_array.shape[1]}")
    print(f"  - 均值: {samples_array.mean(axis=0)}")
    print(f"  - 标准差: {samples_array.std(axis=0)}")
    print(f"  - 最小值: {samples_array.min(axis=0)}")
    print(f"  - 最大值: {samples_array.max(axis=0)}")
    
    # 检查是否有常数特征（方差为0）
    variances = samples_array.var(axis=0)
    constant_features = np.where(variances < 1e-6)[0]
    if len(constant_features) > 0:
        print(f"\n  ⚠️  警告: 发现 {len(constant_features)} 个常数特征（方差<1e-6）: {constant_features}")
    
    print(f"\n  LULC索引统计:")
    unique_lulc, counts = np.unique(lulc_array, return_counts=True)
    print(f"  - 唯一值: {unique_lulc}")
    print(f"  - 分布: {dict(zip(unique_lulc, counts))}")
    if np.all(lulc_array == -1):
        print(f"  ❌ 警告: 所有LULC索引都是-1（无效）！")


def check_point_id_matching(dataset_loader, num_samples=20):
    """检查Point_ID匹配情况"""
    print("\n" + "=" * 80)
    print("5. 检查Point_ID匹配")
    print("=" * 80)
    
    static_features = getattr(dataset_loader, 'static_numeric_features', {})
    
    if len(static_features) == 0:
        print("❌ 静态特征字典为空")
        return
    
    matched_count = 0
    unmatched_count = 0
    
    for i in range(min(num_samples, len(dataset_loader))):
        try:
            l8_img_name = dataset_loader.l8_names[i]
            point_id = l8_img_name.split('_')[0]
            
            # 尝试匹配
            pid_key = str(point_id).replace('.0', '').strip()
            if pid_key in static_features:
                matched_count += 1
            else:
                # 尝试整数匹配
                try:
                    pid_int = int(float(point_id))
                    pid_key_int = str(pid_int)
                    if pid_key_int in static_features:
                        matched_count += 1
                    else:
                        unmatched_count += 1
                        if unmatched_count <= 5:
                            print(f"  ❌ 未匹配: Point_ID={point_id} (尝试了 {pid_key} 和 {pid_key_int})")
                except:
                    unmatched_count += 1
                    if unmatched_count <= 5:
                        print(f"  ❌ 未匹配: Point_ID={point_id} (无法转换为整数)")
        except Exception as e:
            unmatched_count += 1
            if unmatched_count <= 5:
                print(f"  ❌ 错误: {e}")
    
    print(f"\n  匹配统计:")
    print(f"  - 匹配成功: {matched_count}/{num_samples}")
    print(f"  - 匹配失败: {unmatched_count}/{num_samples}")
    print(f"  - 匹配率: {matched_count/num_samples*100:.1f}%")
    
    if unmatched_count > num_samples * 0.5:
        print(f"  ❌ 警告: 超过50%的样本无法匹配静态特征！")


def quick_diagnose(model, train_ds, train_dl, static_csv_path):
    """快速诊断静态特征问题（一键运行所有检查）"""
    print("\n" + "="*80)
    print("静态特征快速诊断")
    print("="*80)
    
    # 1. 检查数据加载
    print("\n[1] 检查静态特征加载...")
    if not check_static_features_loading(static_csv_path, train_ds):
        print("  ❌ 静态特征加载失败")
        return
    
    # 2. 检查模型
    print("\n[2] 检查模型静态分支...")
    if not check_model_static_branch(model):
        print("  ❌ 模型静态分支配置有问题")
        return
    
    # 3. 检查数据分布
    print("\n[3] 检查数据分布...")
    check_data_distribution(train_ds, num_samples=100)
    
    # 4. 检查Point_ID匹配
    print("\n[4] 检查Point_ID匹配...")
    check_point_id_matching(train_ds, num_samples=100)
    
    # 5. 检查前向传播
    print("\n[5] 检查前向传播...")
    try:
        sample_batch = next(iter(train_dl))
        check_forward_pass(model, sample_batch[0])
    except Exception as e:
        print(f"  ❌ 无法检查前向传播: {e}")
    
    print("\n" + "="*80)
    print("诊断完成")
    print("="*80 + "\n")


def main():
    """主函数：运行所有诊断"""
    print("=" * 80)
    print("静态特征诊断工具")
    print("=" * 80)
    
    print("\n使用方法:")
    print("=" * 80)
    print("""
# 在 train.py 中，创建模型和数据加载器后，添加：

from diagnose_static_features import quick_diagnose

# 一键运行所有诊断
if USE_STATIC_FEATURES:
    quick_diagnose(model, train_ds, train_dl, STATIC_CSV)
""")


if __name__ == "__main__":
    main()

