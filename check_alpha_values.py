"""
查看模型中的 Alpha 值

使用方法：
1. 查看训练中的模型：
   python check_alpha_values.py --model_path results/RUN_xxx_best.pth.tar

2. 查看当前加载的模型（在训练脚本中）：
   在训练循环中添加：
   from check_alpha_values import print_alpha_values
   print_alpha_values(model)
"""

import torch
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_alpha_values(model):
    """
    打印模型中的 Alpha 值
    
    Args:
        model: PyTorch 模型实例
    """
    if not hasattr(model, 'use_scmrl_fusion') or not model.use_scmrl_fusion:
        print("[WARN] 模型未启用 S-CMRL 融合")
        return None
    
    if not hasattr(model, 'fusion'):
        print("[WARN] 模型中未找到 fusion 模块")
        return None
    
    try:
        alphas = model.fusion.get_alpha_values()
        print("\n" + "=" * 70)
        print("Alpha 值（控制弱模态贡献）")
        print("=" * 70)
        for key, value in alphas.items():
            print(f"  {key:20s}: {value:8.4f}")
            # 解释 alpha 值的含义
            if 'visual' in key.lower():
                if value < 0.1:
                    print(f"    -> Visual 分支贡献很小（接近噪声，已被屏蔽）")
                elif value < 0.5:
                    print(f"    -> Visual 分支贡献较小")
                elif value < 1.5:
                    print(f"    -> Visual 分支贡献正常")
                else:
                    print(f"    -> Visual 分支贡献较大")
            elif 'static' in key.lower():
                if value < 0.1:
                    print(f"    -> Static 分支贡献很小（接近噪声，已被屏蔽）")
                elif value < 0.5:
                    print(f"    -> Static 分支贡献较小")
                elif value < 1.5:
                    print(f"    -> Static 分支贡献正常")
                else:
                    print(f"    -> Static 分支贡献较大")
        print("=" * 70 + "\n")
        return alphas
    except Exception as e:
        print(f"[ERROR] 获取 Alpha 值失败: {e}")
        return None


def load_model_and_check(checkpoint_path):
    """
    从 checkpoint 加载模型并查看 Alpha 值
    
    Args:
        checkpoint_path: checkpoint 文件路径
    """
    print(f"加载模型: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 尝试从 checkpoint 中获取模型结构信息
        print("\n检查 checkpoint 内容...")
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 查找 alpha 相关的参数
        alpha_keys = [k for k in state_dict.keys() if 'alpha' in k.lower() and 'fusion' in k.lower()]
        
        if alpha_keys:
            print(f"\n找到 {len(alpha_keys)} 个 Alpha 参数:")
            print("=" * 70)
            for key in alpha_keys:
                value = state_dict[key].item() if state_dict[key].numel() == 1 else state_dict[key]
                print(f"  {key:50s}: {value}")
            print("=" * 70)
        else:
            print("\n[WARN] 未在 checkpoint 中找到 Alpha 参数")
            print("可能的原因：")
            print("  1. 模型未使用 S-CMRL 融合")
            print("  2. Alpha 参数名称不匹配")
            print("\n尝试查找所有 fusion 相关参数...")
            fusion_keys = [k for k in state_dict.keys() if 'fusion' in k.lower()]
            if fusion_keys:
                print(f"找到 {len(fusion_keys)} 个 fusion 相关参数（前10个）:")
                for key in fusion_keys[:10]:
                    print(f"  {key}")
        
    except Exception as e:
        print(f"[ERROR] 加载 checkpoint 失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='查看模型中的 Alpha 值')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型 checkpoint 路径（.pth.tar 文件）')
    parser.add_argument('--list_all', action='store_true',
                        help='列出 checkpoint 中所有 fusion 相关的参数')
    
    args = parser.parse_args()
    
    if args.model_path:
        load_model_and_check(args.model_path)
    else:
        print("使用方法:")
        print("  python check_alpha_values.py --model_path results/RUN_xxx_best.pth.tar")
        print("\n或者在训练脚本中使用:")
        print("  from check_alpha_values import print_alpha_values")
        print("  print_alpha_values(model)")


if __name__ == '__main__':
    main()





















