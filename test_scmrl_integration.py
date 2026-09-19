"""
快速测试 S-CMRL 融合在 train.py 中的集成
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_scmrl_integration():
    """测试 S-CMRL 融合模块是否正确集成"""
    print("=" * 70)
    print("测试 S-CMRL 融合集成")
    print("=" * 70)
    
    # 测试导入
    try:
        from soilnet.submodules.semantic_aligned_fusion import (
            SemanticAlignedFusion,
            SemanticAlignmentLoss
        )
        print("[OK] 成功导入 S-CMRL 模块")
    except Exception as e:
        print(f"[FAIL] 导入失败: {e}")
        return False
    
    # 测试模型创建
    try:
        from soilnet.soil_net import SoilNetLSTM
        
        model = SoilNetLSTM(
            cnn_in_channels=12,
            regresor_input_from_cnn=384,
            lstm_n_features=17,
            seq_len=60,
            rnn_arch="Transformer",
            use_scmrl_fusion=True,
            scmrl_alpha_init=1.5,
            scmrl_temperature=0.07,
            static_dim=64
        )
        print("[OK] 成功创建带 S-CMRL 融合的模型")
        
        # 检查融合模块是否存在
        if hasattr(model, 'fusion') and hasattr(model, 'alignment_loss_fn'):
            print("[OK] 融合模块和对齐损失函数已创建")
        else:
            print("[FAIL] 融合模块或对齐损失函数未创建")
            return False
        
        # 测试前向传播
        import torch
        x_img = torch.randn(4, 12, 64, 64)
        x_climate = torch.randn(4, 60, 17)
        x_static = torch.randn(4, 64)
        
        output = model((x_img, x_climate, x_static))
        print(f"[OK] 前向传播成功，输出形状: {output.shape}")
        
        # 检查中间特征是否保存
        if hasattr(model, '_last_climate_feat') and hasattr(model, '_last_visual_feat'):
            print("[OK] 中间特征已保存（可用于对齐损失）")
        else:
            print("[WARN] 中间特征未保存")
        
        # 测试对齐损失
        if hasattr(model, 'alignment_loss_fn'):
            align_loss = model.alignment_loss_fn(
                model._last_climate_feat,
                model._last_visual_feat
            )
            print(f"[OK] 对齐损失计算成功: {align_loss.item():.4f}")
        
        # 测试 alpha 值获取
        if hasattr(model, 'fusion'):
            alphas = model.fusion.get_alpha_values()
            print(f"[OK] Alpha 值: {alphas}")
        
    except Exception as e:
        print(f"[FAIL] 模型创建或测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("[SUCCESS] 所有集成测试通过!")
    print("=" * 70)
    print("\n使用方法:")
    print("python train.py -e newfusion -d CHINA -w 6 -cnn ViT-CoMer -rnn Transformer")
    print("                -trbs 32 -ne 60 -lr 0.0001 -ls step -stm -srtm -static")
    print("                --use_scmrl_fusion")
    print("\n可选参数:")
    print("  --scmrl_alpha_init 1.5      # Alpha 初始值（默认 1.5）")
    print("  --scmrl_lambda_align 0.1   # 对齐损失权重（默认 0.1）")
    print("  --scmrl_temperature 0.07   # 对齐损失温度（默认 0.07）")
    
    return True

if __name__ == '__main__':
    success = test_scmrl_integration()
    sys.exit(0 if success else 1)
