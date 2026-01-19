import torch
import torch.nn as nn
import sys
import os

# 添加路径
sys.path.append('submodules')

def test_basic_components():
    """测试基本组件"""
    print("=== 测试基本组件 ===")
    
    try:
        # 测试增强气候Transformer
        from enhanced_climate_transformer import EnhancedClimateTransformer
        print("✅ 增强气候Transformer导入成功")
        
        # 测试交叉模态融合
        from cross_modal_fusion import CrossModalAttention, GatedFusion, HierarchicalFusion
        print("✅ 交叉模态融合模块导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 组件导入失败: {e}")
        return False

def test_enhanced_climate():
    """测试增强气候模块"""
    print("\n=== 测试增强气候模块 ===")
    
    try:
        from enhanced_climate_transformer import EnhancedClimateTransformer
        
        model = EnhancedClimateTransformer(
            feat_dim=10, d_model=128, num_layers=2, seq_len=61
        )
        
        x = torch.randn(2, 61, 10)
        out = model(x)
        
        print(f"✅ 增强气候模块测试成功")
        print(f"   输入: {x.shape}")
        print(f"   输出: {out.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 增强气候模块测试失败: {e}")
        return False

def test_fusion_modules():
    """测试融合模块"""
    print("\n=== 测试融合模块 ===")
    
    try:
        from cross_modal_fusion import CrossModalAttention, GatedFusion, HierarchicalFusion
        
        # 测试交叉模态注意力
        cross_modal = CrossModalAttention(128, 128, 64)
        vit_feat = torch.randn(2, 16, 128)
        climate_feat = torch.randn(2, 16, 128)
        out = cross_modal(vit_feat, climate_feat)
        print(f"✅ 交叉模态注意力测试成功: {out.shape}")
        
        # 测试门控融合
        gated = GatedFusion(128, 128, 64)
        out = gated(vit_feat, climate_feat)
        print(f"✅ 门控融合测试成功: {out.shape}")
        
        # 测试分层融合
        hierarchical = HierarchicalFusion(128, 128, 64)
        out = hierarchical(vit_feat, climate_feat)
        print(f"✅ 分层融合测试成功: {out.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 融合模块测试失败: {e}")
        return False

def test_hybrid_vit():
    """测试HybridViT模块"""
    print("\n=== 测试HybridViT模块 ===")
    
    try:
        from vit_hybrid import HybridViT
        
        model = HybridViT(
            img_size=128,
            patch_size=16,
            in_chans=12,
            num_classes=128,
            embed_dim=128,
            depths=[2, 2, 2],
            num_heads=4,
            mlp_ratio=4,
            num_paths=2
        )
        
        x = torch.randn(2, 12, 128, 128)
        out = model(x)
        
        print(f"✅ HybridViT测试成功")
        print(f"   输入: {x.shape}")
        print(f"   输出: {out.shape}")
        
        return True
    except Exception as e:
        print(f"❌ HybridViT测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始逐步测试...\n")
    
    results = []
    results.append(test_basic_components())
    results.append(test_enhanced_climate())
    results.append(test_fusion_modules())
    results.append(test_hybrid_vit())
    
    print(f"\n=== 测试结果汇总 ===")
    success_count = sum(results)
    total_count = len(results)
    
    if success_count == total_count:
        print(f"🎉 所有组件测试成功！ ({success_count}/{total_count})")
        print("可以开始集成测试了！")
    else:
        print(f"⚠️  部分组件测试失败 ({success_count}/{total_count})")
        print("请检查失败组件的错误信息") 