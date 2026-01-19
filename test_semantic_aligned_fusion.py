"""
测试 SemanticAlignedFusion 在 SoilNet 中的集成

展示如何使用基于 S-CMRL 的语义对齐跨模态残差融合模块
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from soilnet.submodules.semantic_aligned_fusion import (
    SemanticAlignedFusion,
    SemanticAlignmentLoss
)


def test_integration_with_soilnet():
    """测试与 SoilNet 的集成"""
    print("=" * 70)
    print("测试 SemanticAlignedFusion 与 SoilNet 的集成")
    print("=" * 70)
    
    # 模拟 SoilNetLSTM 的特征维度
    batch_size = 4
    climate_dim = 128  # LSTM/Transformer 输出维度
    visual_dim = 384   # ViT-CoMer 输出维度（或 1024 for ResNet）
    static_dim = 64    # Static 分支输出维度
    
    # 创建融合模块
    fusion = SemanticAlignedFusion(
        climate_dim=climate_dim,
        visual_dim=visual_dim,
        static_dim=static_dim,
        num_heads=8,
        alpha_init=1.5,
        learnable_alpha=True
    )
    
    # 创建语义对齐损失
    alignment_loss_fn = SemanticAlignmentLoss(temperature=0.07)
    
    # 模拟前向传播
    climate_feat = torch.randn(batch_size, climate_dim)
    visual_feat = torch.randn(batch_size, visual_dim)
    static_feat = torch.randn(batch_size, static_dim)
    
    # 融合特征
    fused_feat = fusion(climate_feat, visual_feat, static_feat)
    
    print(f"\n输入特征:")
    print(f"  Climate: {climate_feat.shape}")
    print(f"  Visual:  {visual_feat.shape}")
    print(f"  Static:  {static_feat.shape}")
    print(f"\n融合后特征: {fused_feat.shape}")
    
    # 计算语义对齐损失（仅用于 Climate-Visual）
    align_loss = alignment_loss_fn(climate_feat, visual_feat)
    print(f"\n语义对齐损失: {align_loss.item():.4f}")
    
    # 获取 alpha 值（用于监控）
    alphas = fusion.get_alpha_values()
    print(f"\nAlpha 参数值（监控弱模态贡献）:")
    for key, value in alphas.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n[OK] 集成测试通过!")
    return fusion, alignment_loss_fn


def test_training_loop_example():
    """展示在训练循环中的使用示例"""
    print("\n" + "=" * 70)
    print("训练循环使用示例")
    print("=" * 70)
    
    batch_size = 4
    climate_dim = 128
    visual_dim = 384
    static_dim = 64
    
    # 创建模型组件
    fusion = SemanticAlignedFusion(
        climate_dim=climate_dim,
        visual_dim=visual_dim,
        static_dim=static_dim
    )
    
    # 回归头（输出维度与 Climate 相同）
    regressor = nn.Linear(climate_dim, 1)
    
    # 语义对齐损失
    alignment_loss_fn = SemanticAlignmentLoss(temperature=0.07)
    
    # 主损失函数（MSE）
    main_loss_fn = nn.MSELoss()
    
    # 模拟一个训练步骤
    climate_feat = torch.randn(batch_size, climate_dim)
    visual_feat = torch.randn(batch_size, visual_dim)
    static_feat = torch.randn(batch_size, static_dim)
    y_true = torch.randn(batch_size, 1)
    
    # 前向传播
    fused_feat = fusion(climate_feat, visual_feat, static_feat)
    y_pred = regressor(fused_feat)
    
    # 计算损失
    main_loss = main_loss_fn(y_pred, y_true)
    align_loss = alignment_loss_fn(climate_feat, visual_feat)
    
    # 总损失（可以调整权重）
    lambda_align = 0.1  # 对齐损失的权重
    total_loss = main_loss + lambda_align * align_loss
    
    print(f"\n主损失 (MSE): {main_loss.item():.4f}")
    print(f"对齐损失:     {align_loss.item():.4f}")
    print(f"总损失:       {total_loss.item():.4f}")
    
    # 反向传播（示例）
    total_loss.backward()
    
    # 监控 alpha 值
    alphas = fusion.get_alpha_values()
    print(f"\nAlpha 值（训练后）:")
    for key, value in alphas.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n[OK] 训练循环示例完成!")
    
    return {
        'main_loss': main_loss.item(),
        'align_loss': align_loss.item(),
        'total_loss': total_loss.item(),
        'alphas': alphas
    }


def print_usage_guide():
    """打印使用指南"""
    print("\n" + "=" * 70)
    print("使用指南")
    print("=" * 70)
    
    guide = '''
# ========== 方法 1: 在 SoilNetLSTM 中集成 ==========

from soilnet.submodules.semantic_aligned_fusion import (
    SemanticAlignedFusion,
    SemanticAlignmentLoss
)

class SoilNetLSTMWithSCMRL(SoilNetLSTM):
    def __init__(self, ..., use_scmrl_fusion=True, **kwargs):
        super().__init__(...)
        
        if use_scmrl_fusion:
            # 创建 S-CMRL 融合模块
            self.fusion = SemanticAlignedFusion(
                climate_dim=lstm_out,      # 例如 128
                visual_dim=regresor_input_from_cnn,  # 例如 384 或 1024
                static_dim=static_dim,      # 如果有静态特征
                num_heads=8,
                alpha_init=1.5,
                learnable_alpha=True
            )
            
            # 语义对齐损失
            self.alignment_loss_fn = SemanticAlignmentLoss(temperature=0.07)
            
            # 回归头（输入维度与 Climate 相同）
            self.reg = nn.Linear(lstm_out, 1)
        else:
            # 使用原有的 MultiHeadRegressor
            self.reg = MultiHeadRegressor(...)
    
    def forward(self, input_raster_ts, ...):
        # 提取特征
        cnn_features = self.cnn(raster_stack)      # [B, visual_dim]
        climate_features = self.lstm(ts_features)   # [B, climate_dim]
        static_features = ...                       # [B, static_dim] (可选)
        
        if self.use_scmrl_fusion:
            # 使用 S-CMRL 融合
            fused_features = self.fusion(
                climate_features,
                cnn_features,
                static_features
            )  # [B, climate_dim]
            
            # 回归预测
            output = self.reg(fused_features)
        else:
            # 原有方法：简单 concat
            output = self.reg(cnn_features, climate_features, ...)
        
        return output

# ========== 方法 2: 在训练循环中使用对齐损失 ==========

# 在 train_utils.py 的 train_step 中：

def train_step(model, batch, optimizer, loss_fn, alignment_loss_fn=None, lambda_align=0.1):
    # ... 前向传播 ...
    y_pred = model(inputs)
    
    # 主损失
    main_loss = loss_fn(y_pred, y_true)
    
    # 语义对齐损失（如果使用 S-CMRL）
    total_loss = main_loss
    if alignment_loss_fn is not None:
        # 获取中间特征（需要在模型中返回）
        climate_feat = model.get_climate_feat()
        visual_feat = model.get_visual_feat()
        align_loss = alignment_loss_fn(climate_feat, visual_feat)
        total_loss = main_loss + lambda_align * align_loss
    
    # 反向传播
    total_loss.backward()
    optimizer.step()
    
    return {
        'loss': main_loss.item(),
        'align_loss': align_loss.item() if alignment_loss_fn else 0.0,
        'total_loss': total_loss.item()
    }

# ========== 方法 3: 监控 Alpha 值 ==========

# 在训练过程中定期打印 alpha 值，观察弱模态的贡献：

if epoch % 10 == 0:
    alphas = model.fusion.get_alpha_values()
    print(f"Epoch {epoch} Alpha values:")
    for key, value in alphas.items():
        print(f"  {key}: {value:.4f}")
    
    # 如果 alpha 变得很小（接近 0），说明模型认为弱模态是噪声
    # 如果 alpha 稳定在正值，说明模型成功提取到了互补信息

# ========== 验证修改是否有效 ==========

# 1. 观察 Alpha 值的变化：
#    - 如果 alpha -> 0：模型认为弱模态是噪声，自动屏蔽（成功）
#    - 如果 alpha 稳定在正值：模型成功提取到互补信息（成功）

# 2. 观察 Alignment Loss 的下降：
#    - 如果对齐损失下降：弱模态学习到了与强模态相关的语义（成功）
#    - 如果对齐损失不下降：可能需要调整温度参数或损失权重

# 3. 对比性能指标：
#    - 使用 S-CMRL 融合 vs 简单 concat
#    - 如果 S-CMRL 性能更好或至少不掉点，说明有效
'''
    print(guide)


if __name__ == '__main__':
    print("=" * 70)
    print("SemanticAlignedFusion 集成测试")
    print("基于 S-CMRL (Semantic-Alignment Cross-Modal Residual Learning)")
    print("=" * 70)
    
    try:
        # 测试集成
        fusion, alignment_loss_fn = test_integration_with_soilnet()
        
        # 测试训练循环示例
        results = test_training_loop_example()
        
        # 打印使用指南
        print_usage_guide()
        
        print("\n" + "=" * 70)
        print("[SUCCESS] 所有测试通过!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)








