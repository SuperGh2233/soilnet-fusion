"""
Cross-Attention 退化诊断脚本

检测以下退化模式：
1. 序列长度退化：当输入为 2D 时，seq_len=1 使得 attention 退化为线性变换
2. Alpha 退化：可学习的 alpha 趋近于 0，弱模态被完全屏蔽
3. 注意力权重退化：注意力分布变成均匀分布（无区分能力）
4. 梯度退化：cross-attention 的梯度消失或爆炸
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'soilnet'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'soilnet', 'submodules'))

from submodules.semantic_aligned_fusion import (
    CrossModalResidualBlock,
    SemanticAlignedFusion,
    SemanticAlignedFusionParallel,
    SemanticAlignmentLoss
)
from submodules.cross_modal_fusion import CrossModalAttention, GatedFusion


def diagnose_seq_len_issue():
    """诊断问题1：序列长度为1时的退化"""
    print("=" * 70)
    print("诊断1: 序列长度退化分析")
    print("=" * 70)

    batch_size = 4
    climate_dim = 128
    visual_dim = 768
    num_heads = 8

    # 创建 S-CMRL 融合模块
    block = CrossModalResidualBlock(
        climate_dim=climate_dim,
        weak_dim=visual_dim,
        num_heads=num_heads,
        alpha_init=1.5,
        learnable_alpha=True
    )

    # 场景A：2D 输入（典型情况，来自 CNN/RNN 的 flat 特征）
    climate_2d = torch.randn(batch_size, climate_dim)
    visual_2d = torch.randn(batch_size, visual_dim)

    # 场景B：3D 输入（序列情况）
    seq_len = 16
    climate_3d = torch.randn(batch_size, seq_len, climate_dim)
    visual_3d = torch.randn(batch_size, seq_len, visual_dim)

    print(f"\n场景A (2D 输入 - 模型实际使用情况):")
    print(f"  Climate shape: {climate_2d.shape} -> unsqueeze -> [B, 1, {climate_dim}]")
    print(f"  Visual shape:  {visual_2d.shape} -> unsqueeze -> [B, 1, {visual_dim}]")
    print(f"  Attention 中每个 head 只有 1 个 token 可以 attend")
    print(f"  ⚠️  Softmax([x]) = 1.0 恒成立，注意力权重无区分能力")

    # 前向传播对比
    block.eval()
    with torch.no_grad():
        out_2d = block(climate_2d, visual_2d)
        out_3d = block(climate_3d, visual_3d)

        # 捕获注意力权重
        climate_2d_unsq = climate_2d.unsqueeze(1)
        visual_2d_proj = block.weak_proj(visual_2d.unsqueeze(1))
        attn_out_2d, attn_weights_2d = block.cross_attention(
            query=climate_2d_unsq,
            key=visual_2d_proj,
            value=visual_2d_proj
        )

        climate_3d_proj = climate_3d
        visual_3d_proj = block.weak_proj(visual_3d)
        attn_out_3d, attn_weights_3d = block.cross_attention(
            query=climate_3d,
            key=visual_3d_proj,
            value=visual_3d_proj
        )

    print(f"\n  2D 注意力权重形状: {attn_weights_2d.shape}")
    if attn_weights_2d.dim() == 3:
        print(f"  2D 注意力权重 (前2个样本):")
        print(f"    {attn_weights_2d[:2, :5, :5].numpy()}")
    elif attn_weights_2d.dim() == 4:
        print(f"  2D 注意力权重 (前2个样本, 前2个head):")
        print(f"    {attn_weights_2d[:2, :2, :5].numpy()}")
    print(f"    ⚠️  全部为 1.0 (softmax 对单元素恒为1)")

    print(f"\n  3D 注意力权重形状: {attn_weights_3d.shape}")
    # PyTorch MultiheadAttention 返回的 attn_weights 形状: [batch_size, seq_len, seq_len] (batch_first=True)
    if attn_weights_3d.dim() == 3:
        print(f"  3D 注意力权重 (前1个样本, 前5个query, 前10个key):")
        print(f"    {attn_weights_3d[0, :5, :10].numpy()}")
    elif attn_weights_3d.dim() == 4:
        print(f"  3D 注意力权重 (前1个样本, 前1个head, 前5个query):")
        print(f"    {attn_weights_3d[0, 0, :5, :10].numpy()}")
    print(f"    ✓ 注意力分布有区分度")

    # 验证：2D 情况下 attention 输出是否等于线性变换
    # 如果 attention(Q, K, V) 当 K=V 只有一个 token 时，输出 = V (经过投影)
    print(f"\n  验证: 2D attention 输出 ≈ linear(visual_proj)?")
    # attention output for single token: softmax(q·k^T/sqrt(d)) * v = 1.0 * v = v
    # 所以 attn_out_2d ≈ visual_2d_proj (经过投影后的 visual)
    diff = (attn_out_2d - visual_2d_proj).abs().mean().item()
    print(f"    差异: {diff:.6f}")
    if diff < 1e-5:
        print(f"    ⚠️  确认退化: attention 输出 ≈ 线性投影，无真正的注意力机制")
    else:
        print(f"    注意力机制有作用 (可能因为多头投影)")


def diagnose_alpha_collapse():
    """诊断问题2: Alpha 参数退化"""
    print("\n" + "=" * 70)
    print("诊断2: Alpha 参数退化分析")
    print("=" * 70)

    # 检查是否有训练好的模型
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"  ⚠️  未找到 results/ 目录，跳过模型 alpha 检查")
        print(f"  请先运行训练，或手动指定模型路径")
        return

    # 查找包含 SCMRL 的模型文件
    model_files = [f for f in os.listdir(results_dir) if f.endswith('.pth.tar')]
    scmrl_models = [f for f in model_files if 'scmrl' in f.lower() or 'SCMRL' in f]

    if not scmrl_models:
        print(f"  ⚠️  未找到包含 SCMRL 的模型文件")
        print(f"  找到的模型文件: {model_files[:5]}")
        return

    print(f"  找到 {len(scmrl_models)} 个 SCMRL 模型文件")

    for model_file in scmrl_models[:3]:  # 最多检查3个
        model_path = os.path.join(results_dir, model_file)
        print(f"\n  检查模型: {model_file}")

        try:
            checkpoint = torch.load(model_path, map_location='cpu')

            # 提取 state_dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # 查找 alpha 参数
            alpha_keys = [k for k in state_dict.keys() if 'alpha' in k.lower()]

            if not alpha_keys:
                print(f"    ⚠️  未找到 alpha 参数")
                continue

            for key in alpha_keys:
                alpha_val = state_dict[key].item()
                print(f"    {key}: {alpha_val:.6f}")

                if abs(alpha_val) < 0.01:
                    print(f"    ⚠️  Alpha 接近 0！弱模态贡献被完全屏蔽")
                elif abs(alpha_val) < 0.1:
                    print(f"    ⚠️  Alpha 较小，弱模态贡献很弱")
                elif alpha_val > 5.0:
                    print(f"    ⚠️  Alpha 过大，可能不稳定")
                else:
                    print(f"    ✓ Alpha 值正常范围")

        except Exception as e:
            print(f"    加载失败: {e}")


def diagnose_attention_weights_from_model():
    """诊断问题3: 从训练好的模型中提取注意力权重"""
    print("\n" + "=" * 70)
    print("诊断3: 注意力权重分析 (需要训练好的模型)")
    print("=" * 70)

    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"  ⚠️  未找到 results/ 目录")
        return

    model_files = [f for f in os.listdir(results_dir) if f.endswith('.pth.tar')]

    if not model_files:
        print(f"  ⚠️  未找到模型文件")
        return

    print(f"  找到 {len(model_files)} 个模型文件")
    print(f"  要分析注意力权重，需要在 forward 时捕获 attn_weights")
    print(f"  建议: 在 CrossModalResidualBlock.forward() 中添加:")
    print(f"    self._last_attn_weights = attn_weights")


def diagnose_gradient_flow():
    """诊断问题4: 梯度流分析"""
    print("\n" + "=" * 70)
    print("诊断4: 梯度流分析")
    print("=" * 70)

    batch_size = 4
    climate_dim = 128
    visual_dim = 768

    # 创建模型
    block = CrossModalResidualBlock(
        climate_dim=climate_dim,
        weak_dim=visual_dim,
        num_heads=8,
        alpha_init=1.5,
        learnable_alpha=True
    )

    # 2D 输入
    climate_feat = torch.randn(batch_size, climate_dim, requires_grad=True)
    visual_feat = torch.randn(batch_size, visual_dim, requires_grad=True)

    # 前向传播
    output = block(climate_feat, visual_feat)
    loss = output.sum()
    loss.backward()

    print(f"\n  2D 输入梯度分析:")

    # 检查 alpha 梯度
    if block.alpha.grad is not None:
        alpha_grad = block.alpha.grad.item()
        print(f"    alpha 梯度: {alpha_grad:.6f}")
        if abs(alpha_grad) < 1e-6:
            print(f"    ⚠️  Alpha 梯度接近 0，alpha 可能无法有效学习")
        else:
            print(f"    ✓ Alpha 梯度正常")
    else:
        print(f"    ⚠️  Alpha 无梯度")

    # 检查弱模态投影层梯度
    weak_proj_grad_norm = block.weak_proj.weight.grad.norm().item() if block.weak_proj.weight.grad is not None else 0
    print(f"    weak_proj 梯度范数: {weak_proj_grad_norm:.6f}")

    if weak_proj_grad_norm < 1e-6:
        print(f"    ⚠️  弱模态投影层梯度消失，无法学习有效投影")

    # 检查 cross_attention 梯度
    attn_grad_norm = block.cross_attention.out_proj.weight.grad.norm().item() if block.cross_attention.out_proj.weight.grad is not None else 0
    print(f"    cross_attention 梯度范数: {attn_grad_norm:.6f}")

    if attn_grad_norm < 1e-6:
        print(f"    ⚠️  Cross-attention 梯度消失")

    # 对比：3D 输入
    print(f"\n  3D 输入梯度分析 (对比):")
    seq_len = 16
    climate_3d = torch.randn(batch_size, seq_len, climate_dim, requires_grad=True)
    visual_3d = torch.randn(batch_size, seq_len, visual_dim, requires_grad=True)

    output_3d = block(climate_3d, visual_3d)
    loss_3d = output_3d.sum()
    loss_3d.backward()

    if block.alpha.grad is not None:
        alpha_grad_3d = block.alpha.grad.item()
        print(f"    alpha 梯度: {alpha_grad_3d:.6f}")

    weak_proj_grad_norm_3d = block.weak_proj.weight.grad.norm().item() if block.weak_proj.weight.grad is not None else 0
    print(f"    weak_proj 梯度范数: {weak_proj_grad_norm_3d:.6f}")


def diagnose_semantic_alignment_loss():
    """诊断问题5: 语义对齐损失分析"""
    print("\n" + "=" * 70)
    print("诊断5: 语义对齐损失分析")
    print("=" * 70)

    batch_size = 32
    climate_dim = 128
    visual_dim = 768

    alignment_loss = SemanticAlignmentLoss(temperature=0.07)

    # 场景A: 随机特征（未训练）
    climate_random = torch.randn(batch_size, climate_dim)
    visual_random = torch.randn(batch_size, visual_dim)
    loss_random = alignment_loss(climate_random, visual_random)

    print(f"\n  场景A (随机特征):")
    print(f"    对齐损失: {loss_random.item():.4f}")
    print(f"    预期: -log(1/B) = {np.log(batch_size):.4f} (均匀分布)")

    # 场景B: 完美对齐（相同特征）
    climate_same = torch.randn(batch_size, climate_dim)
    visual_same = climate_same.clone()  # 完全相同
    loss_same = alignment_loss(climate_same, visual_same)

    print(f"\n  场景B (完美对齐):")
    print(f"    对齐损失: {loss_same.item():.4f}")
    print(f"    预期: 接近 0")

    # 场景C: 完全不对齐（随机独立特征，近似正交）
    climate_orth = torch.randn(batch_size, climate_dim)
    visual_orth = torch.randn(batch_size, visual_dim)
    loss_orth = alignment_loss(climate_orth, visual_orth)

    print(f"\n  场景C (近似正交):")
    print(f"    对齐损失: {loss_orth.item():.4f}")
    print(f"    预期: 接近 {np.log(batch_size):.4f}")

    # 分析维度不匹配时的处理
    print(f"\n  维度不匹配处理分析:")
    print(f"    Climate dim: {climate_dim}, Visual dim: {visual_dim}")
    print(f"    当 dim 不匹配时，SemanticAlignmentLoss 使用 group_avg_pool 降维")
    print(f"    这可能导致信息损失")


def diagnose_real_model_forward():
    """诊断问题6: 在真实模型中检查 cross-attention 行为"""
    print("\n" + "=" * 70)
    print("诊断6: 真实模型前向传播分析")
    print("=" * 70)

    # 模拟真实模型的特征维度
    batch_size = 4
    cnn_output_dim = 768  # ViT output
    lstm_out = 128  # Transformer/RNN output
    num_climate_features = 13  # 气候变量数

    # 创建 S-CMRL 融合模块
    fusion = SemanticAlignedFusion(
        climate_dim=lstm_out,
        visual_dim=cnn_output_dim,
        num_heads=8,
        alpha_init=1.5,
        learnable_alpha=True
    )

    # 模拟模型输出（2D 特征）
    # 这是 soil_net.py 中 SoilNetLSTM.forward() 的实际输入
    flat_raster = torch.randn(batch_size, cnn_output_dim)  # CNN 输出
    lstm_output = torch.randn(batch_size, lstm_out)  # LSTM 输出

    print(f"\n  模型实际输出维度:")
    print(f"    CNN output (visual): {flat_raster.shape}")
    print(f"    LSTM output (climate): {lstm_output.shape}")
    print(f"    这些是 2D 张量，会被 unsqueeze 到 [B, 1, D]")

    # 前向传播
    fusion.eval()
    with torch.no_grad():
        # 捕获中间结果
        climate_unsq = lstm_output.unsqueeze(1)
        visual_unsq = flat_raster.unsqueeze(1)

        visual_proj = fusion.climate_visual_fusion.weak_proj(visual_unsq)
        attn_out, attn_weights = fusion.climate_visual_fusion.cross_attention(
            query=climate_unsq,
            key=visual_proj,
            value=visual_proj
        )

        fused = fusion(lstm_output, flat_raster)

    print(f"\n  Cross-Attention 中间结果:")
    print(f"    Query shape: {climate_unsq.shape}")
    print(f"    Key/Value shape: {visual_proj.shape}")
    print(f"    Attention weights shape: {attn_weights.shape}")
    if attn_weights.dim() == 3:
        print(f"    Attention weights (全部): {attn_weights[0, 0, 0].item():.4f} (应为1.0)")
    elif attn_weights.dim() == 4:
        print(f"    Attention weights (全部): {attn_weights[0, 0, 0, 0].item():.4f} (应为1.0)")

    print(f"\n  ⚠️  核心问题:")
    print(f"    当 seq_len=1 时，cross-attention 退化为:")
    print(f"    attn_out = softmax(Q·K^T/√d) · V = 1.0 · V = V")
    print(f"    即 attn_out ≈ visual_proj (线性投影)")
    print(f"    最终: fused = climate + alpha * linear(visual)")
    print(f"    这不是真正的注意力机制，只是一个带残差的线性层！")


def provide_solutions():
    """提供解决方案"""
    print("\n" + "=" * 70)
    print("解决方案")
    print("=" * 70)

    print("""
问题根源:
  SoilNetLSTM.forward() 输出的 flat_raster 和 lstm_output 都是 2D [B, D]，
  传入 S-CMRL 后被 unsqueeze 到 [B, 1, D]，导致 seq_len=1，
  cross-attention 退化为线性变换。

解决方案:

方案1: 在 attention 前使用多头线性投影增加序列长度 (推荐)
  在 CrossModalResidualBlock.forward() 中:

  if climate_feat.dim() == 2:
      # 不直接 unsqueeze，而是通过线性投影创建多头表示
      # 例如: 使用 n_heads 个不同的投影创建 seq_len = n_heads
      climate_feat = self.climate_to_seq(climate_feat)  # [B, n_heads, D]
      squeeze_output = True

方案2: 使用 FiLM (Feature-wise Linear Modulation) 替代 attention
  当 seq_len=1 时，attention 无意义。FiLM 更适合:

  # 用 visual 生成 scale 和 shift 来调制 climate
  scale = self.scale_proj(visual_feat)  # [B, climate_dim]
  shift = self.shift_proj(visual_feat)  # [B, climate_dim]
  fused = climate_feat * (1 + scale) + shift

方案3: 先对特征做 reshape 增加序列长度
  将 flat 特征 reshape 为多段序列:

  # [B, 768] -> [B, 48, 16] (48个token，每个16维)
  n_segments = 16
  seg_dim = visual_dim // n_segments
  visual_seq = visual_feat.view(B, n_segments, seg_dim)
  # 然后投影到 climate_dim

方案4: 使用 self-attention 先增强特征，再做 cross-attention
  先让特征有内部交互:

  # 对 climate 和 visual 分别做 self-attention
  climate_sa = self.climate_self_attn(climate_feat)  # [B, seq, D]
  visual_sa = self.visual_self_attn(visual_feat)     # [B, seq, D]
  # 然后做 cross-attention

方案5: 检查实际训练中 alpha 的值
  如果 alpha 趋近于 0，说明模型学会了忽略弱模态
  这可能是因为弱模态确实是噪声，也可能是训练问题

  在训练循环中添加 alpha 监控:
  if hasattr(model, 'fusion') and hasattr(model.fusion, 'get_alpha_values'):
      alphas = model.fusion.get_alpha_values()
      wandb.log(alphas)  # 或 tensorboard
""")


if __name__ == "__main__":
    print("Cross-Attention 退化诊断工具")
    print("=" * 70)

    diagnose_seq_len_issue()
    diagnose_alpha_collapse()
    diagnose_attention_weights_from_model()
    diagnose_gradient_flow()
    diagnose_semantic_alignment_loss()
    diagnose_real_model_forward()
    provide_solutions()
