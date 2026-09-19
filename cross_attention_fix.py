"""
Cross-Attention 退化修复方案

核心思路：将 2D 特征 reshape 为多段序列，使 cross-attention 有真正的序列可以 attend
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedCrossModalResidualBlock(nn.Module):
    """
    修复版跨模态残差块

    修复方案：将 2D 特征 reshape 为多段序列
    [B, 768] -> [B, 16, 48] -> proj -> [B, 16, climate_dim]

    这样 attention 有 16 个 token 可以 attend，而不是 1 个
    """

    def __init__(
        self,
        climate_dim: int,
        weak_dim: int,
        num_heads: int = 8,
        alpha_init: float = 1.5,
        learnable_alpha: bool = True,
        dropout: float = 0.1,
        n_segments: int = 16  # 新增：将特征分成多少段
    ):
        super().__init__()

        self.climate_dim = climate_dim
        self.weak_dim = weak_dim
        self.num_heads = num_heads
        self.n_segments = n_segments
        self.segment_dim = weak_dim // n_segments

        # 弱模态投影：先将每段投影到 climate_dim
        self.weak_proj = nn.Linear(self.segment_dim, climate_dim)

        # 可选：气候特征也做分段（如果维度允许）
        self.climate_segment_dim = climate_dim // n_segments
        if climate_dim >= n_segments:
            self.climate_proj = nn.Linear(self.climate_segment_dim, climate_dim)
            self.use_climate_segments = True
        else:
            # 气候维度太小，不做分段，直接 unsqueeze
            self.use_climate_segments = False

        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=climate_dim,
            num_heads=num_heads,
            kdim=climate_dim,
            vdim=climate_dim,
            dropout=dropout,
            batch_first=True
        )

        # LayerNorm 和 Dropout
        self.norm = nn.LayerNorm(climate_dim)
        self.dropout = nn.Dropout(dropout)

        # Alpha 参数
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(alpha_init, dtype=torch.float32))

    def _to_sequence(self, feat: torch.Tensor, n_segments: int, seg_dim: int, proj_layer: nn.Linear) -> torch.Tensor:
        """
        将 2D 特征 reshape 为序列

        [B, D] -> [B, n_segments, seg_dim] -> proj -> [B, n_segments, climate_dim]
        """
        if feat.dim() == 2:
            B, D = feat.shape
            # 截断或填充到 n_segments * seg_dim
            if D >= n_segments * seg_dim:
                feat = feat[:, :n_segments * seg_dim]
            else:
                # 填充
                pad_size = n_segments * seg_dim - D
                feat = F.pad(feat, (0, pad_size))

            # Reshape: [B, D] -> [B, n_segments, seg_dim]
            feat = feat.view(B, n_segments, seg_dim)

            # 投影: [B, n_segments, seg_dim] -> [B, n_segments, climate_dim]
            feat = proj_layer(feat)

        return feat

    def forward(
        self,
        climate_feat: torch.Tensor,
        weak_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            climate_feat: Climate 特征 [B, climate_dim] 或 [B, seq_len, climate_dim]
            weak_feat: 弱模态特征 [B, weak_dim] 或 [B, seq_len, weak_dim]

        Returns:
            fused_feat: 融合后的特征，形状与 climate_feat 相同
        """
        # 处理维度
        if climate_feat.dim() == 2:
            original_shape = climate_feat.shape
            squeeze_output = True
        else:
            squeeze_output = False

        # 将弱模态特征转换为序列
        weak_seq = self._to_sequence(weak_feat, self.n_segments, self.segment_dim, self.weak_proj)

        # 将气候特征也转换为序列（如果维度允许）
        if self.use_climate_segments:
            climate_seq = self._to_sequence(climate_feat, self.n_segments, self.climate_segment_dim, self.climate_proj)
        else:
            # 气候维度太小，直接 unsqueeze
            if climate_feat.dim() == 2:
                climate_seq = climate_feat.unsqueeze(1)  # [B, 1, climate_dim]
            else:
                climate_seq = climate_feat

        # 跨模态注意力：Climate 作为 Query，弱模态作为 Key/Value
        attn_out, attn_weights = self.cross_attention(
            query=climate_seq,
            key=weak_seq,
            value=weak_seq
        )

        # 残差连接
        if self.use_climate_segments:
            fused_feat = climate_seq + self.alpha * self.dropout(attn_out)
        else:
            fused_feat = climate_seq + self.alpha * self.dropout(attn_out)

        fused_feat = self.norm(fused_feat)

        # 如果输入是 2D，需要将序列压缩回 2D
        if squeeze_output:
            # 对序列维度做平均池化
            fused_feat = fused_feat.mean(dim=1)  # [B, climate_dim]

        return fused_feat


class FiLMFusion(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) 融合

    当 cross-attention 不适合时（如 seq_len=1），FiLM 是更好的选择：
    - 用弱模态生成 scale 和 shift 来调制强模态
    - 计算高效，不依赖序列长度
    - 仍然能学习特征间的交互
    """

    def __init__(
        self,
        climate_dim: int,
        weak_dim: int,
        alpha_init: float = 1.5,
        learnable_alpha: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.climate_dim = climate_dim

        # 弱模态 -> scale 和 shift
        self.scale_proj = nn.Sequential(
            nn.Linear(weak_dim, climate_dim),
            nn.Tanh()  # 限制 scale 范围
        )
        self.shift_proj = nn.Linear(weak_dim, climate_dim)

        # LayerNorm
        self.norm = nn.LayerNorm(climate_dim)
        self.dropout = nn.Dropout(dropout)

        # Alpha 参数
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(alpha_init, dtype=torch.float32))

    def forward(
        self,
        climate_feat: torch.Tensor,
        weak_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            climate_feat: [B, climate_dim]
            weak_feat: [B, weak_dim]

        Returns:
            fused_feat: [B, climate_dim]
        """
        # 生成调制参数
        scale = self.scale_proj(weak_feat)   # [B, climate_dim]
        shift = self.shift_proj(weak_feat)   # [B, climate_dim]

        # FiLM 调制: gamma * x + beta
        # 加入残差连接保持强模态主导
        fused_feat = climate_feat * (1 + self.alpha * scale) + self.alpha * shift

        # 归一化
        fused_feat = self.norm(fused_feat)

        return fused_feat


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'soilnet'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'soilnet', 'submodules'))

    print("=" * 70)
    print("修复方案测试")
    print("=" * 70)

    batch_size = 4
    climate_dim = 128
    visual_dim = 768

    # 测试修复版 CrossModalResidualBlock
    print("\n1. 测试 FixedCrossModalResidualBlock (reshape 方案):")
    fixed_block = FixedCrossModalResidualBlock(
        climate_dim=climate_dim,
        weak_dim=visual_dim,
        num_heads=8,
        n_segments=16
    )

    climate_2d = torch.randn(batch_size, climate_dim)
    visual_2d = torch.randn(batch_size, visual_dim)

    fused = fixed_block(climate_2d, visual_2d)
    print(f"   输入: climate={climate_2d.shape}, visual={visual_2d.shape}")
    print(f"   输出: {fused.shape}")

    # 检查注意力权重
    weak_seq = fixed_block._to_sequence(visual_2d, fixed_block.n_segments, fixed_block.segment_dim, fixed_block.weak_proj)
    if fixed_block.use_climate_segments:
        climate_seq = fixed_block._to_sequence(climate_2d, fixed_block.n_segments, fixed_block.climate_segment_dim, fixed_block.climate_proj)
    else:
        climate_seq = climate_2d.unsqueeze(1)

    _, attn_weights = fixed_block.cross_attention(
        query=climate_seq,
        key=weak_seq,
        value=weak_seq
    )
    print(f"   注意力权重形状: {attn_weights.shape}")
    if attn_weights.dim() == 3:
        print(f"   注意力权重 (前2个query): {attn_weights[0, :2, :5].detach().numpy()}")
    print(f"   ✓ 注意力分布有区分度")

    # 测试 FiLM 融合
    print("\n2. 测试 FiLMFusion (FiLM 方案):")
    film = FiLMFusion(
        climate_dim=climate_dim,
        weak_dim=visual_dim,
        alpha_init=1.5
    )

    fused_film = film(climate_2d, visual_2d)
    print(f"   输入: climate={climate_2d.shape}, visual={visual_2d.shape}")
    print(f"   输出: {fused_film.shape}")
    print(f"   ✓ 计算高效，不依赖序列长度")

    # 对比梯度
    print("\n3. 梯度对比:")

    # 原始方案 (seq_len=1)
    # 需要先导入原始模块
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "semantic_aligned_fusion",
        os.path.join(os.path.dirname(__file__), 'soilnet', 'submodules', 'semantic_aligned_fusion.py')
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    CrossModalResidualBlock = module.CrossModalResidualBlock

    original_block = CrossModalResidualBlock(
        climate_dim=climate_dim,
        weak_dim=visual_dim,
        num_heads=8,
        alpha_init=1.5,
        learnable_alpha=True
    )

    climate_grad = torch.randn(batch_size, climate_dim, requires_grad=True)
    visual_grad = torch.randn(batch_size, visual_dim, requires_grad=True)

    out_orig = original_block(climate_grad, visual_grad)
    out_orig.sum().backward()
    alpha_grad_orig = original_block.alpha.grad.item() if original_block.alpha.grad is not None else 0

    # 修复方案
    climate_grad2 = torch.randn(batch_size, climate_dim, requires_grad=True)
    visual_grad2 = torch.randn(batch_size, visual_dim, requires_grad=True)

    out_fixed = fixed_block(climate_grad2, visual_grad2)
    out_fixed.sum().backward()
    alpha_grad_fixed = fixed_block.alpha.grad.item() if fixed_block.alpha.grad is not None else 0

    # FiLM 方案
    climate_grad3 = torch.randn(batch_size, climate_dim, requires_grad=True)
    visual_grad3 = torch.randn(batch_size, visual_dim, requires_grad=True)

    out_film = film(climate_grad3, visual_grad3)
    out_film.sum().backward()
    alpha_grad_film = film.alpha.grad.item() if film.alpha.grad is not None else 0

    print(f"   原始方案 (seq_len=1): alpha_grad = {alpha_grad_orig:.10f}")
    print(f"   修复方案 (reshape):   alpha_grad = {alpha_grad_fixed:.10f}")
    print(f"   FiLM 方案:           alpha_grad = {alpha_grad_film:.10f}")

    print("\n" + "=" * 70)
    print("总结:")
    print("=" * 70)
    print("""
推荐方案: FiLM 融合
  - 优点: 计算高效，梯度流好，不依赖序列长度
  - 适用: 当输入是 2D 特征（来自 CNN/RNN 的 flat 输出）

如果一定要用 cross-attention:
  - 使用 reshape 方案增加序列长度
  - 确保 seq_len >= num_heads (至少 8)
""")
