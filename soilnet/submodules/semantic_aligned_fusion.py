"""
Semantic-Aligned Cross-Modal Residual Learning (S-CMRL) Fusion Module

基于 S-CMRL (Semantic-Alignment Cross-Modal Residual Learning) 的多模态融合模块
参考: Brain-Cog-Lab/S-CMRL 仓库和论文 "Enhancing Audio-Visual Spiking Neural Networks 
through Semantic-Alignment and Cross-Modal Residual Learning"

核心思想：
1. 跨模态残差学习：强模态（Climate）作为 Query，主动检索弱模态（Visual/Static）中的有用信息
2. 语义对齐损失：通过对比学习拉近同一样本的不同模态特征，推远不同样本的特征
3. 可学习的 alpha 参数：控制弱模态的贡献，如果弱模态是噪声，alpha 会自动变小

为什么能防止弱模态变成噪声：
- 残差连接保证强模态的主导地位：F_final = F_climate + alpha * Attention(...)
- 如果弱模态是噪声，Attention 机制会让模型学习忽略它（alpha -> 0）
- 语义对齐损失迫使弱模态学习与强模态相关的语义，而不是随机特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossModalResidualBlock(nn.Module):
    """
    跨模态残差块
    
    实现公式：F_final = F_climate + alpha * Attention(Q_cli, K_weak, V_weak)
    
    其中：
    - Q (Query): 来自 Climate 分支（强模态）
    - K, V (Key, Value): 来自 Visual/Static 分支（弱模态）
    - alpha: learnable residual scale controlling the auxiliary contribution
    
    这样设计的好处：
    1. Climate 作为 Query，主动"检索"弱模态中有用的信息
    2. 残差连接保证强模态的主导地位
    3. 如果弱模态是噪声，alpha 会学习变小，自动屏蔽噪声
    """
    
    def __init__(
        self,
        climate_dim: int,
        weak_dim: int,
        num_heads: int = 8,
        alpha_init: float = 1.5,
        learnable_alpha: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.climate_dim = climate_dim
        self.weak_dim = weak_dim
        self.num_heads = num_heads
        
        # 投影层：将弱模态特征投影到与 Climate 相同的维度
        self.weak_proj = nn.Linear(weak_dim, climate_dim)
        
        # 跨模态注意力：Climate 作为 Query，弱模态作为 Key/Value
        # 注意：这里使用标准的 MultiheadAttention，而不是原仓库的 Spiking Attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=climate_dim,
            num_heads=num_heads,
            kdim=climate_dim,  # Key 来自投影后的弱模态
            vdim=climate_dim,  # Value 来自投影后的弱模态
            dropout=dropout,
            batch_first=True
        )
        
        # LayerNorm 和 Dropout
        self.norm = nn.LayerNorm(climate_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Alpha 参数：控制弱模态的贡献
        # 如果 learnable_alpha=True，alpha 是可学习的；否则是固定值
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(alpha_init, dtype=torch.float32))

    def gate_value(self) -> torch.Tensor:
        return self.alpha

    def attention_residual(
        self,
        climate_feat: torch.Tensor,
        weak_feat: torch.Tensor
    ) -> torch.Tensor:
        if weak_feat.dim() == 2:
            weak_feat = weak_feat.unsqueeze(1)
        key_value = self.weak_proj(weak_feat)
        attn_out, _ = self.cross_attention(
            query=climate_feat,
            key=key_value,
            value=key_value
        )
        return self.alpha * self.dropout(attn_out)
    
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
        # 处理维度：如果输入是 2D，添加序列维度
        if climate_feat.dim() == 2:
            climate_feat = climate_feat.unsqueeze(1)  # [B, 1, climate_dim]
            squeeze_output = True
        else:
            squeeze_output = False
        
        if weak_feat.dim() == 2:
            weak_feat = weak_feat.unsqueeze(1)  # [B, 1, weak_dim]
        # 投影弱模态特征到 Climate 维度
        
        # 跨模态注意力：Climate 作为 Query，弱模态作为 Key/Value
        
        # 残差连接：F_final = F_climate + alpha * Attention(...)
        # 这样设计保证强模态的主导地位
        fused_feat = climate_feat + self.attention_residual(climate_feat, weak_feat)
        fused_feat = self.norm(fused_feat)
        
        # 如果输入是 2D，移除序列维度
        if squeeze_output:
            fused_feat = fused_feat.squeeze(1)  # [B, climate_dim]
        
        return fused_feat


class ResidualProjectionBlock(nn.Module):
    """Inject a vector-valued auxiliary modality through a scaled residual."""

    def __init__(
        self,
        base_dim: int,
        auxiliary_dim: int,
        alpha_init: float = 1.5,
        learnable_alpha: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.weak_proj = nn.Linear(auxiliary_dim, base_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(base_dim)
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(alpha_init, dtype=torch.float32))

    def gate_value(self) -> torch.Tensor:
        return self.alpha

    def residual(self, auxiliary_feat: torch.Tensor) -> torch.Tensor:
        if auxiliary_feat.dim() == 3:
            auxiliary_feat = auxiliary_feat.mean(dim=1)
        return self.alpha * self.dropout(self.weak_proj(auxiliary_feat))

    def forward(
        self,
        base_feat: torch.Tensor,
        auxiliary_feat: torch.Tensor
    ) -> torch.Tensor:
        residual = self.residual(auxiliary_feat)
        if base_feat.dim() == 3:
            residual = residual.unsqueeze(1)
        return self.norm(base_feat + residual)


class LegacyCrossModalResidualBlock(nn.Module):
    """Original vector-input residual attention used by the 0.5404 run."""

    def __init__(self, climate_dim, weak_dim, num_heads=8, alpha_init=1.5,
                 learnable_alpha=True, dropout=0.1):
        super().__init__()
        self.weak_proj = nn.Linear(weak_dim, climate_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=climate_dim, num_heads=num_heads, kdim=climate_dim,
            vdim=climate_dim, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(climate_dim)
        self.dropout = nn.Dropout(dropout)
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(alpha_init, dtype=torch.float32))

    def residual(self, climate_feat, weak_feat):
        if weak_feat.dim() == 2:
            weak_feat = weak_feat.unsqueeze(1)
        weak_proj = self.weak_proj(weak_feat)
        attn_out, _ = self.cross_attention(
            query=climate_feat, key=weak_proj, value=weak_proj
        )
        return self.alpha * self.dropout(attn_out)

    def forward(self, climate_feat, weak_feat):
        squeeze_output = climate_feat.dim() == 2
        if squeeze_output:
            climate_feat = climate_feat.unsqueeze(1)
        fused = self.norm(climate_feat + self.residual(climate_feat, weak_feat))
        return fused.squeeze(1) if squeeze_output else fused


class LegacySemanticAlignedFusion(nn.Module):
    """Sequential vector-input residual fusion retained for comparison."""

    def __init__(self, climate_dim, visual_dim, static_dim=None, num_heads=8,
                 alpha_init=1.5, learnable_alpha=True, dropout=0.1):
        super().__init__()
        self.climate_visual_fusion = LegacyCrossModalResidualBlock(
            climate_dim, visual_dim, num_heads, alpha_init, learnable_alpha, dropout
        )
        self.climate_static_fusion = (
            LegacyCrossModalResidualBlock(
                climate_dim, static_dim, num_heads, alpha_init, learnable_alpha, dropout
            ) if static_dim is not None else None
        )

    def forward(self, climate_feat, visual_feat, static_feat=None):
        fused = self.climate_visual_fusion(climate_feat, visual_feat)
        if static_feat is not None and self.climate_static_fusion is not None:
            fused = self.climate_static_fusion(fused, static_feat)
        return fused

    def get_alpha_values(self):
        alphas = {"climate_visual": self.climate_visual_fusion.alpha.item()}
        if self.climate_static_fusion is not None:
            alphas["climate_static"] = self.climate_static_fusion.alpha.item()
        return alphas


class LegacySemanticAlignedFusionParallel(nn.Module):
    """Original parallel vector-input residual fusion used by the 0.5404 run."""

    def __init__(self, climate_dim, visual_dim, static_dim=None, num_heads=8,
                 alpha_init=1.5, learnable_alpha=True, dropout=0.1):
        super().__init__()
        self.climate_visual_fusion = LegacyCrossModalResidualBlock(
            climate_dim, visual_dim, num_heads, alpha_init, learnable_alpha, dropout
        )
        self.climate_static_fusion = (
            LegacyCrossModalResidualBlock(
                climate_dim, static_dim, num_heads, alpha_init, learnable_alpha, dropout
            ) if static_dim is not None else None
        )
        self.final_norm = nn.LayerNorm(climate_dim)

    def forward(self, climate_feat, visual_feat, static_feat=None):
        squeeze_output = climate_feat.dim() == 2
        if squeeze_output:
            climate_feat = climate_feat.unsqueeze(1)
        fused = climate_feat + self.climate_visual_fusion.residual(
            climate_feat, visual_feat
        )
        if static_feat is not None and self.climate_static_fusion is not None:
            fused = fused + self.climate_static_fusion.residual(
                climate_feat, static_feat
            )
        fused = self.final_norm(fused)
        return fused.squeeze(1) if squeeze_output else fused

    def get_alpha_values(self):
        alphas = {"climate_visual": self.climate_visual_fusion.alpha.item()}
        if self.climate_static_fusion is not None:
            alphas["climate_static"] = self.climate_static_fusion.alpha.item()
        return alphas


class CheckpointGatedCrossAttention(nn.Module):
    """Gated cross-attention block used by the released SoilNet-Fusion checkpoint."""

    def __init__(self, climate_dim, weak_dim, num_heads=8, alpha_init=1.5, dropout=0.1):
        super().__init__()
        if climate_dim % num_heads != 0:
            raise ValueError("climate_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = climate_dim // num_heads
        self.weak_proj = nn.Linear(weak_dim, climate_dim)
        self.cross_attention = nn.Module()
        self.cross_attention.q_proj = nn.Linear(climate_dim, climate_dim)
        self.cross_attention.k_proj = nn.Linear(climate_dim, climate_dim)
        self.cross_attention.v_proj = nn.Linear(climate_dim, climate_dim)
        self.cross_attention.o_proj = nn.Linear(climate_dim, climate_dim)
        self.cross_attention.gate_net = nn.Module()
        self.cross_attention.gate_net.gate_net = nn.Sequential(
            nn.Linear(climate_dim * 2, climate_dim // 4, bias=False),
            nn.ReLU(),
            nn.Linear(climate_dim // 4, climate_dim, bias=False),
        )
        self.norm = nn.LayerNorm(climate_dim)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, climate_feat, weak_feat):
        squeeze_output = climate_feat.dim() == 2
        if squeeze_output:
            climate_feat = climate_feat.unsqueeze(1)
        if weak_feat.dim() == 2:
            weak_feat = weak_feat.unsqueeze(1)

        weak_feat = self.weak_proj(weak_feat)
        batch_size, query_len, dim = climate_feat.shape
        key_len = weak_feat.size(1)
        query = self.cross_attention.q_proj(climate_feat).view(
            batch_size, query_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.cross_attention.k_proj(weak_feat).view(
            batch_size, key_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = self.cross_attention.v_proj(weak_feat).view(
            batch_size, key_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        attention = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        context = torch.matmul(attention, value).transpose(1, 2).contiguous().view(batch_size, query_len, dim)
        context = self.cross_attention.o_proj(context)
        gate = torch.sigmoid(self.cross_attention.gate_net.gate_net(torch.cat([climate_feat, context], dim=-1)))
        fused = self.norm(climate_feat + self.alpha * self.dropout(gate * context))
        return fused.squeeze(1) if squeeze_output else fused


class CheckpointCompatibleFusion(nn.Module):
    """Parallel climate-dominant fusion with the parameter layout of the saved model."""

    def __init__(self, climate_dim, visual_dim, static_dim=None, num_heads=8, alpha_init=1.5, dropout=0.1):
        super().__init__()
        self.fusion_weight_visual = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.fusion_weight_static = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.climate_visual_fusion = CheckpointGatedCrossAttention(
            climate_dim, visual_dim, num_heads, alpha_init, dropout
        )
        self.climate_static_fusion = (
            CheckpointGatedCrossAttention(climate_dim, static_dim, num_heads, alpha_init, dropout)
            if static_dim is not None else None
        )

    def forward(self, climate_feat, visual_feat, static_feat=None):
        visual_fused = self.climate_visual_fusion(climate_feat, visual_feat)
        fused = climate_feat + self.fusion_weight_visual * (visual_fused - climate_feat)
        if static_feat is not None and self.climate_static_fusion is not None:
            static_fused = self.climate_static_fusion(climate_feat, static_feat)
            fused = fused + self.fusion_weight_static * (static_fused - climate_feat)
        return fused


class SemanticAlignmentLoss(nn.Module):
    """
    语义对齐损失（Semantic Alignment Loss）
    
    基于对比学习（InfoNCE），拉近同一样本的 Climate-Visual 特征距离，
    推远不同样本间的距离。
    
    参考 S-CMRL 仓库参数：--temperature 0.07
    
    注意：如果两个模态的维度不同，会自动投影到较小的维度（使用平均池化）
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        climate_feat: torch.Tensor,
        visual_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        计算语义对齐损失
        
        Args:
            climate_feat: Climate 特征 [B, D_cli] 或 [B, seq_len, D_cli]
            visual_feat: Visual 特征 [B, D_vis] 或 [B, seq_len, D_vis]
        
        Returns:
            loss: 标量损失值
        """
        # 如果输入是序列，使用全局平均池化
        if climate_feat.dim() == 3:
            climate_feat = climate_feat.mean(dim=1)  # [B, D_cli]
        if visual_feat.dim() == 3:
            visual_feat = visual_feat.mean(dim=1)  # [B, D_vis]
        
        # 如果维度不同，投影到较小的维度（使用简单的平均池化）
        # 将特征 reshape 为 [B, 1, D]，然后使用 adaptive_avg_pool1d
        if climate_feat.size(1) != visual_feat.size(1):
            min_dim = min(climate_feat.size(1), visual_feat.size(1))
            if climate_feat.size(1) > min_dim:
                # 使用平均池化降维：将 D_cli 维度的特征平均分成 min_dim 组
                group_size = climate_feat.size(1) // min_dim
                climate_feat = climate_feat.view(
                    climate_feat.size(0), min_dim, group_size
                ).mean(dim=2)  # [B, min_dim]
            elif visual_feat.size(1) > min_dim:
                group_size = visual_feat.size(1) // min_dim
                visual_feat = visual_feat.view(
                    visual_feat.size(0), min_dim, group_size
                ).mean(dim=2)  # [B, min_dim]
        
        # L2 归一化
        climate_feat = F.normalize(climate_feat, p=2, dim=1)  # [B, D]
        visual_feat = F.normalize(visual_feat, p=2, dim=1)  # [B, D]
        
        batch_size = climate_feat.size(0)
        
        # 计算相似度矩阵
        # similarity[i, j] = cosine_similarity(climate_feat[i], visual_feat[j])
        similarity = torch.matmul(climate_feat, visual_feat.t()) / self.temperature  # [B, B]
        
        # 正样本：对角线元素（同一样本的 Climate-Visual 对）
        # 负样本：非对角线元素（不同样本的对）
        labels = torch.arange(batch_size, device=climate_feat.device)
        
        # InfoNCE Loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # 等价于交叉熵损失
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class SemanticAlignedFusion(nn.Module):
    """
    语义对齐跨模态残差融合模块
    
    输入三个模态的特征：
    - Climate（强模态）：性能最好
    - Visual（弱模态）：可能包含噪声
    - Static（弱模态）：可能包含噪声
    
    输出：融合后的特征，形状与 Climate 特征保持一致
    
    融合策略：
    1. 先融合 Climate + Visual（通过 CrossModalResidualBlock）
    2. 再通过受控残差投影注入 Static 向量
    3. 最终输出与 Climate 维度相同
    """
    
    def __init__(
        self,
        climate_dim: int,
        visual_dim: int,
        static_dim: Optional[int] = None,
        num_heads: int = 8,
        alpha_init: float = 1.5,
        learnable_alpha: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.climate_dim = climate_dim
        self.visual_dim = visual_dim
        self.static_dim = static_dim
        
        # Climate + Visual 融合
        self.climate_visual_fusion = CrossModalResidualBlock(
            climate_dim=climate_dim,
            weak_dim=visual_dim,
            num_heads=num_heads,
            alpha_init=alpha_init,
            learnable_alpha=learnable_alpha,
            dropout=dropout
        )
        
        # Static 是单向量，因此使用残差投影而不是退化的 cross-attention
        if static_dim is not None:
            self.climate_static_fusion = ResidualProjectionBlock(
                base_dim=climate_dim,
                auxiliary_dim=static_dim,
                alpha_init=alpha_init,
                learnable_alpha=learnable_alpha,
                dropout=dropout
            )
        else:
            self.climate_static_fusion = None
    
    def forward(
        self,
        climate_feat: torch.Tensor,
        visual_feat: torch.Tensor,
        static_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            climate_feat: Climate 特征 [B, climate_dim] 或 [B, seq_len, climate_dim]
            visual_feat: Visual 特征 [B, visual_dim] 或 [B, seq_len, visual_dim]
            static_feat: Static 特征 [B, static_dim]（可选）
        
        Returns:
            fused_feat: 融合后的特征，形状与 climate_feat 相同
        """
        # Step 1: Climate + Visual 融合
        fused = self.climate_visual_fusion(climate_feat, visual_feat)
        
        # Step 2: 如果提供了 Static 特征，继续融合
        if static_feat is not None and self.climate_static_fusion is not None:
            fused = self.climate_static_fusion(fused, static_feat)
        
        return fused
    
    def get_alpha_values(self) -> dict:
        """
        获取 alpha 参数值（用于监控弱模态的贡献）
        
        Returns:
            dict: 包含各个融合块的 alpha 值
        """
        alphas = {
            'climate_visual': self.climate_visual_fusion.gate_value().item()
        }
        
        if self.climate_static_fusion is not None:
            alphas['climate_static'] = self.climate_static_fusion.gate_value().item()
        
        return alphas


class SemanticAlignedFusionParallel(nn.Module):
    """
    并行版本的语义对齐跨模态残差融合模块
    
    改进点：从串行融合改为并行融合
    - 串行：Climate + Visual → Result1, Result1 + Static → Final
    - 并行：Climate 查询 Visual tokens，并与 Static 残差投影相加
    
    优点：
    1. Visual 和 Static 互不干扰，梯度传播更直接
    2. 物理意义更清晰：Climate 同时受到 Visual（当前地表）和 Static（固有环境）的修正
    3. 两个弱模态的贡献独立学习，可能更灵活
    
    公式：
    fused_feat = climate_feat + alpha_v * Attention(Q=climate, K=visual, V=visual)
                              + alpha_s * Projection(static)
    """
    
    def __init__(
        self,
        climate_dim: int,
        visual_dim: int,
        static_dim: Optional[int] = None,
        num_heads: int = 8,
        alpha_init: float = 1.5,
        learnable_alpha: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.climate_dim = climate_dim
        self.visual_dim = visual_dim
        self.static_dim = static_dim
        
        # Climate + Visual 融合（独立）
        self.climate_visual_fusion = CrossModalResidualBlock(
            climate_dim=climate_dim,
            weak_dim=visual_dim,
            num_heads=num_heads,
            alpha_init=alpha_init,
            learnable_alpha=learnable_alpha,
            dropout=dropout
        )
        
        # Static 是单向量，因此使用独立残差投影
        if static_dim is not None:
            self.climate_static_fusion = ResidualProjectionBlock(
                base_dim=climate_dim,
                auxiliary_dim=static_dim,
                alpha_init=alpha_init,
                learnable_alpha=learnable_alpha,
                dropout=dropout
            )
        else:
            self.climate_static_fusion = None
        
        # 最终的 LayerNorm（可选，用于稳定训练）
        self.final_norm = nn.LayerNorm(climate_dim)
    
    def forward(
        self,
        climate_feat: torch.Tensor,
        visual_feat: torch.Tensor,
        static_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        并行融合前向传播
        
        Args:
            climate_feat: Climate 特征 [B, climate_dim] 或 [B, seq_len, climate_dim]
            visual_feat: Visual 空间 tokens [B, num_tokens, visual_dim]
            static_feat: Static 特征 [B, static_dim]（可选）
        
        Returns:
            fused_feat: 融合后的特征，形状与 climate_feat 相同
        """
        # 处理维度：如果输入是 2D，添加序列维度
        if climate_feat.dim() == 2:
            climate_feat = climate_feat.unsqueeze(1)  # [B, 1, climate_dim]
            squeeze_output = True
        else:
            squeeze_output = False
        
        if visual_feat.dim() == 2:
            visual_feat = visual_feat.unsqueeze(1)
        if static_feat is not None and static_feat.dim() == 2:
            static_feat = static_feat.unsqueeze(1)
        
        # Step 1: 分别计算残差（并行）
        # Climate 查询 Visual - 只计算残差部分（不包含原始climate_feat）
        resid_visual = self.climate_visual_fusion.attention_residual(
            climate_feat,
            visual_feat
        )
        
        # Static 是单向量，通过投影计算残差
        resid_static = None
        if static_feat is not None and self.climate_static_fusion is not None:
            resid_static = self.climate_static_fusion.residual(static_feat).unsqueeze(1)
        
        # Step 2: 统一相加（并行融合）
        # 公式：fused_feat = climate_feat + resid_visual + resid_static
        fused_feat = climate_feat + resid_visual
        if resid_static is not None:
            fused_feat = fused_feat + resid_static
        
        # 最终归一化
        fused_feat = self.final_norm(fused_feat)
        
        # 如果输入是 2D，移除序列维度
        if squeeze_output:
            fused_feat = fused_feat.squeeze(1)  # [B, climate_dim]
        
        return fused_feat
    
    def get_alpha_values(self) -> dict:
        """
        获取 alpha 参数值（用于监控弱模态的贡献）
        
        Returns:
            dict: 包含各个融合块的 alpha 值
        """
        alphas = {
            'climate_visual': self.climate_visual_fusion.gate_value().item()
        }
        
        if self.climate_static_fusion is not None:
            alphas['climate_static'] = self.climate_static_fusion.gate_value().item()
        
        return alphas


class GatedFeatureFusion(nn.Module):
    """Competitive pooled-feature gated fusion baseline."""

    def __init__(self, climate_dim, visual_dim, static_dim=None, dropout=0.1):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, climate_dim)
        self.visual_gate = nn.Sequential(
            nn.Linear(climate_dim * 2, climate_dim), nn.Sigmoid()
        )
        self.static_proj = nn.Linear(static_dim, climate_dim) if static_dim is not None else None
        self.static_gate = (
            nn.Sequential(nn.Linear(climate_dim * 2, climate_dim), nn.Sigmoid())
            if static_dim is not None else None
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(climate_dim)

    def forward(self, climate_feat, visual_feat, static_feat=None):
        if visual_feat.dim() == 3:
            visual_feat = visual_feat.mean(dim=1)
        visual = self.visual_proj(visual_feat)
        gate = self.visual_gate(torch.cat([climate_feat, visual], dim=-1))
        fused = gate * climate_feat + (1.0 - gate) * self.dropout(visual)
        if static_feat is not None and self.static_proj is not None:
            if static_feat.dim() == 3:
                static_feat = static_feat.mean(dim=1)
            static = self.static_proj(static_feat)
            static_gate = self.static_gate(torch.cat([fused, static], dim=-1))
            fused = static_gate * fused + (1.0 - static_gate) * self.dropout(static)
        return self.norm(fused)


class TokenCrossAttentionFusion(nn.Module):
    """Token-level cross-attention baseline without a learnable residual scale."""

    def __init__(self, climate_dim, visual_dim, static_dim=None, num_heads=8, dropout=0.1):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, climate_dim)
        self.cross_attention = nn.MultiheadAttention(
            climate_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.visual_fusion = nn.Sequential(
            nn.Linear(climate_dim * 2, climate_dim),
            nn.LayerNorm(climate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.static_proj = nn.Linear(static_dim, climate_dim) if static_dim is not None else None
        self.static_fusion = (
            nn.Sequential(
                nn.Linear(climate_dim * 2, climate_dim),
                nn.LayerNorm(climate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ) if static_dim is not None else None
        )

    def forward(self, climate_feat, visual_feat, static_feat=None):
        query = climate_feat.unsqueeze(1) if climate_feat.dim() == 2 else climate_feat
        tokens = visual_feat.unsqueeze(1) if visual_feat.dim() == 2 else visual_feat
        tokens = self.visual_proj(tokens)
        attended, _ = self.cross_attention(query, tokens, tokens, need_weights=False)
        climate_vector = query.mean(dim=1)
        attended_vector = attended.mean(dim=1)
        fused = self.visual_fusion(torch.cat([climate_vector, attended_vector], dim=-1))
        if static_feat is not None and self.static_proj is not None:
            if static_feat.dim() == 3:
                static_feat = static_feat.mean(dim=1)
            static = self.static_proj(static_feat)
            fused = self.static_fusion(torch.cat([fused, static], dim=-1))
        return fused


class FiLMFusion(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) 融合模块

    用弱模态生成 scale 和 shift 来调制强模态（Climate）：
        fused = climate * (1 + alpha * scale) + alpha * shift

    优点：
    - 计算高效，不依赖序列长度
    - 梯度流好，alpha 可以有效学习
    - 适合 2D 特征输入（来自 CNN/RNN 的 flat 输出）

    参考: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018
    """

    def __init__(
        self,
        climate_dim: int,
        visual_dim: int,
        static_dim: Optional[int] = None,
        alpha_init: float = 1.5,
        learnable_alpha: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.climate_dim = climate_dim
        self.visual_dim = visual_dim
        self.static_dim = static_dim

        # Visual -> scale 和 shift
        self.visual_scale = nn.Sequential(
            nn.Linear(visual_dim, climate_dim),
            nn.Tanh()  # 限制 scale 范围在 [-1, 1]
        )
        self.visual_shift = nn.Linear(visual_dim, climate_dim)

        # Static -> scale 和 shift (如果提供)
        if static_dim is not None:
            self.static_scale = nn.Sequential(
                nn.Linear(static_dim, climate_dim),
                nn.Tanh()
            )
            self.static_shift = nn.Linear(static_dim, climate_dim)
        else:
            self.static_scale = None
            self.static_shift = None

        # LayerNorm
        self.norm = nn.LayerNorm(climate_dim)
        self.dropout = nn.Dropout(dropout)

        # Alpha 参数：控制弱模态的贡献
        if learnable_alpha:
            self.alpha_visual = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
            if static_dim is not None:
                self.alpha_static = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        else:
            self.register_buffer('alpha_visual', torch.tensor(alpha_init, dtype=torch.float32))
            if static_dim is not None:
                self.register_buffer('alpha_static', torch.tensor(alpha_init, dtype=torch.float32))

    def forward(
        self,
        climate_feat: torch.Tensor,
        visual_feat: torch.Tensor,
        static_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            climate_feat: Climate 特征 [B, climate_dim]
            visual_feat: Visual 特征 [B, visual_dim]
            static_feat: Static 特征 [B, static_dim] (可选)

        Returns:
            fused_feat: 融合后的特征 [B, climate_dim]
        """
        # Visual FiLM 调制
        v_scale = self.visual_scale(visual_feat)    # [B, climate_dim]
        v_shift = self.visual_shift(visual_feat)    # [B, climate_dim]

        # FiLM: gamma * x + beta
        fused = climate_feat * (1 + self.alpha_visual * v_scale) + self.alpha_visual * v_shift

        # Static FiLM 调制 (如果提供)
        if static_feat is not None and self.static_scale is not None:
            s_scale = self.static_scale(static_feat)
            s_shift = self.static_shift(static_feat)
            fused = fused * (1 + self.alpha_static * s_scale) + self.alpha_static * s_shift

        # 归一化
        fused = self.norm(fused)
        fused = self.dropout(fused)

        return fused

    def get_alpha_values(self) -> dict:
        """获取 alpha 参数值"""
        alphas = {'visual': self.alpha_visual.item()}
        if hasattr(self, 'alpha_static') and self.alpha_static is not None:
            alphas['static'] = self.alpha_static.item()
        return alphas


# ========== 测试代码 ==========
if __name__ == '__main__':
    print("=" * 70)
    print("测试 SemanticAlignedFusion 模块")
    print("=" * 70)
    
    batch_size = 4
    climate_dim = 128
    visual_dim = 384
    static_dim = 64
    
    # 创建融合模块
    fusion = SemanticAlignedFusion(
        climate_dim=climate_dim,
        visual_dim=visual_dim,
        static_dim=static_dim,
        num_heads=8,
        alpha_init=1.5,
        learnable_alpha=True
    )
    
    # 创建输入特征
    climate_feat = torch.randn(batch_size, climate_dim)
    visual_feat = torch.randn(batch_size, 16, visual_dim)
    static_feat = torch.randn(batch_size, static_dim)
    
    # 前向传播
    fused_feat = fusion(climate_feat, visual_feat, static_feat)
    
    print(f"\n输入特征形状:")
    print(f"  Climate: {climate_feat.shape}")
    print(f"  Visual:  {visual_feat.shape}")
    print(f"  Static:  {static_feat.shape}")
    print(f"\n融合后特征形状: {fused_feat.shape}")
    print(f"  预期形状: ({batch_size}, {climate_dim})")
    
    assert fused_feat.shape == (batch_size, climate_dim), \
        f"形状不匹配: 期望 ({batch_size}, {climate_dim}), 得到 {fused_feat.shape}"
    
    # 测试 alpha 值
    alphas = fusion.get_alpha_values()
    print(f"\nAlpha 参数值:")
    for key, value in alphas.items():
        print(f"  {key}: {value:.4f}")
    
    # 测试语义对齐损失
    print("\n" + "=" * 70)
    print("测试 SemanticAlignmentLoss")
    print("=" * 70)
    
    alignment_loss = SemanticAlignmentLoss(temperature=0.07)
    loss = alignment_loss(climate_feat, visual_feat)
    print(f"语义对齐损失: {loss.item():.4f}")
    
    # 测试序列输入
    print("\n" + "=" * 70)
    print("测试序列输入")
    print("=" * 70)
    
    seq_len = 10
    climate_seq = torch.randn(batch_size, seq_len, climate_dim)
    visual_seq = torch.randn(batch_size, seq_len, visual_dim)
    
    fused_seq = fusion(climate_seq, visual_seq, static_feat)
    print(f"序列输入融合后形状: {fused_seq.shape}")
    print(f"  预期形状: ({batch_size}, {seq_len}, {climate_dim})")
    
    assert fused_seq.shape == (batch_size, seq_len, climate_dim), \
        f"形状不匹配: 期望 ({batch_size}, {seq_len}, {climate_dim}), 得到 {fused_seq.shape}"
    
    print("\n[SUCCESS] 所有测试通过!")
