import torch
import torch.nn as nn


class StaticBranch(nn.Module):
    """
    独立的静态特征分支：
      - 数值特征直接线性编码
      - 可选 LULC/CLCD 类别做 embedding 后与数值特征拼接
      - 输出统一的 hidden 维度，供主回归头融合
    """
    def __init__(
        self,
        numeric_dim: int,
        lulc_classes: int = 0,
        lulc_embed_dim: int = 16,
        hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.numeric_dim = numeric_dim
        self.lulc_classes = int(lulc_classes) if lulc_classes is not None else 0
        self.lulc_embed_dim = int(lulc_embed_dim)

        # LULC embedding（可选）
        if self.lulc_classes > 0:
            self.lulc_embedding = nn.Embedding(self.lulc_classes, self.lulc_embed_dim)
        else:
            self.lulc_embedding = None

        in_dim = numeric_dim + (self.lulc_embed_dim if self.lulc_embedding is not None else 0)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden),
        )

    def forward(self, numeric_feats: torch.Tensor, lulc_idx: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            numeric_feats: [B, numeric_dim]
            lulc_idx:     [B]，若无则传 None
        Returns:
            encoded: [B, hidden]
        """
        feats = numeric_feats
        if self.lulc_embedding is not None and lulc_idx is not None:
            # 安全处理无效索引（<0）
            safe_idx = torch.clamp(lulc_idx, min=0)
            emb = self.lulc_embedding(safe_idx)
            invalid_mask = (lulc_idx < 0).unsqueeze(-1).to(emb.dtype)
            emb = emb * (1.0 - invalid_mask)
            feats = torch.cat([feats, emb], dim=-1)
        encoded = self.encoder(feats)
        return encoded


