"""
多预处理光谱 CNN 模块 (Multi-preprocessed Spectral CNN)

基于 Tziolas et al. (Geoderma, 2024) 的多预处理光谱 CNN 思路实现。
该模块将遥感影像 patch 进行空间池化得到光谱向量，然后对光谱进行三种预处理
（原始反射率 / 伪吸光度 / SNV），最后通过 1D CNN 提取特征。

Reference:
    Tziolas, N., et al. (2024). Sentinel-2 soil spectral CNN for SOC prediction.
    Geoderma.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class MultiPreprocSpectralCNN(nn.Module):
    """
    多预处理光谱 CNN 编码器
    
    将输入的遥感影像 patch 转换为固定维度的 embedding：
    1. 空间池化：(B, C_img, H, W) -> (B, C_img)
    2. 三通道预处理：原始反射率 / 伪吸光度 / SNV
    3. 1D CNN 特征提取
    4. MLP 输出 embedding
    
    Args:
        n_bands: 输入光谱维度（波段数），例如 9 或 21
        emb_dim: 输出 embedding 维度，用于后续多模态融合，默认 32
        hidden_dims: MLP 隐藏层维度，默认 (32, 8)
        dropout: Dropout 概率，默认 0.2
        conv_channels: CNN 卷积层通道数，默认 (16, 32)
    
    Input:
        x: 支持两种形状
           - (B, C_img, H, W): 完整 patch，会先做空间池化
           - (B, C_img): 已池化的光谱向量
    
    Output:
        embedding: (B, emb_dim) 的特征向量
    """
    
    def __init__(
        self,
        n_bands: int,
        emb_dim: int = 32,
        hidden_dims: Tuple[int, ...] = (32, 8),
        dropout: float = 0.2,
        conv_channels: Tuple[int, int] = (16, 32)
    ):
        super().__init__()
        
        self.n_bands = n_bands
        self.emb_dim = emb_dim
        self.hidden_dims = hidden_dims
        self.dropout_prob = dropout
        
        # ========== 1D CNN 结构 ==========
        # 基于 Tziolas et al. (Geoderma, 2024) 的浅层 1D CNN 设计
        
        # Conv block 1: 3 -> 16 channels
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=conv_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels[0]),
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Conv block 2: 16 -> 32 channels
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv_channels[0], out_channels=conv_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels[1]),
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # 动态计算池化后的长度
        # 两次 MaxPool1d(kernel_size=2) 后，长度缩小约 4 倍
        self.pooled_len = self._compute_pooled_length(n_bands)
        flat_dim = conv_channels[1] * self.pooled_len
        
        # ========== MLP 结构 ==========
        # 构建 MLP 层：flat_dim -> hidden_dims[0] -> hidden_dims[1] -> ... -> emb_dim
        mlp_layers = []
        in_dim = flat_dim
        
        for h_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LeakyReLU(0.01),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        # 最后一层：输出 emb_dim
        mlp_layers.extend([
            nn.Linear(in_dim, emb_dim),
            nn.LeakyReLU(0.01)
        ])
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # 初始化权重
        self._init_weights()
    
    def _compute_pooled_length(self, n_bands: int) -> int:
        """
        计算经过两次 MaxPool1d(kernel_size=2) 后的序列长度
        
        注意：如果 n_bands 很小（如 < 4），可能会出现问题
        """
        # 模拟前向传播计算长度
        # Conv1d with padding=1 不改变长度
        # MaxPool1d(kernel_size=2) 将长度减半（向下取整）
        len_after_pool1 = n_bands // 2
        len_after_pool2 = len_after_pool1 // 2
        
        # 确保长度至少为 1
        return max(len_after_pool2, 1)
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _preprocess_spectrum(self, spec: torch.Tensor) -> torch.Tensor:
        """
        对光谱向量进行三通道预处理
        
        基于 Tziolas et al. (Geoderma, 2024) 的预处理方法：
        1. 通道 1：原始反射率 (refl)
        2. 通道 2：伪吸光度 (absorb) = log10(1 / refl)
        3. 通道 3：SNV 标准正态变量
        
        Args:
            spec: (B, n_bands) 光谱向量
        
        Returns:
            x_3ch: (B, 3, n_bands) 三通道预处理后的光谱
        """
        # 通道 1：原始反射率
        refl = spec  # (B, n_bands)
        
        # 通道 2：伪吸光度（注意数值稳定性）
        # 先 clamp 避免 log(0) 或 log(负数)
        refl_safe = torch.clamp(refl, min=1e-4)
        absorb = torch.log10(1.0 / refl_safe)  # (B, n_bands)
        
        # 通道 3：SNV 标准正态变量（沿 band 维度标准化）
        mean = refl.mean(dim=1, keepdim=True)  # (B, 1)
        std = refl.std(dim=1, keepdim=True) + 1e-6  # (B, 1)，添加小常数避免除零
        snv = (refl - mean) / std  # (B, n_bands)
        
        # 堆叠成 3 通道
        x_3ch = torch.stack([refl, absorb, snv], dim=1)  # (B, 3, n_bands)
        
        return x_3ch
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，支持两种形状：
               - (B, C_img, H, W): 完整 patch
               - (B, C_img): 已池化的光谱向量
        
        Returns:
            embedding: (B, emb_dim) 的特征向量
        """
        # Step 1: 空间池化（如果输入是 4D）
        if x.dim() == 4:
            # (B, C_img, H, W) -> (B, C_img)
            spec = x.mean(dim=(2, 3))
        elif x.dim() == 2:
            spec = x
        else:
            raise ValueError(f"Input must be 2D (B, C) or 4D (B, C, H, W), got {x.dim()}D")
        
        # Step 2: 检查波段数
        if spec.shape[1] != self.n_bands:
            raise ValueError(
                f"Input spectral dimension ({spec.shape[1]}) does not match "
                f"expected n_bands ({self.n_bands})"
            )
        
        # Step 3: 三通道预处理
        x_3ch = self._preprocess_spectrum(spec)  # (B, 3, n_bands)
        
        # Step 4: 1D CNN 特征提取
        x = self.conv1(x_3ch)  # (B, 16, n_bands//2)
        x = self.conv2(x)      # (B, 32, n_bands//4)
        
        # Step 5: Flatten
        x = x.view(x.size(0), -1)  # (B, 32 * pooled_len)
        
        # Step 6: MLP
        embedding = self.mlp(x)  # (B, emb_dim)
        
        return embedding


class SpectralCNNRegressor(nn.Module):
    """
    光谱 CNN 回归器（单模态版本）
    
    用于单独训练图像分支的对比实验，不影响多模态结构。
    包含 MultiPreprocSpectralCNN 作为 backbone，加上一个回归头。
    
    Args:
        n_bands: 输入光谱维度
        emb_dim: backbone 输出维度，默认 32
        hidden_dims: MLP 隐藏层维度，默认 (32, 8)
        dropout: Dropout 概率，默认 0.2
        use_tanh: 是否使用 tanh 激活输出，默认 False
    
    Input:
        x: (B, C_img, H, W) 或 (B, C_img)
    
    Output:
        soc: (B, 1) SOC 预测值
    """
    
    def __init__(
        self,
        n_bands: int,
        emb_dim: int = 32,
        hidden_dims: Tuple[int, ...] = (32, 8),
        dropout: float = 0.2,
        use_tanh: bool = False
    ):
        super().__init__()
        
        self.backbone = MultiPreprocSpectralCNN(
            n_bands=n_bands,
            emb_dim=emb_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        self.regressor = nn.Linear(emb_dim, 1)
        self.use_tanh = use_tanh
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
        
        Returns:
            soc: (B, 1) SOC 预测值
        """
        embedding = self.backbone(x)  # (B, emb_dim)
        soc = self.regressor(embedding)  # (B, 1)
        
        if self.use_tanh:
            soc = torch.tanh(soc)
        
        return soc


# ========== 测试代码 ==========
if __name__ == '__main__':
    print("=" * 60)
    print("Testing MultiPreprocSpectralCNN")
    print("=" * 60)
    
    # 测试不同波段数
    for n_bands in [9, 12, 21]:
        print(f"\n--- Testing with n_bands={n_bands} ---")
        
        model = MultiPreprocSpectralCNN(
            n_bands=n_bands,
            emb_dim=32,
            hidden_dims=(32, 8),
            dropout=0.2
        )
        
        # 测试 4D 输入 (patch)
        batch_size = 4
        x_4d = torch.randn(batch_size, n_bands, 64, 64)
        out_4d = model(x_4d)
        print(f"  4D input shape: {x_4d.shape}")
        print(f"  Output shape:   {out_4d.shape}")
        assert out_4d.shape == (batch_size, 32), f"Expected (4, 32), got {out_4d.shape}"
        
        # 测试 2D 输入 (已池化)
        x_2d = torch.randn(batch_size, n_bands)
        out_2d = model(x_2d)
        print(f"  2D input shape: {x_2d.shape}")
        print(f"  Output shape:   {out_2d.shape}")
        assert out_2d.shape == (batch_size, 32), f"Expected (4, 32), got {out_2d.shape}"
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total params:     {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
    
    print("\n" + "=" * 60)
    print("Testing SpectralCNNRegressor")
    print("=" * 60)
    
    regressor = SpectralCNNRegressor(n_bands=12, emb_dim=32)
    x = torch.randn(4, 12, 64, 64)
    soc = regressor(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {soc.shape}")
    assert soc.shape == (4, 1), f"Expected (4, 1), got {soc.shape}"
    
    print("\n✅ All tests passed!")
