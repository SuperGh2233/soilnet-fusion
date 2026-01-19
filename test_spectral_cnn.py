"""
测试 MultiPreprocSpectralCNN 模块的集成

基于 Tziolas et al. (Geoderma, 2024) 的多预处理光谱 CNN 实现测试

使用方法:
    python test_spectral_cnn.py
"""

import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_spectral_cnn_module():
    """测试 MultiPreprocSpectralCNN 模块"""
    print("=" * 70)
    print("测试 1: MultiPreprocSpectralCNN 模块")
    print("=" * 70)
    
    from soilnet.submodules.spectral_cnn import MultiPreprocSpectralCNN, SpectralCNNRegressor
    
    # 测试不同波段数
    for n_bands in [9, 12, 21]:
        print(f"\n--- 波段数: {n_bands} ---")
        
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
        print(f"  4D 输入: {x_4d.shape} -> 输出: {out_4d.shape}")
        assert out_4d.shape == (batch_size, 32), f"期望 (4, 32), 得到 {out_4d.shape}"
        
        # 测试 2D 输入 (已池化)
        x_2d = torch.randn(batch_size, n_bands)
        out_2d = model(x_2d)
        print(f"  2D 输入: {x_2d.shape} -> 输出: {out_2d.shape}")
        assert out_2d.shape == (batch_size, 32), f"期望 (4, 32), 得到 {out_2d.shape}"
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {total_params:,}")
    
    print("\n[OK] MultiPreprocSpectralCNN 测试通过!")
    return True


def test_soilnet_with_spectral_cnn():
    """测试 SoilNet 与 spectral_cnn 编码器的集成"""
    print("\n" + "=" * 70)
    print("测试 2: SoilNet + spectral_cnn 编码器")
    print("=" * 70)
    
    from soilnet.soil_net import SoilNet
    
    n_bands = 12
    batch_size = 4
    
    # 使用 spectral_cnn 编码器
    model = SoilNet(
        use_glam=False,
        cnn_arch="ViT",  # 这个参数在 spectral_cnn 模式下会被忽略
        cnn_in_channels=n_bands,
        regresor_input_from_cnn=384,  # 这个参数在 spectral_cnn 模式下会被覆盖
        img_encoder_type="spectral_cnn",
        spectral_cnn_emb_dim=32
    )
    
    x = torch.randn(batch_size, n_bands, 64, 64)
    out = model(x)
    
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    assert out.shape == (batch_size, 1), f"期望 (4, 1), 得到 {out.shape}"
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    print("\n[OK] SoilNet + spectral_cnn 测试通过!")
    return True


def test_soilnet_lstm_with_spectral_cnn():
    """测试 SoilNetLSTM 与 spectral_cnn 编码器的集成"""
    print("\n" + "=" * 70)
    print("测试 3: SoilNetLSTM + spectral_cnn 编码器")
    print("=" * 70)
    
    from soilnet.soil_net import SoilNetLSTM
    
    n_bands = 12
    n_climate_features = 17
    seq_len = 60
    batch_size = 4
    
    # 使用 spectral_cnn 编码器
    model = SoilNetLSTM(
        use_glam=False,
        cnn_arch="ViT",  # 这个参数在 spectral_cnn 模式下会被忽略
        cnn_in_channels=n_bands,
        regresor_input_from_cnn=384,  # 这个参数在 spectral_cnn 模式下会被覆盖
        lstm_n_features=n_climate_features,
        lstm_n_layers=2,
        lstm_out=128,
        hidden_size=128,
        rnn_arch="Transformer",
        seq_len=seq_len,
        img_encoder_type="spectral_cnn",
        spectral_cnn_emb_dim=32
    )
    
    x_img = torch.randn(batch_size, n_bands, 64, 64)
    x_climate = torch.randn(batch_size, seq_len, n_climate_features)
    
    out = model((x_img, x_climate))
    
    print(f"  图像输入: {x_img.shape}")
    print(f"  气候输入: {x_climate.shape}")
    print(f"  输出: {out.shape}")
    assert out.shape == (batch_size, 1), f"期望 (4, 1), 得到 {out.shape}"
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    # 对比原始 ViT 编码器的参数量
    print("\n--- 对比: 使用 ViT 编码器 ---")
    model_vit = SoilNetLSTM(
        use_glam=False,
        cnn_arch="ViT",
        cnn_in_channels=n_bands,
        regresor_input_from_cnn=768,
        lstm_n_features=n_climate_features,
        lstm_n_layers=2,
        lstm_out=128,
        hidden_size=128,
        rnn_arch="Transformer",
        seq_len=seq_len,
        img_encoder_type="cnn"  # 默认 CNN 模式
    )
    
    vit_params = sum(p.numel() for p in model_vit.parameters())
    print(f"  ViT 总参数量: {vit_params:,}")
    print(f"  SpectralCNN 参数量: {total_params:,}")
    print(f"  参数量减少: {(vit_params - total_params) / vit_params * 100:.1f}%")
    
    print("\n[OK] SoilNetLSTM + spectral_cnn 测试通过!")
    return True


def test_gradient_flow():
    """测试梯度流动"""
    print("\n" + "=" * 70)
    print("测试 4: 梯度流动测试")
    print("=" * 70)
    
    from soilnet.soil_net import SoilNetLSTM
    
    n_bands = 12
    n_climate_features = 17
    seq_len = 60
    batch_size = 4
    
    model = SoilNetLSTM(
        cnn_in_channels=n_bands,
        lstm_n_features=n_climate_features,
        seq_len=seq_len,
        img_encoder_type="spectral_cnn",
        spectral_cnn_emb_dim=32
    )
    
    x_img = torch.randn(batch_size, n_bands, 64, 64, requires_grad=True)
    x_climate = torch.randn(batch_size, seq_len, n_climate_features, requires_grad=True)
    y_true = torch.randn(batch_size, 1)
    
    # 前向传播
    y_pred = model((x_img, x_climate))
    
    # 计算损失
    loss = torch.nn.functional.mse_loss(y_pred, y_true)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    print(f"  图像输入梯度范数: {x_img.grad.norm().item():.6f}")
    print(f"  气候输入梯度范数: {x_climate.grad.norm().item():.6f}")
    
    # 检查模型参数梯度
    has_nan_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"  [WARN] {name} 梯度包含 NaN!")
                has_nan_grad = True
    
    if not has_nan_grad:
        print("  所有参数梯度正常 (无 NaN)")
    
    print("\n[OK] 梯度流动测试通过!")
    return True


def print_usage_example():
    """打印使用示例"""
    print("\n" + "=" * 70)
    print("使用示例")
    print("=" * 70)
    
    example = '''
# ========== 方法 1: 在 train.py 中通过命令行参数切换 ==========

# 使用默认 CNN/ViT 编码器
python train.py -e my_exp -lstm -rnn Transformer -cnn ViT-CoMer

# 使用 SpectralCNN 编码器 (基于 Tziolas et al., Geoderma 2024)
python train.py -e my_exp_spectral -lstm -rnn Transformer --img_encoder spectral_cnn --spectral_emb_dim 32

# ========== 方法 2: 在代码中直接使用 ==========

from soilnet.soil_net import SoilNetLSTM

# 使用 SpectralCNN 编码器
model = SoilNetLSTM(
    cnn_in_channels=12,
    lstm_n_features=17,
    seq_len=60,
    rnn_arch="Transformer",
    img_encoder_type="spectral_cnn",  # 关键参数
    spectral_cnn_emb_dim=32
)

# 使用默认 ViT 编码器
model = SoilNetLSTM(
    cnn_arch="ViT-CoMer",
    cnn_in_channels=12,
    lstm_n_features=17,
    seq_len=60,
    rnn_arch="Transformer",
    img_encoder_type="cnn"  # 默认值
)

# ========== 方法 3: 单独使用 SpectralCNN 模块 ==========

from soilnet.submodules.spectral_cnn import MultiPreprocSpectralCNN

# 创建光谱 CNN 编码器
encoder = MultiPreprocSpectralCNN(
    n_bands=12,
    emb_dim=32,
    hidden_dims=(32, 8),
    dropout=0.2
)

# 输入可以是 patch 或已池化的光谱
x_patch = torch.randn(4, 12, 64, 64)  # (B, C, H, W)
embedding = encoder(x_patch)  # (B, 32)

x_spectrum = torch.randn(4, 12)  # (B, C) 已池化
embedding = encoder(x_spectrum)  # (B, 32)
'''
    print(example)


if __name__ == '__main__':
    print("=" * 70)
    print("MultiPreprocSpectralCNN 集成测试")
    print("基于 Tziolas et al. (Geoderma, 2024)")
    print("=" * 70)
    
    try:
        test_spectral_cnn_module()
        test_soilnet_with_spectral_cnn()
        test_soilnet_lstm_with_spectral_cnn()
        test_gradient_flow()
        print_usage_example()
        
        print("\n" + "=" * 70)
        print("[SUCCESS] 所有测试通过!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

