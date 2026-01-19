import torch
import torch.nn as nn
from submodules.cnn_feature_extractor import CNNFlattener64, CNNFlattener128,\
                                                ResNet101, ResNet101GLAM,\
                                                ResNet50,\
                                                    VGG16, VGG16GLAM
from submodules.vit import VisionTransformer as ViT
from submodules.vit_comer import ViTCoMer
from submodules.vit_comerv2 import ViTCoMerV2
from submodules.vit_mpvit import MPViT
from submodules.vit_hybrid import HybridViT
from submodules.regressor import Regressor, MultiHeadRegressor
from submodules.spectral_enhancement import SpectralEnhancer
from typing import Tuple, Optional
from submodules.src.transformer.transformer import TSTransformerEncoderClassiregressor

# 导入新的增强模块
from submodules.enhanced_climate_transformer import EnhancedClimateTransformer
from submodules.cross_modal_fusion import CrossModalAttention, GatedFusion, HierarchicalFusion
from submodules.region_embedding import RegionEmbedding, RegionalFusion


class EnhancedSoilNetLSTM(nn.Module):
    """增强的SoilNet模型，集成改进的气候特征提取和融合"""
    def __init__(self, use_glam=False, cnn_arch="resnet101", reg_version=1,
                 cnn_in_channels=12, regresor_input_from_cnn=1024, 
                 lstm_n_features=10, lstm_n_layers=2, lstm_out=128, 
                 hidden_size=128, rnn_arch="LSTM", seq_len=61, img_size=64,
                 use_enhanced_climate=True, use_cross_modal_fusion=True,
                 fusion_type="hierarchical",
                 use_spectral_enhance: bool = False, spectral_type: str = 'hybrid',
                 use_regional_adaptation: bool = False, num_regions: int = 16,
                 regional_fusion_type: str = 'attention'):
        super().__init__()
        
        self.use_enhanced_climate = use_enhanced_climate
        self.use_cross_modal_fusion = use_cross_modal_fusion
        self.fusion_type = fusion_type
        self.cnn_arch = cnn_arch
        self.use_regional_adaptation = use_regional_adaptation
        self.num_regions = num_regions
        
        # CNN特征提取器 (保持原有架构)
        if use_glam:
            if cnn_arch == "resnet101":
                self.cnn = ResNet101GLAM(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "vgg16":
                self.cnn = VGG16(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "ViT":
                raise ValueError("ViT is not supported when GLAM is enabled. Please choose from 'resnet' or 'vgg16' or disable GLAM.")
            else:
                raise ValueError("Invalid CNN Architecture. Please choose from 'resnet', 'vgg16', 'ViT', 'ViT-CoMer', 'ViT-CoMerV2', 'MPViT', or 'HybridViT'.")
        else:
            if cnn_arch == "resnet101":
                self.cnn = ResNet101(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "resnet50":
                self.cnn = ResNet50(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "vgg16":
                self.cnn = VGG16GLAM(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "ViT":
                self.cnn = ViT(img_size=img_size, patch_size=8, in_chans=cnn_in_channels, n_classes=regresor_input_from_cnn, p=0.1, attn_p=0.1)
            elif cnn_arch == "ViT-CoMer":
                self.cnn = ViTCoMer(in_chans=cnn_in_channels, num_classes=regresor_input_from_cnn,
                                    patch_size=16, embed_dim=768, depth=6, heads=8, drop_path_rate=0.1)
            elif cnn_arch == "ViT-CoMerV2":
                self.cnn = ViTCoMerV2(in_chans=cnn_in_channels, num_classes=regresor_input_from_cnn, patch_size=16, embed_dim=768, depth=6, heads=8)
            elif cnn_arch == "MPViT":
                self.cnn = MPViT(img_size=img_size, patch_size=16, in_chans=cnn_in_channels, num_classes=regresor_input_from_cnn, 
                                embed_dim=768, depth=6, num_heads=8, mlp_ratio=4, num_paths=3)
            elif cnn_arch == "HybridViT":
                self.cnn = HybridViT(img_size=img_size, patch_size=16, in_chans=cnn_in_channels, num_classes=regresor_input_from_cnn, 
                                    embed_dim=768, depths=[2, 2, 6], num_heads=8, mlp_ratio=4, num_paths=3)
            else:
                raise ValueError("Invalid CNN Architecture. Please choose from 'resnet', 'vgg16', 'ViT', 'ViT-CoMer', 'ViT-CoMerV2', 'MPViT', or 'HybridViT'.")

        # Spectral enhancement
        self.use_spectral_enhance = use_spectral_enhance
        if self.use_spectral_enhance:
            self.spectral = SpectralEnhancer(cnn_in_channels, spectral_type=spectral_type)

        # 气候特征处理 (选择增强或原始)
        if use_enhanced_climate:
            # 增强的气候Transformer
            self.climate_encoder = EnhancedClimateTransformer(
                feat_dim=lstm_n_features, d_model=lstm_out, 
                num_layers=lstm_n_layers, seq_len=seq_len
            )
        else:
            # 原始的气候处理
            if rnn_arch == "LSTM":
                self.climate_encoder = rnn.LSTM(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
            elif rnn_arch == "GRU":
                self.climate_encoder = rnn.GRU(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
            elif rnn_arch == "RNN":
                self.climate_encoder = rnn.RNN(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
            elif rnn_arch == "Transformer":
                self.climate_encoder = TSTransformerEncoderClassiregressor(
                    feat_dim=lstm_n_features, max_len=seq_len, d_model=512,
                    n_heads=8, num_layers=6, dim_feedforward=2048, 
                    num_classes=lstm_out, dropout=0.1, pos_encoding="fixed",
                    activation="gelu", norm="BatchNorm", freeze=False
                )
            else:
                raise ValueError("Invalid RNN Architecture. Please choose from 'LSTM', 'GRU' or 'RNN'.")

        # 交叉模态融合 (如果启用)
        if use_cross_modal_fusion:
            # 获取ViT输出维度
            if "ViT" in cnn_arch:
                vit_dim = 384 if cnn_arch in ["ViT-CoMer", "ViT-CoMerV2", "MPViT", "HybridViT"] else 768
            else:
                vit_dim = regresor_input_from_cnn
            
            climate_dim = lstm_out
            fusion_dim = 256
            
            if fusion_type == "cross_modal":
                self.fusion_module = CrossModalAttention(vit_dim, climate_dim, fusion_dim)
            elif fusion_type == "gated":
                self.fusion_module = GatedFusion(vit_dim, climate_dim, fusion_dim)
            elif fusion_type == "hierarchical":
                self.fusion_module = HierarchicalFusion(vit_dim, climate_dim, fusion_dim)
            else:
                raise ValueError("Invalid fusion type. Choose from 'cross_modal', 'gated', or 'hierarchical'")
            
            # 调整回归器输入维度
            regressor_input_dim = fusion_dim
        else:
            # 原始融合方式
            regressor_input_dim = regresor_input_from_cnn + lstm_out

        # 区域自适应模块 (如果启用)
        if use_regional_adaptation:
            # 确定区域嵌入维度 - 回退到原始维度设置
            if "ViT" in cnn_arch:
                # 恢复ViT区域嵌入维度：使用384维
                region_embed_dim = 384 if cnn_arch in ["ViT-CoMer", "ViT-CoMerV2", "MPViT", "HybridViT"] else 768
            else:
                # 恢复CNN区域嵌入维度：使用1024维
                region_embed_dim = max(regresor_input_from_cnn, 1024)
            
            self.region_embedding = RegionEmbedding(num_regions, region_embed_dim)
            
            # 区域感知融合
            if use_cross_modal_fusion:
                # 如果已经使用了交叉模态融合，添加区域信息
                self.regional_fusion = RegionalFusion(
                    image_dim=fusion_dim,
                    climate_dim=lstm_out, 
                    region_dim=region_embed_dim,
                    hidden_dim=fusion_dim,
                    fusion_type=regional_fusion_type
                )
                regressor_input_dim = fusion_dim
            else:
                # 直接融合图像、气候和区域特征
                self.regional_fusion = RegionalFusion(
                    image_dim=regresor_input_from_cnn,
                    climate_dim=lstm_out,
                    region_dim=region_embed_dim, 
                    hidden_dim=hidden_size,
                    fusion_type=regional_fusion_type
                )
                regressor_input_dim = hidden_size

        # 回归器
        self.reg = MultiHeadRegressor(regressor_input_dim, hidden_size=hidden_size, version=reg_version)
        
    def forward(self, input_raster_ts: Tuple[torch.Tensor, torch.Tensor], 
                region_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the enhanced SoilNet model.
        
        Args:
            input_raster_ts: Tuple of (raster_stack, climate_data)
                - raster_stack: [B, C, H, W] - 遥感影像
                - climate_data: [B, T, F] - 气候时间序列数据
            region_ids: Optional tensor [B] - 区域ID，用于区域自适应
        
        Returns:
            torch.Tensor: Predicted SOC values [B, 1]
        """
        raster_stack, climate_data = input_raster_ts
        
        # 1. CNN特征提取
        if self.use_spectral_enhance:
            raster_stack = self.spectral(raster_stack)
        cnn_features = self.cnn(raster_stack)  # [B, regresor_input_from_cnn]
        
        # 2. 气候特征提取
        if self.use_enhanced_climate:
            climate_features = self.climate_encoder(climate_data)  # [B, lstm_out]
        else:
            climate_features = self.climate_encoder(climate_data)  # [B, lstm_out]
        
        # 3. 特征融合
        if self.use_cross_modal_fusion:
            # 调整维度以匹配融合模块的期望输入
            if "ViT" in self.cnn_arch:
                # 对于ViT模型，使用全局特征
                vit_features = cnn_features.unsqueeze(1)  # [B, 1, D]
            else:
                # 对于CNN模型，将特征重塑为序列形式
                vit_features = cnn_features.unsqueeze(1)  # [B, 1, D]
            
            # 调整气候特征维度
            climate_features_seq = climate_features.unsqueeze(1)  # [B, 1, climate_dim]
            
            # 交叉模态融合
            fused_features = self.fusion_module(vit_features, climate_features_seq)  # [B, N, fusion_dim]
            
            # 全局平均池化
            fused_features = fused_features.mean(dim=1)  # [B, fusion_dim]
        else:
            # 原始融合方式：简单拼接
            fused_features = torch.cat([cnn_features, climate_features], dim=1)
        
        # 4. 区域自适应 (如果启用)
        if self.use_regional_adaptation and region_ids is not None:
            # 获取区域嵌入
            region_features = self.region_embedding(region_ids)  # [B, region_embed_dim]
            
            # 区域感知融合
            if self.use_cross_modal_fusion:
                # 如果已经使用了交叉模态融合，添加区域信息
                fused_features = self.regional_fusion(fused_features, climate_features, region_features)
            else:
                # 直接融合图像、气候和区域特征
                fused_features = self.regional_fusion(cnn_features, climate_features, region_features)
        
        # 5. 回归预测
        output = self.reg(fused_features)
        
        return output


if __name__ == "__main__":
    # 测试代码
    print("=== 测试增强SoilNet模型 ===")
    
    try:
        model = EnhancedSoilNetLSTM(
            cnn_arch="HybridViT",
            rnn_arch="Transformer",
            cnn_in_channels=12,  # 修复：改为12个通道
            regresor_input_from_cnn=384,  # 修复：匹配HybridViT的输出维度
            img_size=64,  # 修复：使用更小的图像尺寸
            use_enhanced_climate=True,
            use_cross_modal_fusion=True,
            fusion_type="hierarchical"
        )
        
        # 创建测试数据
        batch_size = 4
        raster_data = torch.randn(batch_size, 12, 64, 64)  # 修复：使用64x64图像尺寸
        climate_data = torch.randn(batch_size, 61, 10)        # [B, T, F]
        
        print(f"✅ 模型创建成功")
        print(f"   遥感输入形状: {raster_data.shape}")
        print(f"   气候输入形状: {climate_data.shape}")
        
        # 前向传播
        output = model((raster_data, climate_data))
        
        print(f"✅ 前向传播成功")
        print(f"   输出形状: {output.shape}")
        print(f"   模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        print("\n🎉 增强SoilNet模型测试完全成功！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc() 