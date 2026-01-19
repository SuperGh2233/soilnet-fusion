import torch
import torch.nn as nn
import torch.nn.functional as F
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
from submodules.spectral_cnn import MultiPreprocSpectralCNN  # 基于 Tziolas et al. (Geoderma, 2024)
from submodules.semantic_aligned_fusion import SemanticAlignedFusion, SemanticAlignedFusionParallel, SemanticAlignmentLoss  # S-CMRL 融合
from typing import Tuple, Optional
from submodules.src.transformer.transformer import TSTransformerEncoderClassiregressor
from submodules import rnn

class SoilNet(nn.Module):
    """
    SoilNet 模型 - 仅使用图像分支的 SOC 预测模型
    
    Args:
        use_glam: 是否使用 GLAM 注意力机制
        cnn_arch: CNN 架构类型
        reg_version: 回归器版本
        cnn_in_channels: 输入图像通道数
        regresor_input_from_cnn: CNN 输出维度
        hidden_size: 隐藏层大小
        img_size: 输入图像尺寸
        use_spectral_enhance: 是否使用光谱增强
        spectral_type: 光谱增强类型
        use_regional_adaptation: 是否使用区域自适应
        num_regions: 区域数量
        pretrained_path: 预训练权重路径
        img_encoder_type: 图像编码器类型，可选 "cnn" (默认，使用 cnn_arch) 或 "spectral_cnn"
                          (使用基于 Tziolas et al. 的多预处理光谱 CNN)
        spectral_cnn_emb_dim: spectral_cnn 输出维度，默认 32
    """
    def __init__(self, use_glam = False , cnn_arch = "resnet101", reg_version = 1,
                 cnn_in_channels = 14 ,regresor_input_from_cnn = 1024, hidden_size=128, img_size = 64,
                 use_spectral_enhance: bool = False, spectral_type: str = 'hybrid',
                 use_regional_adaptation: bool = False, num_regions: int = 16,
                 pretrained_path: str = None,
                 img_encoder_type: str = "cnn",
                 spectral_cnn_emb_dim: int = 32):
        super().__init__()
        
        # 保存通道数供后续适配使用
        self.cnn_in_channels = cnn_in_channels
        self.img_encoder_type = img_encoder_type
        
        # ========== 图像编码器选择 ==========
        # 基于 Tziolas et al. (Geoderma, 2024) 的多预处理光谱 CNN
        if img_encoder_type == "spectral_cnn":
            print(f"[Info] Using MultiPreprocSpectralCNN (based on Tziolas et al., Geoderma 2024)")
            print(f"       n_bands={cnn_in_channels}, emb_dim={spectral_cnn_emb_dim}")
            self.cnn = MultiPreprocSpectralCNN(
                n_bands=cnn_in_channels,
                emb_dim=spectral_cnn_emb_dim,
                hidden_dims=(32, 8),
                dropout=0.2
            )
            # 更新回归器输入维度
            regresor_input_from_cnn = spectral_cnn_emb_dim
        elif img_encoder_type == "cnn":
            # 原有的 CNN/ViT 编码器逻辑
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
                    # 加载预训练权重
                    if pretrained_path:
                        self._load_pretrained_weights(pretrained_path)
                elif cnn_arch == "ViT-CoMer":
                    # 使用 DeiT-Base 配置以匹配预训练权重: embed_dim=768, depth=12, heads=12
                    self.cnn = ViTCoMer(in_chans=cnn_in_channels, num_classes=regresor_input_from_cnn,
                                        patch_size=16, embed_dim=768, depth=12, heads=12, drop_path_rate=0.1)
                    # 加载预训练权重
                    if pretrained_path:
                        self._load_pretrained_weights(pretrained_path)
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
        else:
            raise ValueError(f"Invalid img_encoder_type: {img_encoder_type}. Choose from 'cnn' or 'spectral_cnn'.")
            
        # Spectral enhancement (before CNN) if using conv/ViT with patch embedding expects channels-first inputs
        self.use_spectral_enhance = use_spectral_enhance
        if self.use_spectral_enhance:
            self.spectral = SpectralEnhancer(cnn_in_channels, spectral_type=spectral_type)

        # Regional adaptation
        self.use_regional_adaptation = use_regional_adaptation
        if self.use_regional_adaptation:
            from submodules.region_embedding import RegionEmbedding
            self.region_embedding = RegionEmbedding(num_regions, regresor_input_from_cnn)
        
        self.reg = MultiHeadRegressor(regresor_input_from_cnn, hidden_size= hidden_size, version=reg_version)

    def _load_pretrained_weights(self, pretrained_path):
        """
        智能加载预训练权重，自动适配通道数并过滤不匹配的键
        支持 ViT 和 ViT-CoMer 模型
        """
        print(f"[Info] Loading pretrained weights from: {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # (A) 提取 state_dict
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 处理可能的 'module.' 前缀（来自 DataParallel 或 DistributedDataParallel）
            if state_dict and list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
            
            # (B) 获取当前模型的键值
            model_dict = self.cnn.state_dict()
            
            # (C) 过滤与适配
            pretrained_dict = {}
            skipped_keys = []
            
            for k, v in state_dict.items():
                # 处理键名前缀差异（例如 timm 可能有 backbone. 前缀）
                model_key = k
                if k.startswith('backbone.'):
                    model_key = k[9:]  # 移除 'backbone.' 前缀
                elif k.startswith('cnn.'):
                    model_key = k[4:]  # 移除 'cnn.' 前缀
                
                # 跳过分类头
                if 'head' in model_key or 'head' in k or 'fc' in model_key or 'fc' in k:
                    continue
                
                if model_key in model_dict:
                    if model_dict[model_key].shape == v.shape:
                        pretrained_dict[model_key] = v
                    else:
                        # 形状不匹配的情况
                        if 'patch_embed.proj.weight' in model_key or 'patch_embed.proj.weight' in k:
                            # v.shape = [out_dim, in_dim, H, W]
                            # model_dict[model_key].shape = [out_dim, in_dim, H, W]
                            # 首先检查输出维度（embed_dim）是否匹配
                            if v.shape[0] != model_dict[model_key].shape[0]:
                                # embed_dim 不匹配，跳过（例如 DeiT-Small 384 vs ViT-CoMer 768）
                                skipped_keys.append(f"{model_key} (embed_dim mismatch: {v.shape[0]} != {model_dict[model_key].shape[0]})")
                            else:
                                # embed_dim 匹配，可以适配输入通道数
                                print(f"[Adapt] Adapting first conv layer: {model_key} from {v.shape} to {model_dict[model_key].shape}")
                                # 假设 v 是 [Out, 3, K, K], 目标是 [Out, 14, K, K]
                                if v.shape[1] == 3 and model_dict[model_key].shape[1] == self.cnn_in_channels:
                                    # 方法1: 平均复制（适用于光谱数据）
                                    avg_weight = torch.mean(v, dim=1, keepdim=True)  # [Out, 1, K, K]
                                    new_weight = avg_weight.repeat(1, self.cnn_in_channels, 1, 1)  # [Out, 14, K, K]
                                    pretrained_dict[model_key] = new_weight
                                elif v.shape[1] < model_dict[model_key].shape[1]:
                                    # 如果预训练权重通道数更少，进行扩展
                                    repeat_times = model_dict[model_key].shape[1] // v.shape[1]
                                    remainder = model_dict[model_key].shape[1] % v.shape[1]
                                    new_weight = v.repeat(1, repeat_times, 1, 1)
                                    if remainder > 0:
                                        new_weight = torch.cat([new_weight, v[:, :remainder, :, :]], dim=1)
                                    pretrained_dict[model_key] = new_weight
                                else:
                                    skipped_keys.append(f"{model_key} (input channel mismatch)")
                        elif 'patch_embed.proj.bias' in model_key or 'patch_embed.proj.bias' in k:
                            # bias 通常可以直接使用（如果输出维度匹配）
                            if v.shape[0] == model_dict[model_key].shape[0]:
                                pretrained_dict[model_key] = v
                            else:
                                skipped_keys.append(f"{model_key} (bias dim mismatch)")
                        elif 'cnn.conv' in model_key or 'cnn.conv' in k:
                            # ViT-CoMer 的 CNN 部分
                            if 'conv.0.weight' in model_key or 'conv.0.weight' in k:
                                # 第一层 CNN
                                if v.shape[1] == 3 and model_dict[model_key].shape[1] == self.cnn_in_channels:
                                    avg_weight = torch.mean(v, dim=1, keepdim=True)
                                    new_weight = avg_weight.repeat(1, self.cnn_in_channels, 1, 1)
                                    pretrained_dict[model_key] = new_weight
                                else:
                                    skipped_keys.append(f"{model_key} (CNN conv shape mismatch)")
                        elif 'pos_embed' in model_key or 'pos_embed' in k:
                            # 如果输入图片尺寸不同，位置编码可能不匹配
                            if v.shape == model_dict[model_key].shape:
                                pretrained_dict[model_key] = v
                            else:
                                # 尝试插值适配位置编码
                                if len(v.shape) == 3 and len(model_dict[model_key].shape) == 3:
                                    # v: [1, N1, D1], model: [1, N2, D2]
                                    if v.shape[2] == model_dict[model_key].shape[2]:  # embed_dim 相同
                                        # 只适配序列长度
                                        if v.shape[1] != model_dict[model_key].shape[1]:
                                            print(f"[Adapt] Interpolating pos_embed from {v.shape} to {model_dict[model_key].shape}")
                                            # 使用插值
                                            v_reshaped = v.permute(0, 2, 1).unsqueeze(3)  # [1, D, N1, 1]
                                            target_n = model_dict[model_key].shape[1]
                                            v_interp = F.interpolate(v_reshaped, size=(target_n, 1), mode='bilinear', align_corners=False)
                                            v_interp = v_interp.squeeze(3).permute(0, 2, 1)  # [1, N2, D]
                                            pretrained_dict[model_key] = v_interp
                                        else:
                                            pretrained_dict[model_key] = v
                                    else:
                                        skipped_keys.append(f"{model_key} (pos_embed embed_dim mismatch)")
                                else:
                                    skipped_keys.append(f"{model_key} (pos_embed shape mismatch)")
                        elif 'blocks' in model_key:
                            # 处理 blocks 中的层，如果 embed_dim 不匹配，跳过
                            if len(v.shape) > 0 and len(model_dict[model_key].shape) > 0:
                                # 检查最后一维（通常是 embed_dim）
                                if len(v.shape) == len(model_dict[model_key].shape):
                                    if v.shape[-1] != model_dict[model_key].shape[-1]:
                                        # embed_dim 不匹配，跳过（例如 DeiT-Small 384 vs ViT-CoMer 768）
                                        skipped_keys.append(f"{model_key} (embed_dim mismatch)")
                                    elif v.shape == model_dict[model_key].shape:
                                        pretrained_dict[model_key] = v
                                    else:
                                        skipped_keys.append(f"{model_key} (shape mismatch)")
                                else:
                                    skipped_keys.append(f"{model_key} (dimension mismatch)")
                            else:
                                skipped_keys.append(f"{model_key} (unexpected shape)")
                        else:
                            skipped_keys.append(f"{model_key} (shape mismatch)")
                else:
                    # 官方权重里有，但您的模型里没有的键（例如反向注入参数）自动被丢弃
                    pass
            
            # (D) 更新权重
            model_dict.update(pretrained_dict)
            self.cnn.load_state_dict(model_dict, strict=False)
            
            # (E) 验证
            total_keys = len(self.cnn.state_dict())
            loaded_keys = len(pretrained_dict)
            print(f"[Success] Loaded {loaded_keys}/{total_keys} keys ({(loaded_keys/total_keys)*100:.1f}%)")
            
            if skipped_keys:
                print(f"[Info] Skipped {len(skipped_keys)} keys due to shape mismatch (this is normal for different model sizes)")
                if len(skipped_keys) <= 10:
                    for sk in skipped_keys:
                        print(f"  - {sk}")
                else:
                    print(f"  (showing first 10 of {len(skipped_keys)} skipped keys)")
                    for sk in skipped_keys[:10]:
                        print(f"  - {sk}")
            
        except Exception as e:
            print(f"[Error] Failed to load pretrained weights: {e}")
            import traceback
            traceback.print_exc()

    def forward(self, raster_stack):
        """
        Forward pass of the Resnet module.
        
        Args:
            raster_stack (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            auxillary_data (torch.Tensor): Auxiliary input tensor of shape (batch_size, aux_size).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        if self.use_spectral_enhance:
            raster_stack = self.spectral(raster_stack)
        flat_raster = self.cnn(raster_stack)
        output = self.reg(flat_raster)
        return output
        
class SoilNetLSTM(nn.Module):
    """
    SoilNetLSTM 模型 - 图像 + 气候数据的多模态 SOC 预测模型
    
    Args:
        use_glam: 是否使用 GLAM 注意力机制
        cnn_arch: CNN 架构类型
        reg_version: 回归器版本
        cnn_in_channels: 输入图像通道数
        regresor_input_from_cnn: CNN 输出维度
        lstm_n_features: 气候数据特征数
        lstm_n_layers: LSTM 层数
        lstm_out: LSTM 输出维度
        hidden_size: 隐藏层大小
        rnn_arch: RNN 架构类型
        seq_len: 序列长度
        img_size: 输入图像尺寸
        use_spectral_enhance: 是否使用光谱增强
        spectral_type: 光谱增强类型
        use_regional_adaptation: 是否使用区域自适应
        num_regions: 区域数量
        region_embed_dim: 区域嵌入维度
        pretrained_path: 预训练权重路径
        img_encoder_type: 图像编码器类型，可选 "cnn" (默认) 或 "spectral_cnn"
                          (使用基于 Tziolas et al. 的多预处理光谱 CNN)
        spectral_cnn_emb_dim: spectral_cnn 输出维度，默认 32
    """
    def __init__(self, use_glam = False  , cnn_arch = "resnet101", reg_version = 1,
                 cnn_in_channels = 14 ,regresor_input_from_cnn = 1024, 
                 lstm_n_features = 10,lstm_n_layers =2, lstm_out = 128, hidden_size=128, rnn_arch = "LSTM", seq_len = 61, img_size = 64,
                 use_spectral_enhance: bool = False, spectral_type: str = 'hybrid',
                 use_regional_adaptation: bool = False, num_regions: int = 16, region_embed_dim: Optional[int] = None,
                 pretrained_path: str = None,
                 img_encoder_type: str = "cnn",
                 spectral_cnn_emb_dim: int = 32,
                 use_scmrl_fusion: bool = False,
                 scmrl_alpha_init: float = 1.5,
                 scmrl_temperature: float = 0.07,
                 scmrl_parallel: bool = False,  # 是否使用并行融合（默认False为串行）
                 static_dim: Optional[int] = None):
        
        super().__init__()
        
        # 保存通道数供后续适配使用
        self.cnn_in_channels = cnn_in_channels
        self.img_encoder_type = img_encoder_type
        
        self.use_regional_adaptation = bool(use_regional_adaptation)
        self.num_regions = int(num_regions)
        self.region_embed_dim = int(region_embed_dim) if region_embed_dim is not None else hidden_size
        
        # S-CMRL 融合选项
        self.use_scmrl_fusion = bool(use_scmrl_fusion)
        self.scmrl_parallel = bool(scmrl_parallel) if use_scmrl_fusion else False
        
        # ========== 图像编码器选择 ==========
        # 基于 Tziolas et al. (Geoderma, 2024) 的多预处理光谱 CNN
        if img_encoder_type == "spectral_cnn":
            print(f"[Info] Using MultiPreprocSpectralCNN (based on Tziolas et al., Geoderma 2024)")
            print(f"       n_bands={cnn_in_channels}, emb_dim={spectral_cnn_emb_dim}")
            self.cnn = MultiPreprocSpectralCNN(
                n_bands=cnn_in_channels,
                emb_dim=spectral_cnn_emb_dim,
                hidden_dims=(32, 8),
                dropout=0.2
            )
            # 更新回归器输入维度
            regresor_input_from_cnn = spectral_cnn_emb_dim
        elif img_encoder_type == "cnn":
            # 原有的 CNN/ViT 编码器逻辑
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
                    # 加载预训练权重
                    if pretrained_path:
                        self._load_pretrained_weights(pretrained_path)
                elif cnn_arch == "ViT-CoMer":
                    # 使用 DeiT-Base 配置以匹配预训练权重: embed_dim=768, depth=12, heads=12
                    self.cnn = ViTCoMer(in_chans=cnn_in_channels, num_classes=regresor_input_from_cnn,
                                        patch_size=16, embed_dim=768, depth=12, heads=12, drop_path_rate=0.1)
                    # 加载预训练权重
                    if pretrained_path:
                        self._load_pretrained_weights(pretrained_path)
                elif cnn_arch == "ViT-CoMerV2":
                    self.cnn = ViTCoMerV2(in_chans=cnn_in_channels, num_classes=regresor_input_from_cnn, patch_size=16, embed_dim=768, depth=12, heads=12)
                elif cnn_arch == "MPViT":
                    self.cnn = MPViT(img_size=img_size, patch_size=16, in_chans=cnn_in_channels, num_classes=regresor_input_from_cnn, 
                                    embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, num_paths=3)
                elif cnn_arch == "HybridViT":
                    self.cnn = HybridViT(img_size=img_size, patch_size=16, in_chans=cnn_in_channels, num_classes=regresor_input_from_cnn, 
                                        embed_dim=768, depths=[2, 2, 6], num_heads=12, mlp_ratio=4, num_paths=3)
                else:
                    raise ValueError("Invalid CNN Architecture. Please choose from 'resnet', 'vgg16', 'ViT', 'ViT-CoMer', 'ViT-CoMerV2', 'MPViT', or 'HybridViT'.")
        else:
            raise ValueError(f"Invalid img_encoder_type: {img_encoder_type}. Choose from 'cnn' or 'spectral_cnn'.")

        # Spectral enhancement
        self.use_spectral_enhance = use_spectral_enhance
        if self.use_spectral_enhance:
            self.spectral = SpectralEnhancer(cnn_in_channels, spectral_type=spectral_type)
        if rnn_arch == "LSTM":
            self.lstm = rnn.LSTM(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "GRU":
            self.lstm = rnn.GRU(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "RNN":
            self.lstm = rnn.RNN(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "Transformer":
            self.lstm = TSTransformerEncoderClassiregressor(
                        feat_dim=lstm_n_features,
                        max_len=seq_len,
                        d_model=512,
                        n_heads=8,
                        num_layers=6,
                        dim_feedforward=2048, 
                        num_classes=lstm_out,
                        dropout=0.1,
                        pos_encoding="fixed",
                        activation="gelu",
                        norm="BatchNorm",
                        freeze=False,
                        )
        else:
            raise ValueError("Invalid RNN Architecture. Please choose from 'LSTM', 'GRU' or 'RNN'.")
        
        if self.use_regional_adaptation:
            from submodules.region_embedding import RegionEmbedding
            self.region_embedding = RegionEmbedding(self.num_regions, self.region_embed_dim)
        else:
            self.region_embedding = None

        # S-CMRL 融合模块（如果启用）
        if self.use_scmrl_fusion:
            fusion_type = "Parallel" if self.scmrl_parallel else "Sequential"
            print(f"[Info] Using S-CMRL Fusion (Semantic-Alignment Cross-Modal Residual Learning)")
            print(f"       Mode: {fusion_type} fusion")
            print(f"       alpha_init={scmrl_alpha_init}, temperature={scmrl_temperature}")
            
            # 创建融合模块（串行或并行）
            if self.scmrl_parallel:
                self.fusion = SemanticAlignedFusionParallel(
                    climate_dim=lstm_out,
                    visual_dim=regresor_input_from_cnn,
                    static_dim=static_dim,
                    num_heads=8,
                    alpha_init=scmrl_alpha_init,
                    learnable_alpha=True,
                    dropout=0.1
                )
            else:
                self.fusion = SemanticAlignedFusion(
                    climate_dim=lstm_out,
                    visual_dim=regresor_input_from_cnn,
                    static_dim=static_dim,
                    num_heads=8,
                    alpha_init=scmrl_alpha_init,
                    learnable_alpha=True,
                    dropout=0.1
                )
            
            # 语义对齐损失（用于训练）
            self.alignment_loss_fn = SemanticAlignmentLoss(temperature=scmrl_temperature)
            
            # 回归头（输入维度与 Climate 相同）
            reg_input_dim = lstm_out
            if self.use_regional_adaptation:
                # 如果有区域自适应，需要额外处理
                reg_input_dim = lstm_out + self.region_embed_dim
            self.reg = nn.Linear(reg_input_dim, 1)
        else:
            # 原有的 MultiHeadRegressor
            reg_input_dims = [regresor_input_from_cnn, lstm_out]
            if self.use_regional_adaptation:
                reg_input_dims.append(self.region_embed_dim)
            self.reg = MultiHeadRegressor(*reg_input_dims, hidden_size= hidden_size, version=reg_version)

    def _load_pretrained_weights(self, pretrained_path):
        """
        智能加载预训练权重，自动适配通道数并过滤不匹配的键
        支持 ViT 和 ViT-CoMer 模型
        """
        print(f"[Info] Loading pretrained weights from: {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # (A) 提取 state_dict
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 处理可能的 'module.' 前缀（来自 DataParallel 或 DistributedDataParallel）
            if state_dict and list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
            
            # (B) 获取当前模型的键值
            model_dict = self.cnn.state_dict()
            
            # (C) 过滤与适配
            pretrained_dict = {}
            skipped_keys = []
            
            for k, v in state_dict.items():
                # 处理键名前缀差异（例如 timm 可能有 backbone. 前缀）
                model_key = k
                if k.startswith('backbone.'):
                    model_key = k[9:]  # 移除 'backbone.' 前缀
                elif k.startswith('cnn.'):
                    model_key = k[4:]  # 移除 'cnn.' 前缀
                
                # 跳过分类头
                if 'head' in model_key or 'head' in k or 'fc' in model_key or 'fc' in k:
                    continue
                
                if model_key in model_dict:
                    if model_dict[model_key].shape == v.shape:
                        pretrained_dict[model_key] = v
                    else:
                        # 形状不匹配的情况
                        if 'patch_embed.proj.weight' in model_key or 'patch_embed.proj.weight' in k:
                            # v.shape = [out_dim, in_dim, H, W]
                            # model_dict[model_key].shape = [out_dim, in_dim, H, W]
                            # 首先检查输出维度（embed_dim）是否匹配
                            if v.shape[0] != model_dict[model_key].shape[0]:
                                # embed_dim 不匹配，跳过（例如 DeiT-Small 384 vs ViT-CoMer 768）
                                skipped_keys.append(f"{model_key} (embed_dim mismatch: {v.shape[0]} != {model_dict[model_key].shape[0]})")
                            else:
                                # embed_dim 匹配，可以适配输入通道数
                                print(f"[Adapt] Adapting first conv layer: {model_key} from {v.shape} to {model_dict[model_key].shape}")
                                # 假设 v 是 [Out, 3, K, K], 目标是 [Out, 14, K, K]
                                if v.shape[1] == 3 and model_dict[model_key].shape[1] == self.cnn_in_channels:
                                    # 方法1: 平均复制（适用于光谱数据）
                                    avg_weight = torch.mean(v, dim=1, keepdim=True)  # [Out, 1, K, K]
                                    new_weight = avg_weight.repeat(1, self.cnn_in_channels, 1, 1)  # [Out, 14, K, K]
                                    pretrained_dict[model_key] = new_weight
                                elif v.shape[1] < model_dict[model_key].shape[1]:
                                    # 如果预训练权重通道数更少，进行扩展
                                    repeat_times = model_dict[model_key].shape[1] // v.shape[1]
                                    remainder = model_dict[model_key].shape[1] % v.shape[1]
                                    new_weight = v.repeat(1, repeat_times, 1, 1)
                                    if remainder > 0:
                                        new_weight = torch.cat([new_weight, v[:, :remainder, :, :]], dim=1)
                                    pretrained_dict[model_key] = new_weight
                                else:
                                    skipped_keys.append(f"{model_key} (input channel mismatch)")
                        elif 'patch_embed.proj.bias' in model_key or 'patch_embed.proj.bias' in k:
                            # bias 通常可以直接使用（如果输出维度匹配）
                            if v.shape[0] == model_dict[model_key].shape[0]:
                                pretrained_dict[model_key] = v
                            else:
                                skipped_keys.append(f"{model_key} (bias dim mismatch)")
                        elif 'cnn.conv' in model_key or 'cnn.conv' in k:
                            # ViT-CoMer 的 CNN 部分
                            if 'conv.0.weight' in model_key or 'conv.0.weight' in k:
                                # 第一层 CNN
                                if v.shape[1] == 3 and model_dict[model_key].shape[1] == self.cnn_in_channels:
                                    avg_weight = torch.mean(v, dim=1, keepdim=True)
                                    new_weight = avg_weight.repeat(1, self.cnn_in_channels, 1, 1)
                                    pretrained_dict[model_key] = new_weight
                                else:
                                    skipped_keys.append(f"{model_key} (CNN conv shape mismatch)")
                        elif 'pos_embed' in model_key or 'pos_embed' in k:
                            # 如果输入图片尺寸不同，位置编码可能不匹配
                            if v.shape == model_dict[model_key].shape:
                                pretrained_dict[model_key] = v
                            else:
                                # 尝试插值适配位置编码
                                if len(v.shape) == 3 and len(model_dict[model_key].shape) == 3:
                                    # v: [1, N1, D1], model: [1, N2, D2]
                                    if v.shape[2] == model_dict[model_key].shape[2]:  # embed_dim 相同
                                        # 只适配序列长度
                                        if v.shape[1] != model_dict[model_key].shape[1]:
                                            print(f"[Adapt] Interpolating pos_embed from {v.shape} to {model_dict[model_key].shape}")
                                            # 使用插值
                                            v_reshaped = v.permute(0, 2, 1).unsqueeze(3)  # [1, D, N1, 1]
                                            target_n = model_dict[model_key].shape[1]
                                            v_interp = F.interpolate(v_reshaped, size=(target_n, 1), mode='bilinear', align_corners=False)
                                            v_interp = v_interp.squeeze(3).permute(0, 2, 1)  # [1, N2, D]
                                            pretrained_dict[model_key] = v_interp
                                        else:
                                            pretrained_dict[model_key] = v
                                    else:
                                        skipped_keys.append(f"{model_key} (pos_embed embed_dim mismatch)")
                                else:
                                    skipped_keys.append(f"{model_key} (pos_embed shape mismatch)")
                        elif 'blocks' in model_key:
                            # 处理 blocks 中的层，如果 embed_dim 不匹配，跳过
                            if len(v.shape) > 0 and len(model_dict[model_key].shape) > 0:
                                # 检查最后一维（通常是 embed_dim）
                                if len(v.shape) == len(model_dict[model_key].shape):
                                    if v.shape[-1] != model_dict[model_key].shape[-1]:
                                        # embed_dim 不匹配，跳过（例如 DeiT-Small 384 vs ViT-CoMer 768）
                                        skipped_keys.append(f"{model_key} (embed_dim mismatch)")
                                    elif v.shape == model_dict[model_key].shape:
                                        pretrained_dict[model_key] = v
                                    else:
                                        skipped_keys.append(f"{model_key} (shape mismatch)")
                                else:
                                    skipped_keys.append(f"{model_key} (dimension mismatch)")
                            else:
                                skipped_keys.append(f"{model_key} (unexpected shape)")
                        else:
                            skipped_keys.append(f"{model_key} (shape mismatch)")
                else:
                    # 官方权重里有，但您的模型里没有的键（例如反向注入参数）自动被丢弃
                    pass
            
            # (D) 更新权重
            model_dict.update(pretrained_dict)
            self.cnn.load_state_dict(model_dict, strict=False)
            
            # (E) 验证
            total_keys = len(self.cnn.state_dict())
            loaded_keys = len(pretrained_dict)
            print(f"[Success] Loaded {loaded_keys}/{total_keys} keys ({(loaded_keys/total_keys)*100:.1f}%)")
            
            if skipped_keys:
                print(f"[Info] Skipped {len(skipped_keys)} keys due to shape mismatch (this is normal for different model sizes)")
                if len(skipped_keys) <= 10:
                    for sk in skipped_keys:
                        print(f"  - {sk}")
                else:
                    print(f"  (showing first 10 of {len(skipped_keys)} skipped keys)")
                    for sk in skipped_keys[:10]:
                        print(f"  - {sk}")
            
        except Exception as e:
            print(f"[Error] Failed to load pretrained weights: {e}")
            import traceback
            traceback.print_exc()
        
    def forward(self, input_raster_ts: Tuple[torch.Tensor, torch.Tensor], region_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inputs
        ------
        input_raster_ts : A tupple containing the following two tensors:
            * raster_stack (torch.Tensor): A 4D tensor of shape `(batch_size, channels, height, width)` representing a stack of raster images.
            * ts_features (torch.Tensor): A 3D tensor of shape `(batch_size, seq_length, , n_features)` representing a sequence of time-series features. | `seq_length` is the number of time steps in the sequence. e.g. months in our climate data
            
        Outputs
        -------
            - output (torch.Tensor): A tensor of shape `(batch_size, 1)` representing the predicted output of regression.
        """
        # 允许输入包含额外元素（例如静态特征等），这里只取前两个（影像、气候）
        if isinstance(input_raster_ts, (list, tuple)):
            if len(input_raster_ts) < 2:
                raise ValueError("input_raster_ts 需要至少包含影像与气候两个张量")
            raster_stack = input_raster_ts[0]
            ts_features = input_raster_ts[1]
        else:
            raise ValueError("input_raster_ts 应为包含(影像, 气候)的 tuple/list")
        if self.use_spectral_enhance:
            raster_stack = self.spectral(raster_stack)
        flat_raster = self.cnn(raster_stack)
        lstm_output = self.lstm(ts_features)
        
        # 保存中间特征（用于对齐损失计算）
        self._last_climate_feat = lstm_output
        self._last_visual_feat = flat_raster
        
        # S-CMRL 融合或原有融合方式
        if self.use_scmrl_fusion:
            # 使用 S-CMRL 融合
            # 注意：这里假设 static_feat 在 input_raster_ts 的第三个元素（如果有）
            static_feat = None
            if isinstance(input_raster_ts, (list, tuple)) and len(input_raster_ts) >= 3:
                static_feat = input_raster_ts[2]
            
            fused_feat = self.fusion(lstm_output, flat_raster, static_feat)
            
            # 如果有区域自适应，需要额外处理
            if self.use_regional_adaptation and (region_ids is not None) and (self.region_embedding is not None):
                region_ids = region_ids.to(fused_feat.device).long()
                region_features = self.region_embedding(region_ids)
                fused_feat = torch.cat([fused_feat, region_features], dim=1)
            
            output = self.reg(fused_feat)
        else:
            # 原有的融合方式
            reg_inputs = [flat_raster, lstm_output]
            if self.use_regional_adaptation and (region_ids is not None) and (self.region_embedding is not None):
                region_ids = region_ids.to(flat_raster.device).long()
                region_features = self.region_embedding(region_ids)
                reg_inputs.append(region_features)
            output = self.reg(*reg_inputs)
        
        return output
    
class SoilNetJustLSTM(SoilNetLSTM):
    """
    This class inherits from SoilNetLSTM but disables the CNN pathway to use only the climate data.
    注意：不创建CNN分支，只使用气候数据，避免浪费内存。
    """
    def __init__(self, use_glam=False, cnn_arch="resnet101", reg_version=1,
                 cnn_in_channels=14, regresor_input_from_cnn=1024, 
                 lstm_n_features=10, lstm_n_layers=2, lstm_out=128, hidden_size=128,
                 rnn_arch="LSTM", seq_len=61, img_size=64):
        
        # 直接初始化nn.Module，不调用super().__init__()，避免创建CNN分支
        nn.Module.__init__(self)
        
        # 保存必要的属性
        self.cnn_in_channels = cnn_in_channels
        self.use_regional_adaptation = False
        self.num_regions = 16
        self.region_embed_dim = hidden_size
        self.use_spectral_enhance = False
        
        # 不创建CNN分支（关键：避免创建ViT等CNN模型）
        self.cnn = None
        
        # 创建RNN/LSTM分支（只创建这部分）
        if rnn_arch == "LSTM":
            self.lstm = rnn.LSTM(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "GRU":
            self.lstm = rnn.GRU(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "RNN":
            self.lstm = rnn.RNN(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "Transformer":
            self.lstm = TSTransformerEncoderClassiregressor(
                        feat_dim=lstm_n_features,
                        max_len=seq_len,
                        d_model=512,
                        n_heads=8,
                        num_layers=6,
                        dim_feedforward=2048, 
                        num_classes=lstm_out,
                        dropout=0.1,
                        pos_encoding="fixed",
                        activation="gelu",
                        norm="BatchNorm",
                        freeze=False,
                        )
        else:
            raise ValueError("Invalid RNN Architecture. Please choose from 'LSTM', 'GRU', 'RNN' or 'Transformer'.")
        
        # 只使用LSTM输出的回归器（不包含CNN特征）
        self.reg = MultiHeadRegressor(lstm_out, hidden_size=hidden_size, version=reg_version)
            
    def forward(self, input_raster_ts: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass using only climate data (no CNN).
        """
        # 只取气候序列，忽略图像数据
        if isinstance(input_raster_ts, (list, tuple)):
            if len(input_raster_ts) < 2:
                raise ValueError("input_raster_ts 需要至少包含影像与气候两个张量")
            ts_features = input_raster_ts[1]  # 只使用气候数据 [B, T, F]
        else:
            raise ValueError("input_raster_ts 应为包含(影像, 气候)的 tuple/list")

        # 只使用LSTM处理气候数据
        lstm_output = self.lstm(ts_features)  # [B, lstm_out]
        output = self.reg(lstm_output)  # [B, 1]

        return output

 
class SoilNetSimCLR(nn.Module):
    def __init__(self, use_glam = False  , cnn_arch = "resnet101", reg_version = 1,
                 cnn_in_channels = 14 ,regresor_input_from_cnn = 128, 
                 lstm_n_features = 10,lstm_n_layers =2, lstm_out = 128, hidden_size=128, rnn_arch = "LSTM", seq_len = 61, img_size = 64):
        
        super().__init__()
        
        if use_glam:
            if cnn_arch == "resnet101":
                self.cnn = ResNet101GLAM(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "vgg16":
                self.cnn = VGG16(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "ViT":
                raise ValueError("ViT is not supported when GLAM is enabled. Please choose from 'resnet' or 'vgg16' or disable GLAM.")
            else:
                raise ValueError("Invalid CNN Architecture. Please choose from 'resnet' or 'vgg16'.")

        else:
            if cnn_arch == "resnet101":
                self.cnn = ResNet101(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            if cnn_arch == "resnet50":
                self.cnn = ResNet50(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "vgg16":
                self.cnn = VGG16GLAM(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "ViT":
                self.cnn = ViT(img_size=img_size, patch_size=8, in_chans=cnn_in_channels, n_classes=regresor_input_from_cnn, p=0.1, attn_p=0.1)
            else:
                raise ValueError("Invalid CNN Architecture. Please choose from 'resnet' or 'vgg16'.")
            

        if rnn_arch == "LSTM":
            self.lstm = rnn.LSTM(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "GRU":
            self.lstm = rnn.GRU(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "RNN":
            self.lstm = rnn.RNN(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "Transformer":
            self.lstm = TSTransformerEncoderClassiregressor(
                        feat_dim=lstm_n_features,
                        max_len=seq_len,
                        d_model=512,
                        n_heads=8,
                        num_layers=6,
                        dim_feedforward=2048, 
                        num_classes=lstm_out,
                        dropout=0.1,
                        pos_encoding="fixed",
                        activation="gelu",
                        norm="BatchNorm",
                        freeze=False,
                        )
        else:
            raise ValueError("Invalid RNN Architecture. Please choose from 'LSTM', 'GRU' or 'RNN'.")
        
        #self.reg = MultiHeadRegressor(regresor_input_from_cnn, lstm_out, hidden_size= hidden_size, version=reg_version)
        
    def forward(self, input_raster_ts: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Inputs
        ------
        input_raster_ts : A tupple containing the following two tensors:
            * raster_stack (torch.Tensor): A 4D tensor of shape `(batch_size, channels, height, width)` representing a stack of raster images.
            * ts_features (torch.Tensor): A 3D tensor of shape `(batch_size, seq_length, , n_features)` representing a sequence of time-series features. | `seq_length` is the number of time steps in the sequence. e.g. months in our climate data
            
        Outputs
        -------
            - output (torch.Tensor, torch.Tensor): A tupples of tensors of shape `(batch_size, 128)` representing the embeddings of image and climate data.
            to be used in SimCLR loss.
        """
        raster_stack, ts_features = input_raster_ts
        flat_raster = self.cnn(raster_stack)
        lstm_output = self.lstm(ts_features)

        return flat_raster, lstm_output
                
    
class SoilNetSimCLRwRegHead(nn.Module):
    def __init__(self, soilnet_simclr: SoilNetSimCLR, cnn_output_dim=128, lstm_output_dim=128, hidden_size=128, reg_version = 1):
        super().__init__()
        self.soilnet_simclr = soilnet_simclr
        # 修复回归头输入维度：第一个是CNN输出维度，第二个是LSTM输出维度
        self.reg = MultiHeadRegressor(cnn_output_dim, lstm_output_dim, hidden_size= hidden_size, version=reg_version)
    def forward(self, input_raster_ts: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Inputs
        ------
        input_raster_ts : A tupple containing the following two tensors:
            * raster_stack (torch.Tensor): A 4D tensor of shape `(batch_size, channels, height, width)` representing a stack of raster images.
            * ts_features (torch.Tensor): A 3D tensor of shape `(batch_size, seq_length, , n_features)` representing a sequence of time-series features. | `seq_length` is the number of time steps in the sequence. e.g. months in our climate data
            
        Outputs
        -------
            - output (torch.Tensor): A tensor of shape `(batch_size, 1)` representing the predicted output of regression.
        """
        raster_stack, ts_features = input_raster_ts
        flat_raster, lstm_output = self.soilnet_simclr((raster_stack, ts_features))
        output = self.reg(flat_raster, lstm_output)
        return output
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
    # print("Testing SoilNet...")
    # x = torch.randn((32,12,128,128))
    # y = torch.rand((32,12))
    # model = SoilNet()
    # z = model(x,y)
    # print(z.detach().shape)
    # print('Testing SoilNetFC...')
    # x = torch.randn((32,12,64,64))
    # model = SoilNetFC(cnn_in_channels=12)
    # y = model(x)
    # print(y.detach().shape)
    
    # print("Testing SoilNetMonoLSTM...")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
    # x_cnn = torch.randn((32,12,64,64)).to(device)
    # x_lstm = torch.randn((32, 12, 10)).to(device)
    # model = SoilNetMonoLSTM(cnn_in_channels=12, lstm_n_features=10).to(device)
    # y= model(x_cnn, x_lstm)
    # print(y.detach().shape)
    
    print("Testing SoilNet...")
    x = torch.randn((32,12,64,64))
    model = SoilNet(cnn_in_channels=12, cnn_arch="ViT")
    y = model(x)
    print(y.detach().shape)
    
    print('Testing SoilNetLSTM...')
    x_cnn = torch.randn((32,12,64,64)).to(device)
    x_lstm = torch.randn((32, 60, 10)).to(device)
    model = SoilNetLSTM(cnn_arch="ViT", cnn_in_channels= 12, regresor_input_from_cnn=1024,
                       lstm_n_features= 10, lstm_n_layers=2, lstm_out=128, hidden_size=128).to(device)
    y= model((x_cnn, x_lstm))
    print(y.detach().shape)
    
    print("Testing SoilNetSimCLR...")
    modelSimCLR = SoilNetSimCLR(cnn_arch="ViT", cnn_in_channels= 12, regresor_input_from_cnn=128,
                       lstm_n_features= 10, lstm_n_layers=2, lstm_out=128, hidden_size=128).to(device)
    y1, y2 = modelSimCLR((x_cnn, x_lstm))
    print(y1.detach().shape, y2.detach().shape)