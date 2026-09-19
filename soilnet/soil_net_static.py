"""
支持静态特征的SoilNet模型
扩展 SoilNetLSTM 以支持土地利用、土壤质地等静态特征
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from soilnet.soil_net import SoilNetLSTM
from soilnet.submodules.regressor import MultiHeadRegressor
from soilnet.static_branch import StaticBranch


class SoilNetLSTMWithStatic(SoilNetLSTM):
    """
    支持静态特征的SoilNetLSTM模型
    在原有架构基础上添加静态特征输入
    """
    
    def __init__(self, static_feature_dim=0, lulc_num_classes: int = 0, lulc_embed_dim: int = 16, **kwargs):
        """
        Args:
            static_feature_dim: 静态特征维度（土地利用、土壤质地等特征的数量）
            **kwargs: 传递给父类的其他参数
        """
        # 移除static_feature_dim，避免传递给父类
        self.static_feature_dim = static_feature_dim
        self.lulc_num_classes = int(lulc_num_classes) if lulc_num_classes is not None else 0
        self.lulc_embed_dim = int(lulc_embed_dim)
        
        # 如果使用SCMRL fusion，需要传入编码后的静态特征维度（与hidden_size相同）
        # 因为StaticBranch会将静态特征编码为hidden_size维度
        use_scmrl = kwargs.get('use_scmrl_fusion', False)
        use_film = kwargs.get('use_film_fusion', False)
        use_baseline_fusion = kwargs.get('fusion_baseline') is not None
        hidden_size = kwargs.get('hidden_size', 128)
        static_hidden_dim = kwargs.get('lstm_out', hidden_size) if kwargs.get('scmrl_checkpoint_compatible', False) else hidden_size
        if (use_scmrl or use_film or use_baseline_fusion) and static_feature_dim > 0:
            # 编码后的静态特征维度 = hidden_size（StaticBranch的输出维度）
            kwargs['static_dim'] = hidden_size
        
        super().__init__(**kwargs)
        
        # 重新定义回归器以支持静态特征
        # 如果static_feature_dim > 0，添加静态特征分支
        if static_feature_dim > 0:
            # 获取回归器输入维度
            cnn_dim = kwargs.get('regresor_input_from_cnn', 1024)
            lstm_dim = kwargs.get('lstm_out', 128)
            hidden_size = kwargs.get('hidden_size', 128)
            reg_version = kwargs.get('reg_version', 1)
            dropout_prob = kwargs.get('dropout_prob', 0.5)
            
            # 静态分支：独立模块（数值 + 可选 LULC embedding）
            static_in_dim = static_feature_dim
            # 保持属性兼容性，避免未定义时报错
            self.lulc_embedding = None
            if static_in_dim > 0:
                self.static_branch = StaticBranch(
                    numeric_dim=static_in_dim,
                    lulc_classes=self.lulc_num_classes,
                    lulc_embed_dim=self.lulc_embed_dim,
                    hidden=static_hidden_dim,
                    dropout=0.3,
                )
                # 若需要在外部检查嵌入可用性，提供同名属性
                self.lulc_embedding = self.static_branch.lulc_embedding
            else:
                self.static_branch = None

            # 检查是否使用SCMRL fusion（父类已经初始化了use_scmrl_fusion属性）
            use_scmrl = getattr(self, 'use_scmrl_fusion', False)
            
            if use_scmrl or getattr(self, 'use_film_fusion', False) or getattr(self, 'fusion_baseline', None):
                # 如果使用SCMRL fusion，回归器接收融合后的单一特征向量（维度为lstm_out）
                reg_input_dim = lstm_dim
                if self.use_regional_adaptation and getattr(self, "region_embedding", None) is not None:
                    reg_input_dim = lstm_dim + self.region_embed_dim
                self.reg = nn.Linear(reg_input_dim, 1)
                print(f"[OK] 已启用静态特征支持 (维度: {static_feature_dim})")
                if self.lulc_embedding is not None:
                    print(f"[OK] 启用LULC嵌入: num_classes={self.lulc_num_classes}, embed_dim={self.lulc_embed_dim}")
            else:
                # 如果不使用SCMRL fusion，使用MultiHeadRegressor接收多个输入
                reg_input_dims = [cnn_dim, lstm_dim]
                static_branch_dim = hidden_size if self.static_branch is not None else static_feature_dim
                if static_branch_dim > 0:
                    reg_input_dims.append(static_branch_dim)
                if self.use_regional_adaptation and getattr(self, "region_embedding", None) is not None:
                    reg_input_dims.append(self.region_embed_dim)
                self.reg = MultiHeadRegressor(
                    *reg_input_dims,
                    hidden_size=hidden_size,
                    activation="sigmoid",
                    version=reg_version,
                    dropout_prob=dropout_prob
                )
                print(f"[OK] 已启用静态特征支持 (维度: {static_feature_dim})")
                if self.lulc_embedding is not None:
                    print(f"[OK] 启用LULC嵌入: num_classes={self.lulc_num_classes}, embed_dim={self.lulc_embed_dim}")
            
            if self.use_regional_adaptation and getattr(self, "region_embedding", None) is not None:
                print(f"[OK] 启用区域自适应: num_regions={self.num_regions}, embed_dim={self.region_embed_dim}")
        else:
            print("[WARN] 未启用静态特征 (static_feature_dim=0)")
    
    def forward(self, input_raster_ts_static: Tuple, region_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with static features
        
        Args:
            input_raster_ts_static: Tuple containing:
                - raster_stack: 影像数据 [B, C, H, W]
                - ts_features: 气候时间序列 [B, T, F]
                - static_features: 静态特征 [B, S] (可选)
        
        Returns:
            output: 预测的SOC值 [B, 1]
        """
        # 解包输入（兼容：两路、三路（旧：直接静态向量）、四路（新：数值静态+LULC索引））
        raster_stack = input_raster_ts_static[0]
        ts_features = input_raster_ts_static[1]
        static_features = None
        lulc_indices = None
        if len(input_raster_ts_static) >= 3:
            static_features = input_raster_ts_static[2]
        if len(input_raster_ts_static) >= 4:
            lulc_indices = input_raster_ts_static[3]
        
        # CNN特征提取
        if self.use_spectral_enhance:
            raster_stack = self.spectral(raster_stack)
        cnn_features, visual_fusion_features = self._extract_visual_features(raster_stack)
        
        # 气候特征提取
        climate_features = self.lstm(ts_features)  # [B, lstm_dim]
        
        # 保存中间特征（用于对齐损失计算，如果使用SCMRL）
        if (hasattr(self, 'use_scmrl_fusion') and self.use_scmrl_fusion) or getattr(self, 'use_film_fusion', False) or getattr(self, 'fusion_baseline', None):
            self._last_climate_feat = climate_features
            self._last_visual_feat = visual_fusion_features
        
        # S-CMRL 融合或原有融合方式
        if (hasattr(self, 'use_scmrl_fusion') and self.use_scmrl_fusion) or getattr(self, 'use_film_fusion', False) or getattr(self, 'fusion_baseline', None):
            # 使用 S-CMRL 融合（静态特征会参与融合）
            # 先使用StaticBranch编码静态特征
            static_encoded = None
            if (static_features is not None) and self.static_feature_dim > 0:
                if (self.static_branch is not None):
                    static_encoded = self.static_branch(static_features, lulc_indices)
                    # 调试：检查静态特征是否有效
                    if hasattr(self, '_static_debug_count'):
                        self._static_debug_count += 1
                    else:
                        self._static_debug_count = 1
                        # 首次检查
                        static_mean = static_features.mean().item()
                        static_std = static_features.std().item()
                        static_is_zero = torch.allclose(static_features, torch.zeros_like(static_features))
                        encoded_mean = static_encoded.mean().item()
                        encoded_std = static_encoded.std().item()
                        encoded_is_zero = torch.allclose(static_encoded, torch.zeros_like(static_encoded))
                        if self._static_debug_count == 1:
                            print(f"\n[DEBUG] 静态特征检查:")
                            print(f"  - 原始静态特征: shape={static_features.shape}, mean={static_mean:.6f}, std={static_std:.6f}, is_all_zero={static_is_zero}")
                            print(f"  - 编码后特征: shape={static_encoded.shape}, mean={encoded_mean:.6f}, std={encoded_std:.6f}, is_all_zero={encoded_is_zero}")
                            if static_is_zero:
                                print(f"  ⚠️  警告: 原始静态特征全为0！")
                            if encoded_is_zero:
                                print(f"  ⚠️  警告: 编码后静态特征全为0！")
                else:
                    # 兼容旧路径：直接将静态向量送入融合
                    static_encoded = static_features
            
            # 检查static_encoded是否为None
            if static_encoded is None:
                if hasattr(self, '_static_none_warning_count'):
                    self._static_none_warning_count += 1
                else:
                    self._static_none_warning_count = 1
                    if self._static_none_warning_count == 1:
                        print(f"\n[DEBUG] ⚠️  警告: static_encoded 为 None，静态特征不会参与SCMRL融合")
                        print(f"  - static_features is None: {static_features is None}")
                        print(f"  - static_feature_dim: {self.static_feature_dim}")
                        print(f"  - static_branch is None: {self.static_branch is None}")
            
            fused_feat = self.fusion(climate_features, visual_fusion_features, static_encoded)
            
            # 如果有区域自适应，需要额外处理
            if self.use_regional_adaptation and getattr(self, "region_embedding", None) is not None and region_ids is not None:
                region_ids = region_ids.to(fused_feat.device).long()
                region_features = self.region_embedding(region_ids)
                fused_feat = torch.cat([fused_feat, region_features], dim=1)
            
            output = self.reg(fused_feat)
        else:
            # 回归预测（不使用SCMRL fusion）
            if (static_features is not None) and self.static_feature_dim > 0:
                if (self.static_branch is not None):
                    static_encoded = self.static_branch(static_features, lulc_indices)
                    reg_inputs = [cnn_features, climate_features, static_encoded]
                else:
                    # 兼容旧路径：直接将静态向量送入回归头
                    reg_inputs = [cnn_features, climate_features, static_features]
            else:
                # 两输入回归器（原有逻辑）
                reg_inputs = [cnn_features, climate_features]

            if self.use_regional_adaptation and getattr(self, "region_embedding", None) is not None and region_ids is not None:
                region_ids = region_ids.to(cnn_features.device).long()
                region_features = self.region_embedding(region_ids)
                reg_inputs.append(region_features)

            output = self.reg(*reg_inputs)
        
        return output
