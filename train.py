from train_utils import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import random
import numpy as np
import pandas as pd
from dataset.utils.utils import TextColors as tc
from plot_utils.plot import plot_train_test_losses
from datetime import date, datetime
import torch.nn.functional as F
import cv2
import json
import warnings
import config
from soilnet.soil_net import SoilNet, SoilNetLSTM, SoilNetSimCLRwRegHead, SoilNetJustLSTM
import csv
import train_utils
from train_utils import *
from datetime import date, datetime
import argparse
from pathlib import Path
# Format the date and time
# create a folder called 'results' in the current directory if it doesn't exist
if not os.path.exists('results'):
	os.mkdir('results')
 
# Format the date and time
now = datetime.now()
start_string = now.strftime("%Y-%m-%d %H:%M:%S")
# print("Current Date and Time:", start_string)
# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

train_l8_folder_path = config.train_l8_folder_path
test_l8_folder_path = config.test_l8_folder_path
val_l8_folder_path = config.val_l8_folder_path
lucas_csv_path = config.lucas_csv_path
climate_csv_folder_path = config.climate_csv_folder_path
SIMCLR_PATH = config.SIMCLR_PATH

EXP_NAME = 'LUCAS_Transformer_NoImage'
# 默认数据集可从 config.default_dataset 提供；若未设置则退回 'CHINA'
DATASET = getattr(config, 'default_dataset', 'CHINA')  # 'LUCAS', 'RaCA', 'CHINA'
NUM_WORKERS = 2
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0  # disable weight decay
NUM_EPOCHS = 2
LR_SCHEDULER = "step" # step, plateau or None
USE_SRTM = False
USE_SPATIAL_ATTENTION = False
CNN_ARCHITECTURE = "ViT" # vgg16 or resnet101 or "ViT" or resnet50 or "ViT-CoMer" or "MPViT" or "HybridViT"
RNN_ARCHITECTURE = None  # 默认不指定RNN架构，需要通过 -rnn 显式指定才会开启RNN分支
IMG_ENCODER_TYPE = "cnn"  # 图像编码器类型：'cnn' (默认) 或 'spectral_cnn' (基于 Tziolas et al., Geoderma 2024)
SPECTRAL_EMB_DIM = 32  # spectral_cnn 输出维度
REG_VERSION = 1
SEEDS = [1,] # Seeds for cross-validation and reproducibility
USE_LSTM_BRANCH = False
LOG_LOSS = False
SAVE_TRAIN_DATA_METRICS = False
LOAD_SIMCLR_MODEL = False
JUST_LSTM = False # Using Only Climate Data

# 增强功能参数（新增）
USE_ENHANCED_CLIMATE = False  # 使用增强气候特征提取
USE_CROSS_MODAL_FUSION = False  # 使用交叉模态融合
FUSION_TYPE = "hierarchical"  # 融合类型：cross_modal, gated, hierarchical

# 学习率调度器设置
LR_SCHEDULER = "step"  # 默认使用step调度器

def parse_arguments():
	parser = argparse.ArgumentParser(description='SoilNet Training')
	parser.add_argument('-e', '--exp_name', type=str, default=EXP_NAME, help='Experiment name - helps to identify the experiment')
	parser.add_argument('-d', '--dataset', type=str, default=DATASET, choices=['LUCAS', 'RaCA', 'CHINA'], help='Dataset name to use (default from config.default_dataset if set, else CHINA)')
	parser.add_argument('-w', '--num_workers', type=int, default=NUM_WORKERS, help='Number of workers for data loading')
	parser.add_argument('-trbs', '--train_batch_size', type=int, default=TRAIN_BATCH_SIZE, help='Batch size for training')
	parser.add_argument('-tsbs', '--test_batch_size', type=int, default=TEST_BATCH_SIZE, help='Batch size for testing')
	parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
	parser.add_argument('-wd', '--weight_decay', type=float, default=WEIGHT_DECAY, help='Weight decay (set larger than default for ViT small images)')
	parser.add_argument('-ne', '--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
	parser.add_argument('-ls', '--lr_scheduler', type=str, default=LR_SCHEDULER, choices=['step', 'plateau', 'None'], help='Learning rate scheduler')
	parser.add_argument('-srtm', '--use_srtm', action='store_true', default=USE_SRTM, help='Use SRTM data')
	parser.add_argument('-sa', '--use_spatial_attention', action='store_true', default=USE_SPATIAL_ATTENTION, help='Use spatial attention')
	parser.add_argument('-cnn', '--cnn_architecture', type=str, default=CNN_ARCHITECTURE, choices=['vgg16', 'resnet101', 'ViT', 'resnet50', 'ViT-CoMer', 'ViT-CoMerV2', 'MPViT', 'HybridViT'], help='CNN architecture')
	parser.add_argument('-rnn', '--rnn_architecture', type=str, default=None, choices=['LSTM', 'GRU', 'RNN', 'Transformer'], help='RNN architecture (if specified, will automatically enable RNN branch)')
	# 图像编码器类型开关 - 基于 Tziolas et al. (Geoderma, 2024)
	parser.add_argument('--img_encoder', type=str, default='cnn', choices=['cnn', 'spectral_cnn'],
						help='Image encoder type: "cnn" (default, uses --cnn_architecture) or "spectral_cnn" (Multi-preprocessed Spectral CNN based on Tziolas et al., Geoderma 2024)')
	parser.add_argument('--spectral_emb_dim', type=int, default=32, help='Embedding dimension for spectral_cnn encoder (default: 32)')
	parser.add_argument('-rv', '--reg_version', type=int, default=REG_VERSION, help='Regression version')
	parser.add_argument('-seed', '--seeds', nargs='+', type=int, default=SEEDS, help='Seeds for cross-validation. input example: 1 2 3 4 5')
	parser.add_argument('-lstm', '--use_lstm_branch', action='store_true', default=USE_LSTM_BRANCH, help='Use Cliamte data - I know! the name is misleading')
	parser.add_argument('-log', '--log_loss', action='store_true', default=LOG_LOSS, help='Use logarithmic loss')
	parser.add_argument('-stm', '--save_train_data_metrics', action='store_true', default=SAVE_TRAIN_DATA_METRICS, help='Save training data metrics')
	parser.add_argument('-simclr', '--load_simclr_model', action='store_true', default=LOAD_SIMCLR_MODEL, help='Load Self-supervised model to fine-tune')
	parser.add_argument('-jlstm', '--just_lstm', action='store_true', default=JUST_LSTM, help='Use only climate data')
	parser.add_argument('-pt', '--pretrained_model', type=str, default=None, help='Path to pretrained model checkpoint')
	# 静态特征（可选）
	parser.add_argument('-static', '--use_static', action='store_true', default=False, help='Enable static features branch (will use config.static_csv_path or main CSV if not specified)')
	parser.add_argument('--static_csv', type=str, default=None, help='[可选] 指定静态特征CSV路径，如果不指定则使用config.static_csv_path或主CSV文件')
	parser.add_argument('-fpt', '--freeze_pretrained', action='store_true', default=False, help='Freeze pretrained model parameters')
	
	# 增强功能参数（新增）
	parser.add_argument('-enh', '--enhanced_climate', type=str, default=str(USE_ENHANCED_CLIMATE), choices=['True', 'False', 'true', 'false'], help='使用增强气候特征提取 (True/False)')
	parser.add_argument('-cmf', '--cross_modal_fusion', type=str, default=str(USE_CROSS_MODAL_FUSION), choices=['True', 'False', 'true', 'false'], help='使用交叉模态融合 (True/False)')
	parser.add_argument('-ft', '--fusion_type', type=str, default=FUSION_TYPE, choices=['cross_modal', 'gated', 'hierarchical'], help='融合类型')
	
	# 区域自适应参数
	parser.add_argument('-ra', '--use_regional_adaptation', action='store_true', default=False, help='Use regional adaptation')
	parser.add_argument('-nr', '--num_regions', type=int, default=6, help='Number of regions (6 for China natural regions)')
	parser.add_argument('-rs', '--region_strategy', type=str, default='geographic', choices=['grid', 'geographic'], help='Region assignment strategy')
	
	# 标签策略开关参数（长尾SOC回归消融实验）
	parser.add_argument('--label_mode', type=str, default=None, 
						choices=['baseline_raw_mse', 'log1p_mse', 'log1p_huber', 'log1p_huber_w'],
						help='Label strategy mode for long-tail SOC regression ablation experiments')
	parser.add_argument('--huber_beta', type=float, default=1.0, 
						help='Beta parameter for Huber loss (SmoothL1Loss)')
	parser.add_argument('--tail_threshold', type=float, default=30.0, 
						help='Threshold for high-value samples in weighted loss')
	parser.add_argument('--tail_weight', type=float, default=2.0, 
						help='Weight for high-value samples (SOC > tail_threshold)')
	
	# S-CMRL 融合参数（基于 Semantic-Alignment Cross-Modal Residual Learning）
	parser.add_argument('--use_scmrl_fusion', action='store_true', default=False,
						help='Use S-CMRL fusion (Semantic-Alignment Cross-Modal Residual Learning) to handle weak modality noise')
	parser.add_argument('--scmrl_alpha_init', type=float, default=1.5,
						help='Initial value for alpha parameter in S-CMRL fusion (default: 1.5)')
	parser.add_argument('--scmrl_lambda_align', type=float, default=0.1,
						help='Weight for semantic alignment loss in total loss (default: 0.1)')
	parser.add_argument('--scmrl_temperature', type=float, default=0.07,
						help='Temperature parameter for semantic alignment loss (default: 0.07)')
	parser.add_argument('--scmrl_parallel', action='store_true', default=False,
						help='Use parallel fusion instead of sequential fusion for S-CMRL (default: False, sequential)')

	# FiLM 融合参数（Feature-wise Linear Modulation）
	parser.add_argument('--use_film_fusion', action='store_true', default=False,
						help='Use FiLM fusion (Feature-wise Linear Modulation) - more efficient than S-CMRL for 2D features')
	parser.add_argument('--film_alpha_init', type=float, default=1.5,
						help='Initial value for alpha parameter in FiLM fusion (default: 1.5)')
	parser.add_argument('--scmrl_legacy', action='store_true', default=False,
						help='Use the vector-input SAP-RF implementation used by the submitted 0.5404 run')
	parser.add_argument('--split_manifest', type=str, default=None,
						help='CSV manifest with Point_id and split columns (train/val/test)')
	parser.add_argument('--image_root', type=str, default=None,
						help='Shared image directory used with --split_manifest')
	parser.add_argument('--train_only_preprocessing', action='store_true', default=False,
						help='Fit climate/static preprocessing statistics using training Point_ids only')
	parser.add_argument('--revision_output_dir', type=str, default='revision_outputs/runs',
						help='Directory for per-seed revision predictions and preprocessing metadata')
	parser.add_argument('--revision_fusion', type=str, default=None,
						choices=['concat', 'sap_rf', 'gated', 'cross_attention', 'film'],
						help='Fusion strategy for controlled GRSL revision comparisons')

	args = parser.parse_args()
	return args





if __name__ == '__main__':
    
	args = parse_arguments()
	EXP_NAME = args.exp_name
	DATASET = args.dataset
	NUM_WORKERS = args.num_workers
	TRAIN_BATCH_SIZE = args.train_batch_size
	TEST_BATCH_SIZE = args.test_batch_size
	LEARNING_RATE = args.learning_rate
	WEIGHT_DECAY = args.weight_decay
	NUM_EPOCHS = args.num_epochs
	LR_SCHEDULER = args.lr_scheduler
	USE_SRTM = args.use_srtm
	USE_SPATIAL_ATTENTION = args.use_spatial_attention
	CNN_ARCHITECTURE = args.cnn_architecture
	REG_VERSION = args.reg_version
	SEEDS = args.seeds
	IMG_ENCODER_TYPE = args.img_encoder
	SPECTRAL_EMB_DIM = args.spectral_emb_dim
	
	# RNN分支控制逻辑：
	# 1. 如果显式指定了 -lstm，则开启
	# 2. 如果指定了 -rnn（不为None），则自动开启
	# 3. 否则默认关闭
	USE_LSTM_BRANCH = args.use_lstm_branch
	
	# 如果指定了 -rnn 参数，自动开启RNN分支
	if args.rnn_architecture is not None:
		USE_LSTM_BRANCH = True
		print(f"\033[93m[自动检测] 检测到 -rnn {args.rnn_architecture}，自动启用RNN分支\033[0m")
	
	# 设置RNN架构（如果未指定但开启了分支，使用默认值）
	RNN_ARCHITECTURE = args.rnn_architecture if args.rnn_architecture is not None else 'Transformer'
	LOG_LOSS = args.log_loss
	SAVE_TRAIN_DATA_METRICS = args.save_train_data_metrics
	LOAD_SIMCLR_MODEL = args.load_simclr_model
	JUST_LSTM = args.just_lstm
	PRETRAINED_MODEL = args.pretrained_model
	FREEZE_PRETRAINED = args.freeze_pretrained
	USE_STATIC_FEATURES = args.use_static  # 静态特征开关
	
	# 静态特征CSV路径：优先命令行，其次 config.static_csv_path，最后使用默认路径
	if USE_STATIC_FEATURES:
		if args.static_csv and args.static_csv not in [None, ""]:
			STATIC_CSV = args.static_csv
		elif hasattr(config, 'static_csv_path') and config.static_csv_path:
			STATIC_CSV = config.static_csv_path
		else:
			# 默认使用主CSV文件（通常静态特征和主数据在同一个CSV中）
			STATIC_CSV = lucas_csv_path
			print(f"\033[93m[静态特征] 未指定 --static_csv，使用默认路径: {STATIC_CSV}\033[0m")
		
		# 验证文件是否存在
		if not os.path.exists(STATIC_CSV):
			print(f"\033[91m错误: 静态特征CSV文件不存在: {STATIC_CSV}\033[0m")
			raise FileNotFoundError(f"Static CSV file not found: {STATIC_CSV}")
	else:
		STATIC_CSV = None
	
	# 增强功能参数（新增）
	USE_ENHANCED_CLIMATE = args.enhanced_climate.lower() == 'true'
	USE_CROSS_MODAL_FUSION = args.cross_modal_fusion.lower() == 'true'
	FUSION_TYPE = args.fusion_type


	if DATASET == 'LUCAS':
		from dataset.dataset_loader import SNDataset,SNDatasetClimate, myNormalize, myToTensor, Augmentations
		OC_MAX = 87
		# OC_MAX = 560.2
	if DATASET == 'RaCA':
		from dataset.dataset_loader_us import SNDataset,SNDatasetClimate, myNormalize, myToTensor, Augmentations
		OC_MAX = 4115
	if DATASET == 'CHINA':
		from dataset.dataset_loader_china import ChinaSNDataset as SNDataset, ChinaSNDatasetClimate as SNDatasetClimate, myNormalize, myToTensor, Augmentations
		# 尝试导入带静态特征的数据集（可选）
		try:
			from dataset.dataset_loader_china_static import ChinaSNDatasetClimateStatic
			HAS_STATIC_DS = True
		except Exception:
			HAS_STATIC_DS = False
		OC_MAX = 60.0

	SPLIT_POINT_IDS = None
	FIT_PREPROCESSING_ON_TRAIN = bool(args.train_only_preprocessing)
	REVISION_PROTOCOL = bool(args.split_manifest or args.train_only_preprocessing)
	if args.split_manifest:
		if DATASET != 'CHINA':
			raise ValueError('--split_manifest currently supports only the CHINA dataset')
		manifest_df = pd.read_csv(args.split_manifest)
		required_manifest_cols = {'Point_id', 'split'}
		if not required_manifest_cols.issubset(manifest_df.columns):
			raise ValueError(f"Split manifest must contain {sorted(required_manifest_cols)}")
		manifest_df['Point_id'] = manifest_df['Point_id'].astype(str).str.replace('.0', '', regex=False).str.strip()
		SPLIT_POINT_IDS = {
			name: set(manifest_df.loc[manifest_df['split'] == name, 'Point_id'])
			for name in ('train', 'val', 'test')
		}
		if any(not values for values in SPLIT_POINT_IDS.values()):
			raise ValueError('Split manifest must contain non-empty train, val, and test subsets')
		shared_image_root = args.image_root or 'dataset/l8_images_CN'
		if not os.path.isdir(shared_image_root):
			raise FileNotFoundError(f'Image root not found: {shared_image_root}')
		train_l8_folder_path = shared_image_root
		val_l8_folder_path = shared_image_root
		test_l8_folder_path = shared_image_root
		FIT_PREPROCESSING_ON_TRAIN = True
		print(f"[Revision] Loaded split manifest: {args.split_manifest}")
		print('[Revision] Split sizes: ' + ', '.join(f"{name}={len(ids)}" for name, ids in SPLIT_POINT_IDS.items()))
		print('[Revision] Climate/static preprocessing will be fitted on training IDs only')
  

	if JUST_LSTM:
		USE_LSTM_BRANCH = True
		USE_SPATIAL_ATTENTION = False

	if LOAD_SIMCLR_MODEL:
		if USE_LSTM_BRANCH == False:
			raise Exception("LOAD_SIMCLR_MODEL is enabled but LSTM branch is disabled. Please enable LSTM branch.")
		if JUST_LSTM:
			raise Exception("LOAD_SIMCLR_MODEL is enabled but JUST_LSTM is enabled. Please disable JUST_LSTM.")

	if LOAD_SIMCLR_MODEL:
		print("\033[91m\033[1m\033[5mWARNING!\033[0m")
		print("\033[93m Loading SimCLR Model is enabled.\
			\n This will overwrite Chosen Architectures\
			\n Also, make sure that LSTM is enabled. \033[0m")    


	# 根据 label_mode 决定是否归一化标签
	# 如果使用四组消融实验之一，则不归一化标签（保持原尺度）
	LABEL_MODE = args.label_mode
	HUBER_BETA = args.huber_beta
	TAIL_THRESHOLD = args.tail_threshold
	TAIL_WEIGHT = args.tail_weight
	
	# S-CMRL 融合参数
	USE_SCMRL_FUSION = args.use_scmrl_fusion
	SCMRL_ALPHA_INIT = args.scmrl_alpha_init
	SCMRL_LAMBDA_ALIGN = args.scmrl_lambda_align
	SCMRL_TEMPERATURE = args.scmrl_temperature
	SCMRL_PARALLEL = args.scmrl_parallel
	SCMRL_LEGACY = args.scmrl_legacy

	# FiLM 融合参数
	USE_FILM_FUSION = args.use_film_fusion
	FILM_ALPHA_INIT = args.film_alpha_init
	REVISION_FUSION = args.revision_fusion
	FUSION_BASELINE = None
	if REVISION_FUSION:
		USE_SCMRL_FUSION = REVISION_FUSION == 'sap_rf'
		USE_FILM_FUSION = REVISION_FUSION == 'film'
		FUSION_BASELINE = REVISION_FUSION if REVISION_FUSION in ('gated', 'cross_attention') else None
		if USE_SCMRL_FUSION:
			SCMRL_PARALLEL = True
			SCMRL_LEGACY = True
		print(f"[Revision] Controlled fusion mode: {REVISION_FUSION}")
	
	if LABEL_MODE in ['baseline_raw_mse', 'log1p_mse', 'log1p_huber', 'log1p_huber_w']:
		# 四组消融实验：标签保持原尺度，不归一化，不clip
		normalize_oc_flag = False
		clip_oc_flag = False
		print(f"\n{'='*80}")
		print(f"[标签策略] 模式: {LABEL_MODE}")
		print(f"[标签策略] 标签将保持原尺度 SOC (不归一化, 不clip)")
		if LABEL_MODE.startswith('log1p'):
			print(f"[标签策略] 训练目标: log1p(SOC)")
		else:
			print(f"[标签策略] 训练目标: 原尺度 SOC")
		if 'huber' in LABEL_MODE:
			print(f"[标签策略] 损失函数: Huber (beta={HUBER_BETA})")
		else:
			print(f"[标签策略] 损失函数: MSE")
		if LABEL_MODE == 'log1p_huber_w':
			print(f"[标签策略] 高值加权: SOC>{TAIL_THRESHOLD} 权重={TAIL_WEIGHT}")
		print(f"{'='*80}\n")
	else:
		# 默认行为：归一化标签到 [0,1]
		normalize_oc_flag = True
		clip_oc_flag = True
		print(f"\n[标签策略] 使用默认归一化模式 (标签归一化到 [0,1])\n")
	
	if USE_SRTM:
		mynorm = myNormalize(img_bands_min_max =[[(0,7),(0,1)], [(7,12),(-1,1)], [(12), (-4,2963)], [(13), (0, 90)]], 
							 oc_min = 0, oc_max = OC_MAX, normalize_oc=normalize_oc_flag, clip_oc=clip_oc_flag)
	else:
		mynorm = myNormalize(img_bands_min_max =[[(0,7),(0,1)], [(7,12),(-1,1)]], 
							 oc_min = 0, oc_max = OC_MAX, normalize_oc=normalize_oc_flag, clip_oc=clip_oc_flag)
		
	my_to_tensor = myToTensor()
	my_augmentation = Augmentations()
	train_transform = transforms.Compose([mynorm, my_to_tensor,my_augmentation])
	test_transform = transforms.Compose([mynorm, my_to_tensor])





	bands = [0,1,2,3,4,5,6,7,8,9,10,11] if not USE_SRTM else [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

	def split_ids(name):
		return None if SPLIT_POINT_IDS is None else SPLIT_POINT_IDS[name]

	################################# IF Not USE_LSTM_BRANCH ###############################
	if not USE_LSTM_BRANCH: # NOT USING THE CLIMATE DATA
		point_id_kwargs = ({'point_ids': split_ids('train')} if DATASET == 'CHINA' else {})
		train_ds = SNDataset(train_l8_folder_path, lucas_csv_path,l8_bands=bands, transform=train_transform, **point_id_kwargs)
		point_id_kwargs = ({'point_ids': split_ids('test')} if DATASET == 'CHINA' else {})
		test_ds = SNDataset(test_l8_folder_path, lucas_csv_path,l8_bands=bands, transform=test_transform, **point_id_kwargs)
		point_id_kwargs = ({'point_ids': split_ids('val')} if DATASET == 'CHINA' else {})
		val_ds = SNDataset(val_l8_folder_path, lucas_csv_path,l8_bands=bands, transform=test_transform, **point_id_kwargs)
		point_id_kwargs = ({'point_ids': split_ids('test')} if DATASET == 'CHINA' else {})
		test_ds_w_id = SNDataset(test_l8_folder_path, lucas_csv_path,l8_bands=bands, transform=test_transform, return_point_id=True, **point_id_kwargs)
		
	################################### IF USE_LSTM_BRANCH #################################
	else: # USING THE CLIMATE DATA
		# 如果启用了静态特征开关且提供了静态特征CSV且模块可用，则使用带静态特征的数据集
		if USE_STATIC_FEATURES and ('HAS_STATIC_DS' in locals()) and HAS_STATIC_DS and (STATIC_CSV is not None) and (STATIC_CSV != '') and os.path.exists(STATIC_CSV):
			train_ds = ChinaSNDatasetClimateStatic(train_l8_folder_path,
											lucas_csv_path,
											climate_csv_folder_path,
											static_csv_path=STATIC_CSV,
											l8_bands=bands, transform=train_transform,
											point_ids=split_ids('train'),
											fit_climate_stats=FIT_PREPROCESSING_ON_TRAIN,
											fit_static_stats=FIT_PREPROCESSING_ON_TRAIN)
			climate_stats = train_ds.get_climate_stats() if FIT_PREPROCESSING_ON_TRAIN else None
			static_stats = train_ds.get_static_stats() if FIT_PREPROCESSING_ON_TRAIN else None

			test_ds = ChinaSNDatasetClimateStatic(test_l8_folder_path,
										lucas_csv_path,
										climate_csv_folder_path,
										static_csv_path=STATIC_CSV,
										l8_bands=bands, transform=test_transform,
										point_ids=split_ids('test'),
										climate_stats=climate_stats,
										static_stats=static_stats)
			
			val_ds = ChinaSNDatasetClimateStatic(val_l8_folder_path,
										lucas_csv_path,
										climate_csv_folder_path,
										static_csv_path=STATIC_CSV,
										l8_bands=bands, transform=test_transform,
										point_ids=split_ids('val'),
										climate_stats=climate_stats,
										static_stats=static_stats)
			
			test_ds_w_id = ChinaSNDatasetClimateStatic(test_l8_folder_path,
										lucas_csv_path,
										climate_csv_folder_path,
										static_csv_path=STATIC_CSV,
										l8_bands=bands, transform=test_transform, return_point_id=True,
										point_ids=split_ids('test'),
										climate_stats=climate_stats,
										static_stats=static_stats)
			USING_STATIC_FEATURES = True
		else:
			train_ds = SNDatasetClimate(
				train_l8_folder_path,
				lucas_csv_path,
				climate_csv_folder_path,
				l8_bands=bands,
				transform=train_transform,
				**({'point_ids': split_ids('train'), 'fit_climate_stats': FIT_PREPROCESSING_ON_TRAIN} if DATASET == 'CHINA' else {})
			)
			climate_stats = train_ds.get_climate_stats() if DATASET == 'CHINA' and FIT_PREPROCESSING_ON_TRAIN else None

			test_ds = SNDatasetClimate(
				test_l8_folder_path,
				lucas_csv_path,
				climate_csv_folder_path,
				l8_bands=bands,
				transform=test_transform,
				**({'point_ids': split_ids('test'), 'climate_stats': climate_stats} if DATASET == 'CHINA' else {})
			)
			
			val_ds = SNDatasetClimate(
				val_l8_folder_path,
				lucas_csv_path,
				climate_csv_folder_path,
				l8_bands=bands,
				transform=test_transform,
				**({'point_ids': split_ids('val'), 'climate_stats': climate_stats} if DATASET == 'CHINA' else {})
			)
			
			test_ds_w_id = SNDatasetClimate(
				test_l8_folder_path,
				lucas_csv_path,
				climate_csv_folder_path,
				l8_bands=bands,
				transform=test_transform,
				return_point_id=True,
				**({'point_ids': split_ids('test'), 'climate_stats': climate_stats} if DATASET == 'CHINA' else {})
			)
			USING_STATIC_FEATURES = False

	SEQ_LEN = test_ds_w_id[0][0][1].shape[0]

	# 生成区域ID（如果启用区域自适应）
	if args.use_regional_adaptation:
		from soilnet.submodules.region_embedding import create_region_assigner
		import pandas as pd
		
		# 读取CSV文件
		df = pd.read_csv(lucas_csv_path)
		
		# 创建区域分配器
		if args.region_strategy == 'grid':
			n_lat = int(np.sqrt(args.num_regions))
			n_lon = args.num_regions // n_lat
			assigner = create_region_assigner('grid', n_lat=n_lat, n_lon=n_lon)
		else:
			assigner = create_region_assigner(args.region_strategy)
		
		# 为所有样本分配区域ID
		all_region_ids = assigner.assign_regions(df['Latitude'].values, df['Longitude'].values)
		df['region_id'] = all_region_ids
		
		# 确定Point_ID列
		id_col = None
		for col in df.columns:
			if col.lower() in ['point_id', 'pointid', 'point_id', 'pointid', 'pid', 'pointid']:
				id_col = col
				break
		if id_col is None:
			raise ValueError("无法在CSV中找到Point_ID列以构建区域映射。")
		
		df['pid_key'] = df[id_col].astype(str).str.replace('.0', '', regex=False).str.strip()
		
		def majority_vote(values: np.ndarray) -> int:
			values = np.asarray(values, dtype=np.int64)
			if values.size == 0:
				return 0
			counts = np.bincount(values)
			return int(np.argmax(counts))
		
		region_map_series = df.groupby('pid_key')['region_id'].agg(majority_vote)
		region_map = region_map_series.to_dict()
		
		def extract_pid_from_name(name: str) -> str:
			return str(name).split('_')[0].replace('.0', '').strip()
		
		def build_split_region_ids(dataset, split_name: str) -> np.ndarray:
			mapped = []
			missing = 0
			for fname in getattr(dataset, 'l8_names', []):
				pid = extract_pid_from_name(fname)
				if pid in region_map:
					mapped.append(region_map[pid])
				else:
					missing += 1
					mapped.append(0)
			if missing > 0:
				print(f"[WARN] 区域映射缺失 {missing} 个样本（{split_name} 分集），已回退到区域0。")
			return np.array(mapped, dtype=np.int64)
		
		train_region_ids = build_split_region_ids(train_ds, 'train')
		val_region_ids = build_split_region_ids(val_ds, 'val')
		test_region_ids = build_split_region_ids(test_ds, 'test')
		
		def print_region_distribution(name: str, region_ids: np.ndarray):
			if region_ids.size == 0:
				print(f"区域分布 - {name}: 空集")
			else:
				counts = np.bincount(region_ids)
				print(f"区域分布 - {name}: {counts}")
		
		print_region_distribution("训练集", train_region_ids)
		print_region_distribution("验证集", val_region_ids)
		print_region_distribution("测试集", test_region_ids)
	else:
		train_region_ids = val_region_ids = test_region_ids = None

	# COUNTING the csv files in the csv folder
	CSV_FILES = [f for f in os.listdir(climate_csv_folder_path) if f.endswith('.csv')]
	NUM_CLIMATE_FEATURES = len(CSV_FILES)

	cv_results = {"train_loss": [],
				"val_loss": [],
				"MAE": [],
				"RMSE": [],
				"R2": [],
				"best_epoch": [],
				"best_val_loss": [],
				"alpha_values": [],
				"train_MAE": [],
					"train_RMSE": [],
					"train_R2": []
		}



	now = datetime.now()
	run_name = now.strftime("D_%Y_%m_%d_T_%H_%M")
	print("Current Date and Time:", run_name)
	# create a folder called 'results' in the current directory if it doesn't exist
	if not os.path.exists('results'):
		os.mkdir('results')
	revision_run_dir = None
	if REVISION_PROTOCOL:
		revision_run_dir = Path(args.revision_output_dir) / f"{EXP_NAME}_{run_name}"
		revision_run_dir.mkdir(parents=True, exist_ok=True)
		preprocessing_metadata = {
			'split_manifest': str(Path(args.split_manifest).resolve()) if args.split_manifest else None,
			'image_root': str(Path(train_l8_folder_path).resolve()),
			'train_only_preprocessing': FIT_PREPROCESSING_ON_TRAIN,
			'climate_stats': getattr(train_ds, 'get_climate_stats', lambda: None)(),
			'static_stats': getattr(train_ds, 'get_static_stats', lambda: None)(),
			'split_sizes': {name: len(ids) for name, ids in SPLIT_POINT_IDS.items()} if SPLIT_POINT_IDS else None,
		}
		with open(revision_run_dir / 'preprocessing.json', 'w', encoding='utf-8') as fp:
			json.dump(preprocessing_metadata, fp, indent=2, ensure_ascii=False)
		
		
		
	# 初始化最佳/最差指标
	# 注意：使用标签策略开关时，指标在原尺度上计算，不再是 [0,1] 范围
	best_mae = float('inf')
	worst_mae = 0.0

	best_rmse = float('inf')
	worst_rmse = 0.0

	best_seed = SEEDS[0]
	worst_seed = SEEDS[0]

	def train_with_regional_adaptation(model, train_dl, test_dl, val_dl, optimizer, loss_fn, 
									  epochs, lr_scheduler, save_model_path, 
									  train_region_ids, val_region_ids, test_region_ids):
		"""
		支持区域自适应的训练函数
		"""
		from tqdm import tqdm
		import torch
		from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
		from dataset.utils.utils import TextColors as tc
		from train_utils import evaluate_regression_metrics
		import pandas as pd
		import numpy as np
		
		# 学习率调度器
		if lr_scheduler == "step":
			scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
		elif lr_scheduler == "plateau":
			scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
		else:
			scheduler = None
		
		train_region_ids = np.asarray(train_region_ids, dtype=np.int64)
		val_region_ids = np.asarray(val_region_ids, dtype=np.int64)
		test_region_ids = np.asarray(test_region_ids, dtype=np.int64)
		
		train_losses = []
		val_losses = []
		best_val_loss = float('inf')
		
		for epoch in range(1, epochs + 1):
			print(tc.OKGREEN, f"Epoch {epoch}\n-------------------------------", tc.ENDC)
			
			# 训练阶段
			model.train()
			train_loss = 0.0
			train_loop = tqdm(train_dl, leave=True)
			
			train_ptr = 0
			for batch_idx, (X, y) in enumerate(train_loop):
				# 获取对应的区域ID
				current_batch_size = y.shape[0] if hasattr(y, 'shape') else len(y)
				end_ptr = min(train_ptr + current_batch_size, len(train_region_ids))
				batch_region_slice = train_region_ids[train_ptr:end_ptr]
				train_ptr = end_ptr
				batch_region_ids = torch.as_tensor(batch_region_slice, dtype=torch.long, device=device)
				
				# 数据移动到设备
				if isinstance(X, tuple) or isinstance(X, list):
					X = [tensor.to(device) for tensor in list(X)]
					y = y.to(device)
				else:
					X, y = X.to(device), y.to(device)
				
				# 前向传播
				if hasattr(model, 'use_regional_adaptation') and model.use_regional_adaptation:
					y_pred = model(X, batch_region_ids)
				else:
					y_pred = model(X)
				
				# 计算损失
				loss = loss_fn(y_pred, y.unsqueeze(1))
				train_loss += loss.item()
				
				# 反向传播
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				# 更新进度条
				if batch_idx % 10 == 0 or batch_idx == len(train_dl) - 1:
					train_loop.set_postfix(loss=loss.item())
			
			# 验证阶段
			model.eval()
			val_loss = 0.0
			val_predictions = []
			val_targets = []
			
			with torch.no_grad():
				val_ptr = 0
				for batch_idx, (X, y) in enumerate(val_dl):
					# 获取对应的区域ID
					current_batch_size = y.shape[0] if hasattr(y, 'shape') else len(y)
					end_ptr = min(val_ptr + current_batch_size, len(val_region_ids))
					batch_region_slice = val_region_ids[val_ptr:end_ptr]
					val_ptr = end_ptr
					batch_region_ids = torch.as_tensor(batch_region_slice, dtype=torch.long, device=device)
					
					# 数据移动到设备
					if isinstance(X, tuple) or isinstance(X, list):
						X = [tensor.to(device) for tensor in list(X)]
						y = y.to(device)
					else:
						X, y = X.to(device), y.to(device)
					
					# 前向传播
					if hasattr(model, 'use_regional_adaptation') and model.use_regional_adaptation:
						y_pred = model(X, batch_region_ids)
					else:
						y_pred = model(X)
					
					# 计算损失
					loss = loss_fn(y_pred, y.unsqueeze(1))
					val_loss += loss.item()
					
					# 收集预测结果
					val_predictions.extend(y_pred.cpu().numpy())
					val_targets.extend(y.cpu().numpy())
			
			# 计算平均损失
			train_loss /= len(train_dl)
			val_loss /= len(val_dl)
			
			train_losses.append(train_loss)
			val_losses.append(val_loss)
			
			# 学习率调度
			if scheduler:
				if lr_scheduler == "plateau":
					scheduler.step(val_loss)
				else:
					scheduler.step()
			
			# 保存最佳模型
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				torch.save({
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'epoch': epoch,
					'val_loss': val_loss
				}, save_model_path)
			
			# 计算验证指标
			val_predictions = np.array(val_predictions).flatten()
			val_targets = np.array(val_targets).flatten()
			rmse, r2, rpiq, mae, mec, ccc = evaluate_regression_metrics(val_targets, val_predictions)
			
			print(tc.OKCYAN, f"Epoch {epoch} Results: | ", tc.ENDC)
			print(tc.OKCYAN, f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", tc.ENDC)
			print(tc.OKCYAN, f"Val RMSE: {rmse:.4f} | Val R²: {r2:.4f} | Val MAE: {mae:.4f}", tc.ENDC)
		
		# 测试阶段
		model.eval()
		test_predictions = []
		test_targets = []
		
		with torch.no_grad():
			test_ptr = 0
			for batch_idx, (X, y) in enumerate(test_dl):
				# 获取对应的区域ID
				current_batch_size = y.shape[0] if hasattr(y, 'shape') else len(y)
				end_ptr = min(test_ptr + current_batch_size, len(test_region_ids))
				batch_region_slice = test_region_ids[test_ptr:end_ptr]
				test_ptr = end_ptr
				batch_region_ids = torch.as_tensor(batch_region_slice, dtype=torch.long, device=device)
				
				# 数据移动到设备
				if isinstance(X, tuple) or isinstance(X, list):
					X = [tensor.to(device) for tensor in list(X)]
					y = y.to(device)
				else:
					X, y = X.to(device), y.to(device)
				
				# 前向传播
				if hasattr(model, 'use_regional_adaptation') and model.use_regional_adaptation:
					y_pred = model(X, batch_region_ids)
				else:
					y_pred = model(X)
				
				# 收集预测结果
				test_predictions.extend(y_pred.cpu().numpy())
				test_targets.extend(y.cpu().numpy())
		
		# 计算测试指标
		test_predictions = np.array(test_predictions).flatten()
		test_targets = np.array(test_targets).flatten()
		rmse, r2, rpiq, mae, mec, ccc = evaluate_regression_metrics(test_targets, test_predictions)
		
		print(tc.OKGREEN, f"Final Test Results:", tc.ENDC)
		print(tc.OKGREEN, f"Test RMSE: {rmse:.4f} | Test R²: {r2:.4f} | Test MAE: {mae:.4f}", tc.ENDC)
		
		return {
			'train_loss': train_losses,
			'val_loss': val_losses,
			'MAE': [mae],
			'RMSE': [rmse],
			'R2': [r2],
			'train_MAE': [0.0],  # 占位符
			'train_RMSE': [0.0],  # 占位符
			'train_R2': [0.0]     # 占位符
		}

	for idx, seed in enumerate(SEEDS):
		print(tc.BOLD_BAKGROUNDs.PURPLE, f"CROSS VAL {idx+1}", tc.ENDC)
		
		
		train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
		test_dl = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
		val_dl = DataLoader(val_ds, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
		
		if args.use_regional_adaptation:
			print("[INFO] 区域自适应已启用：训练数据加载器将设置为 shuffle=False 以对齐区域ID")
			train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
		
		#model = SoilNetFC(cnn_in_channels=12, regresor_input_from_cnn=1024, hidden_size=128).to(device)
		# 根据CNN架构确定输出维度
		if CNN_ARCHITECTURE in ['ViT-CoMer', 'MPViT', 'HybridViT']:
			cnn_output_dim = 384
		elif CNN_ARCHITECTURE == 'ViT':
			cnn_output_dim = 768
		else:
			cnn_output_dim = 1024
		
		# 使用SoilNetLSTM模型
		if USE_ENHANCED_CLIMATE or USE_CROSS_MODAL_FUSION:
			print("⚠️  警告：SoilNetLSTM 不支持增强气候特征提取和交叉模态融合功能")
			print("   将使用标准的 SoilNetLSTM 模型")
		if True:  # 总是使用SoilNetLSTM
			if not JUST_LSTM:
				if USE_LSTM_BRANCH:
					if ('USING_STATIC_FEATURES' in locals()) and USING_STATIC_FEATURES:
						# 使用支持静态特征的模型
						try:
							from soilnet.soil_net_static import SoilNetLSTMWithStatic
							# 从训练数据集获取静态数值维度与LULC类别数
							static_numeric_dim = getattr(train_ds, 'get_static_numeric_dim', getattr(train_ds, 'get_static_feature_dim', lambda: 0))()
							lulc_classes = getattr(train_ds, 'get_lulc_num_classes', lambda: 0)()
							# 注意：static_dim 参数会在 SoilNetLSTMWithStatic 中根据 SCMRL fusion 自动设置
							# 如果使用SCMRL fusion，StaticBranch会将静态特征编码为lstm_dim维度
							model = SoilNetLSTMWithStatic(
								static_feature_dim=static_numeric_dim,
								lulc_num_classes=lulc_classes,
								lulc_embed_dim=16,  # LULC Embedding维度
								use_glam=USE_SPATIAL_ATTENTION,
								cnn_arch=CNN_ARCHITECTURE,
								reg_version=REG_VERSION,
								cnn_in_channels=len(bands),
								regresor_input_from_cnn=cnn_output_dim,
								lstm_n_features=NUM_CLIMATE_FEATURES,
								lstm_n_layers=2,
								lstm_out=128,
								hidden_size=128,
								rnn_arch=RNN_ARCHITECTURE,
								seq_len=SEQ_LEN,
								use_spectral_enhance=False,
								spectral_type='hybrid',
								use_regional_adaptation=args.use_regional_adaptation,
								num_regions=args.num_regions,
								img_encoder_type=IMG_ENCODER_TYPE,
								spectral_cnn_emb_dim=SPECTRAL_EMB_DIM,
								use_scmrl_fusion=USE_SCMRL_FUSION,
								scmrl_alpha_init=SCMRL_ALPHA_INIT,
								scmrl_temperature=SCMRL_TEMPERATURE,
								scmrl_parallel=SCMRL_PARALLEL,
								scmrl_legacy=SCMRL_LEGACY,
								use_film_fusion=USE_FILM_FUSION,
								fusion_baseline=FUSION_BASELINE,
								# static_dim 会在模型内部根据SCMRL fusion自动设置，不需要手动传递
							).to(device)
							print(f"Using static features (numeric_dim={static_numeric_dim}, lulc_classes={lulc_classes}) in model.")
						except Exception as e:
							if REVISION_PROTOCOL:
								raise RuntimeError(
									"Revision runs require the static-feature model; refusing to "
									"silently fall back to a different dataset/model protocol."
								) from e
							print(f"Warning: Failed to import static model, fallback to standard SoilNetLSTM. Error: {e}")
							print("Warning: Recreating datasets without static features to match model...")
							# 重新创建标准数据集（不包含静态特征）
							try:
								from dataset.dataset_loader_china import ChinaSNDatasetClimate as SNDatasetClimate
							except ImportError:
								from dataset.dataset_loader_china import SNDatasetClimate
							train_ds = SNDatasetClimate(
								train_l8_folder_path,
								lucas_csv_path,
								climate_csv_folder_path,
								l8_bands=bands,
								transform=train_transform
							)
							test_ds = SNDatasetClimate(
								test_l8_folder_path,
								lucas_csv_path,
								climate_csv_folder_path,
								l8_bands=bands,
								transform=test_transform
							)
							val_ds = SNDatasetClimate(
								val_l8_folder_path,
								lucas_csv_path,
								climate_csv_folder_path,
								l8_bands=bands,
								transform=test_transform
							)
							test_ds_w_id = SNDatasetClimate(
								test_l8_folder_path,
								lucas_csv_path,
								climate_csv_folder_path,
								l8_bands=bands,
								transform=test_transform,
								return_point_id=True
							)
							USING_STATIC_FEATURES = False
							# 重新创建数据加载器
							train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
							if args.use_regional_adaptation:
								print("[INFO] 区域自适应已启用：回退数据集训练加载器设置为 shuffle=False")
								train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
							test_dl = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
							val_dl = DataLoader(val_ds, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
							from soilnet.soil_net import SoilNetLSTM
							model = SoilNetLSTM(
								use_glam=USE_SPATIAL_ATTENTION,
								cnn_arch=CNN_ARCHITECTURE,
								reg_version=REG_VERSION,
								cnn_in_channels=len(bands),
								regresor_input_from_cnn=cnn_output_dim,
								lstm_n_features=NUM_CLIMATE_FEATURES,
								lstm_n_layers=2,
								lstm_out=128,
								hidden_size=128,
								rnn_arch=RNN_ARCHITECTURE,
								seq_len=SEQ_LEN,
								use_spectral_enhance=False,
								spectral_type='hybrid',
								use_regional_adaptation=args.use_regional_adaptation,
								num_regions=args.num_regions,
								img_encoder_type=IMG_ENCODER_TYPE,
								spectral_cnn_emb_dim=SPECTRAL_EMB_DIM,
								use_film_fusion=USE_FILM_FUSION
							).to(device)
					else:
						from soilnet.soil_net import SoilNetLSTM
						# 获取静态特征维度（如果使用）
						static_dim = None
						if USE_STATIC_FEATURES and ('USING_STATIC_FEATURES' in locals()) and USING_STATIC_FEATURES:
							try:
								static_dim = getattr(train_ds, 'get_static_numeric_dim', getattr(train_ds, 'get_static_feature_dim', lambda: 0))()
							except:
								static_dim = None
						
						model = SoilNetLSTM(
							use_glam=USE_SPATIAL_ATTENTION,
							cnn_arch=CNN_ARCHITECTURE,
							reg_version=REG_VERSION,
							cnn_in_channels=len(bands),
							regresor_input_from_cnn=cnn_output_dim,
							lstm_n_features=NUM_CLIMATE_FEATURES,
							lstm_n_layers=2,
							lstm_out=128,
							hidden_size=128,
							rnn_arch=RNN_ARCHITECTURE,
							seq_len=SEQ_LEN,
							use_spectral_enhance=False,
							spectral_type='hybrid',
							use_regional_adaptation=args.use_regional_adaptation,
							num_regions=args.num_regions,
							img_encoder_type=IMG_ENCODER_TYPE,
							spectral_cnn_emb_dim=SPECTRAL_EMB_DIM,
							use_scmrl_fusion=USE_SCMRL_FUSION,
							scmrl_alpha_init=SCMRL_ALPHA_INIT,
							scmrl_temperature=SCMRL_TEMPERATURE,
							scmrl_parallel=SCMRL_PARALLEL,
							scmrl_legacy=SCMRL_LEGACY,
							use_film_fusion=USE_FILM_FUSION,
							fusion_baseline=FUSION_BASELINE,
							static_dim=static_dim
						).to(device)
				else:
					# 如果不使用气候数据，仍然使用原始模型
					model = SoilNet(use_glam=USE_SPATIAL_ATTENTION, cnn_arch=CNN_ARCHITECTURE, reg_version=REG_VERSION,
							cnn_in_channels=len(bands), regresor_input_from_cnn=cnn_output_dim, hidden_size=128,
							img_encoder_type=IMG_ENCODER_TYPE, spectral_cnn_emb_dim=SPECTRAL_EMB_DIM).to(device)
			else:
				# 如果只使用LSTM，仍然使用原始模型
				model = SoilNetJustLSTM(use_glam=USE_SPATIAL_ATTENTION, cnn_arch=CNN_ARCHITECTURE, reg_version=REG_VERSION,
								cnn_in_channels=len(bands), regresor_input_from_cnn=cnn_output_dim,
								lstm_n_features=NUM_CLIMATE_FEATURES, lstm_n_layers=2, lstm_out=128,
								hidden_size=128, rnn_arch=RNN_ARCHITECTURE, seq_len=SEQ_LEN).to(device)
		
		if LOAD_SIMCLR_MODEL:
			model = torch.load(SIMCLR_PATH).to(device)
			model = SoilNetSimCLRwRegHead(model, hidden_size=128, reg_version=REG_VERSION).to(device)
		
		# Load pretrained model if specified
		if PRETRAINED_MODEL is not None:
			print(f"\033[93mLoading pretrained model from: {PRETRAINED_MODEL}\033[0m")
			checkpoint = torch.load(PRETRAINED_MODEL, map_location=device)
			
			# Handle different checkpoint formats
			if 'model_state_dict' in checkpoint:
				model.load_state_dict(checkpoint['model_state_dict'], strict=False)
			elif 'state_dict' in checkpoint:
				model.load_state_dict(checkpoint['state_dict'], strict=False)
			else:
				model.load_state_dict(checkpoint, strict=False)
			
			print(f"\033[92mSuccessfully loaded pretrained model\033[0m")
			
			# Freeze pretrained parameters if requested
			if FREEZE_PRETRAINED:
				print(f"\033[93mFreezing pretrained model parameters\033[0m")
				for name, param in model.named_parameters():
					if 'cnn' in name:  # Freeze CNN backbone
						param.requires_grad = False
			
		
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		
		# Saving the model on the last epoch - 使用seed特定的路径，避免覆盖best模型
		save_model_path = f"results/RUN_{EXP_NAME}_{run_name}_seed{seed}.pth.tar"

		
		loss_instance = RMSLELoss() if LOG_LOSS else RMSELoss()
		
		# 如果启用区域自适应，为区域嵌入设置独立的学习率（更小）
		if args.use_regional_adaptation:
			region_params = []
			other_params = []
			for name, param in model.named_parameters():
				if 'region_embedding' in name:
					region_params.append(param)
				else:
					other_params.append(param)
			
			if len(region_params) > 0:
				optimizer = torch.optim.AdamW([
					{'params': other_params, 'lr': LEARNING_RATE},
					{'params': region_params, 'lr': LEARNING_RATE * 0.1}  # 区域嵌入用更小的学习率
				], weight_decay=WEIGHT_DECAY)
				print(f"[INFO] 区域自适应已启用：区域嵌入学习率 = {LEARNING_RATE * 0.1:.6f} (主学习率的10%), WD={WEIGHT_DECAY}")
			else:
				optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
				print(f"[WARN] 未找到区域嵌入参数，使用统一学习率, WD={WEIGHT_DECAY}")
		else:
			optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
		# 如果启用区域自适应，使用自定义训练循环
		if args.use_regional_adaptation:
			if train_region_ids is None or val_region_ids is None or test_region_ids is None:
				raise ValueError("Region adaptation requires region IDs for train/val/test splits.")
			results = train_with_regional_adaptation(
							model, train_dl, test_dl, val_dl,
							optimizer,
							loss_instance, epochs=NUM_EPOCHS, lr_scheduler=LR_SCHEDULER,
							save_model_path=save_model_path,
							train_region_ids=train_region_ids,
							val_region_ids=val_region_ids,
							test_region_ids=test_region_ids
							)
		else:
			# 准备对齐损失函数（如果使用 S-CMRL）
			alignment_loss_fn = None
			if USE_SCMRL_FUSION and hasattr(model, 'alignment_loss_fn') and model.alignment_loss_fn is not None:
				alignment_loss_fn = model.alignment_loss_fn
				print(f"[Info] S-CMRL 对齐损失已启用: lambda={SCMRL_LAMBDA_ALIGN}, temperature={SCMRL_TEMPERATURE}")
			elif USE_FILM_FUSION:
				print(f"[Info] FiLM 融合已启用: alpha_init={FILM_ALPHA_INIT} (无对齐损失)")

			results = train(model, train_dl, test_dl, val_dl,
							optimizer,
							loss_instance, epochs=NUM_EPOCHS, lr_scheduler=LR_SCHEDULER,
							save_model_path= save_model_path,
							save_model_if_mae_lower_than= best_rmse,
							save_train_data_metrics=SAVE_TRAIN_DATA_METRICS,
							label_mode=LABEL_MODE,
							huber_beta=HUBER_BETA,
							tail_threshold=TAIL_THRESHOLD,
							tail_weight=TAIL_WEIGHT,
							alignment_loss_fn=alignment_loss_fn,
							lambda_align=SCMRL_LAMBDA_ALIGN,
							select_best_on_val=REVISION_PROTOCOL
							)

		
		cv_results['train_loss'].append(results['train_loss'])
		cv_results['val_loss'].append(results['val_loss'])
		cv_results['MAE'].append(results['MAE'][0])
		cv_results['RMSE'].append(results['RMSE'][0])
		cv_results['R2'].append(results['R2'][0])
		cv_results['best_epoch'].append(results.get('best_epoch'))
		cv_results['best_val_loss'].append(results.get('best_val_loss'))
		alpha_values = None
		if hasattr(model, 'fusion') and hasattr(model.fusion, 'get_alpha_values'):
			alpha_values = model.fusion.get_alpha_values()
		cv_results['alpha_values'].append(alpha_values)

		if REVISION_PROTOCOL:
			seed_prediction_path = revision_run_dir / f'seed_{seed}_predictions.csv'
			seed_test_loader = DataLoader(
				test_ds_w_id,
				batch_size=TEST_BATCH_SIZE,
				shuffle=False,
				num_workers=NUM_WORKERS,
			)
			test_step_w_id(
				model=model,
				data_loader=seed_test_loader,
				loss_fn=nn.L1Loss(),
				verbose=False,
				csv_file=str(seed_prediction_path),
				region_ids=test_region_ids if args.use_regional_adaptation else None,
				label_mode=LABEL_MODE,
			)
			with open(revision_run_dir / f'seed_{seed}_metadata.json', 'w', encoding='utf-8') as fp:
				json.dump({
					'seed': seed,
					'best_epoch': results.get('best_epoch'),
					'best_val_loss': results.get('best_val_loss'),
					'MAE': float(results['MAE'][0]),
					'RMSE': float(results['RMSE'][0]),
					'R2': float(results['R2'][0]),
					'alpha_values': alpha_values,
					'predictions': str(seed_prediction_path.resolve()),
				}, fp, indent=2, ensure_ascii=False)
		
		if SAVE_TRAIN_DATA_METRICS:
			cv_results['train_MAE'].append(results['train_MAE'])
			cv_results['train_RMSE'].append(results['train_RMSE'])
			cv_results['train_R2'].append(results['train_R2'])
		
		# Stop the training loop via RMSE 
		if not REVISION_PROTOCOL and results['RMSE'][0] < best_rmse:
			best_rmse = results['RMSE'][0]
			best_seed = seed
			print(tc.BOLD_BAKGROUNDs.GREEN, f"Best RMSE improved to {best_rmse}", tc.ENDC)
			# Save the best model
			best_model_path = f"results/RUN_{EXP_NAME}_{run_name}_best.pth.tar"
			save_checkpoint(model, optimizer=optimizer, filename=best_model_path)
			
		if not REVISION_PROTOCOL and results['RMSE'][0] > worst_rmse:
			worst_rmse = results['RMSE'][0]
			worst_seed = seed
			print(tc.BOLD_BAKGROUNDs.RED, f"Worst RMSE worsened to {worst_rmse}", tc.ENDC)
			# Save the worst model
			worst_model_path = f"results/RUN_{EXP_NAME}_{run_name}_worst.pth.tar"
			save_checkpoint(model, optimizer=optimizer, filename=worst_model_path)
			
		print(f"This Runs RMSE: {results['RMSE'][0]}")
		
		
		
	train_arr = np.asarray(cv_results['train_loss'])
	val_arr = np.asarray(cv_results['val_loss'])


	y_label = "RMSLE" if LOG_LOSS else "RMSE"
	plot_train_test_losses(train_arr,val_arr, title="Train/Validation Losses", x_label="Epochs", y_label=y_label,
						min_max_bounds= True, tight_x_lim= True,
						train_legend = "Train", test_legend = "Validation",
						save_path=f"results/RUN_{EXP_NAME}_{run_name}.png")

	# Format the date and time
	now = datetime.now()
	finish_string = now.strftime("%Y-%m-%d %H:%M:%S")
	print("Current Date and Time:", finish_string)


	cv_results_full = {}
	cv_results_full['MAE_MEAN'] = np.mean(cv_results['MAE'])
	cv_results_full['RMSE_MEAN'] = np.mean(cv_results['RMSE'])
	cv_results_full['R2_MEAN'] = np.mean(cv_results['R2'])
	cv_results_full['LOAD_SIMCLR_MODEL'] = LOAD_SIMCLR_MODEL
	cv_results_full['JUST_LSTM'] = JUST_LSTM
	cv_results_full['USE_LSTM_BRANCH'] = USE_LSTM_BRANCH
	# cv_results_full['USE_PRIM_CLIM'] = USE_PRIM_CLIM
	# cv_results_full['USE_SEC_CLIM'] = USE_SEC_CLIM
	cv_results_full['LOG_LOSS'] = LOG_LOSS
	cv_results_full['NUM_CLIMATE_FEATURES'] = NUM_CLIMATE_FEATURES if USE_LSTM_BRANCH else None
	cv_results_full['CSV_FILES'] = CSV_FILES if USE_LSTM_BRANCH else None
	cv_results_full['NUM_WORKERS'] = NUM_WORKERS
	cv_results_full['TRAIN_BATCH_SIZE'] = TRAIN_BATCH_SIZE
	cv_results_full['TEST_BATCH_SIZE'] = TEST_BATCH_SIZE
	cv_results_full['LEARNING_RATE'] = LEARNING_RATE
	cv_results_full['NUM_EPOCHS'] = NUM_EPOCHS
	cv_results_full['LR_SCHEDULER'] = LR_SCHEDULER
	cv_results_full['CNN_ARCHITECTURE'] = CNN_ARCHITECTURE
	cv_results_full['IMG_ENCODER_TYPE'] = IMG_ENCODER_TYPE
	cv_results_full['SPECTRAL_EMB_DIM'] = SPECTRAL_EMB_DIM if IMG_ENCODER_TYPE == 'spectral_cnn' else None
	cv_results_full['USE_SCMRL_FUSION'] = USE_SCMRL_FUSION
	cv_results_full['SCMRL_ALPHA_INIT'] = SCMRL_ALPHA_INIT if USE_SCMRL_FUSION else None
	cv_results_full['SCMRL_LAMBDA_ALIGN'] = SCMRL_LAMBDA_ALIGN if USE_SCMRL_FUSION else None
	cv_results_full['SCMRL_TEMPERATURE'] = SCMRL_TEMPERATURE if USE_SCMRL_FUSION else None
	cv_results_full['SCMRL_PARALLEL'] = SCMRL_PARALLEL if USE_SCMRL_FUSION else None
	cv_results_full['SCMRL_LEGACY'] = SCMRL_LEGACY if USE_SCMRL_FUSION else None
	cv_results_full['REG_VERSION'] = REG_VERSION
	cv_results_full['USE_SPATIAL_ATTENTION'] = USE_SPATIAL_ATTENTION
	cv_results_full['Best Seed'] = None if REVISION_PROTOCOL else best_seed
	cv_results_full['SEEDS'] = SEEDS
	cv_results_full['OC_MAX'] = OC_MAX
	cv_results_full['USE_SRTM'] = USE_SRTM
	cv_results_full['TIME'] = {"start": start_string, "finish": finish_string}
	cv_results_full['cv_results'] = cv_results
	cv_results_full['REVISION_PROTOCOL'] = REVISION_PROTOCOL
	cv_results_full['SPLIT_MANIFEST'] = str(Path(args.split_manifest).resolve()) if args.split_manifest else None
	cv_results_full['TRAIN_ONLY_PREPROCESSING'] = FIT_PREPROCESSING_ON_TRAIN
	cv_results_full['REVISION_OUTPUT_DIR'] = str(revision_run_dir.resolve()) if revision_run_dir else None

	# 增强功能参数（新增）
	cv_results_full['USE_ENHANCED_CLIMATE'] = USE_ENHANCED_CLIMATE
	cv_results_full['USE_CROSS_MODAL_FUSION'] = USE_CROSS_MODAL_FUSION
	cv_results_full['FUSION_TYPE'] = FUSION_TYPE
	cv_results_full['REVISION_FUSION'] = REVISION_FUSION
	cv_results_full['PRETRAINED_MODEL'] = PRETRAINED_MODEL
	cv_results_full['FREEZE_PRETRAINED'] = FREEZE_PRETRAINED
	
	# 静态特征参数
	cv_results_full['USE_STATIC_FEATURES'] = USE_STATIC_FEATURES
	cv_results_full['STATIC_CSV'] = STATIC_CSV

	# 标签策略参数（消融实验）
	cv_results_full['LABEL_MODE'] = LABEL_MODE
	cv_results_full['HUBER_BETA'] = HUBER_BETA
	cv_results_full['TAIL_THRESHOLD'] = TAIL_THRESHOLD
	cv_results_full['TAIL_WEIGHT'] = TAIL_WEIGHT

	# FiLM 融合参数
	cv_results_full['USE_FILM_FUSION'] = USE_FILM_FUSION
	cv_results_full['FILM_ALPHA_INIT'] = FILM_ALPHA_INIT if USE_FILM_FUSION else None


	# 将 numpy 类型递归转换为 Python 原生类型，避免 JSON 序列化错误
	def to_serializable(obj):
		if isinstance(obj, (np.floating,)):
			return float(obj)
		if isinstance(obj, (np.integer,)):
			return int(obj)
		if isinstance(obj, (np.ndarray,)):
			return obj.tolist()
		if isinstance(obj, dict):
			return {k: to_serializable(v) for k, v in obj.items()}
		if isinstance(obj, list):
			return [to_serializable(v) for v in obj]
		return obj

	with open(f"results/Metrics_{EXP_NAME}_{run_name}.json", "w") as fp:
		json.dump(to_serializable(cv_results), fp, indent=4)

	if REVISION_PROTOCOL:
		revision_summary_path = revision_run_dir / 'run_summary.json'
		with open(revision_summary_path, 'w', encoding='utf-8') as fp:
			json.dump(to_serializable(cv_results_full), fp, indent=2, ensure_ascii=False)
		with open(f"results/RUN_{EXP_NAME}_{run_name}.json", "w") as fp:
			json.dump(to_serializable(cv_results_full), fp, indent=4)
		print(f"[Revision] Completed without test-set model/seed selection: {revision_summary_path}")
		raise SystemExit(0)
		
	# Load the best model
	best_model_path = f"results/RUN_{EXP_NAME}_{run_name}_best.pth.tar"
	import os
	if not os.path.exists(best_model_path):
		print(f"Warning: Best model file not found: {best_model_path}")
		print(f"Best model should be from Seed {best_seed}. Using current model state.")
	else:
		try:
			# 保存当前模型状态（用于验证）
			import torch
			model_state_before = {k: v.clone() for k, v in model.state_dict().items()}
			
			load_checkpoint(model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY), filename=best_model_path)
			
			# 验证模型状态是否改变
			model_state_after = model.state_dict()
			state_changed = any(not torch.equal(model_state_before[k], model_state_after[k]) for k in model_state_before.keys())
			
			if state_changed:
				print(f"Best Model loaded from Seed {best_seed} (RMSE={cv_results['RMSE'][SEEDS.index(best_seed)]:.6f})")
			else:
				print(f"Warning: Model state did not change after loading best checkpoint. Current model may already be the best model.")
		except Exception as e:
			print(f"Error loading best model: {e}")
			print(f"Using current model state (may be from last seed)")

	model.eval()

	test_dl_w_id = DataLoader(test_ds_w_id, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
	# Pass region_ids if regional adaptation is enabled
	test_region_ids_for_eval = test_region_ids if args.use_regional_adaptation else None
	test_step_w_id(model=model, data_loader=test_dl_w_id, loss_fn=nn.L1Loss(), verbose=False, 
				   csv_file=f"results/RUN_{EXP_NAME}_{run_name}_best.csv", 
				   region_ids=test_region_ids_for_eval, label_mode=LABEL_MODE)
	print(f"Best model saved to results/RUN_{EXP_NAME}_{run_name}_best.csv")

	# Augment saved CSV with denormalized columns so downstream analysis uses real-scale values
	best_csv_path = f"results/RUN_{EXP_NAME}_{run_name}_best.csv"
	try:
		df = pd.read_csv(best_csv_path)
		
		# 根据 label_mode 决定如何获取原尺度值
		if LABEL_MODE in ['baseline_raw_mse', 'log1p_mse', 'log1p_huber', 'log1p_huber_w']:
			# 四组消融实验：CSV中已经包含原尺度值 (y_real_raw, y_pred_raw)
			if 'y_real_raw' in df.columns and 'y_pred_raw' in df.columns:
				print(f"[标签策略] CSV已包含原尺度列 (y_real_raw, y_pred_raw)，直接使用")
				y_true = df['y_real_raw']
				y_pred = df['y_pred_raw']
			else:
				# 兼容旧版：如果没有 _raw 列，假设 y_real/y_pred 已经是原尺度
				print(f"[标签策略] 警告: CSV中未找到 y_real_raw/y_pred_raw 列，假设 y_real/y_pred 为原尺度")
				y_true = df['y_real']
				y_pred = df['y_pred']
		else:
			# 默认模式：需要反归一化
			if 'y_real_denorm' not in df.columns or 'y_pred_denorm' not in df.columns:
				df['y_real_denorm'] = df['y_real'] * OC_MAX
				df['y_pred_denorm'] = df['y_pred'] * OC_MAX
				denorm_best_csv_path = f"results/RUN_{EXP_NAME}_{run_name}_best_denorm.csv"
				df.to_csv(denorm_best_csv_path, index=False)
				print(f"Augmented denormalized CSV saved: {denorm_best_csv_path}")
			y_true = df['y_real'] * OC_MAX
			y_pred = df['y_pred'] * OC_MAX
	except Exception as e:
		print(f"Warning: failed to process best CSV: {e}")
		# 回退到默认行为
		y_true = df['y_real'] * OC_MAX
		y_pred = df['y_pred'] * OC_MAX

	rmse, r2, rpiq, mae, mec, ccc = evaluate_regression_metrics(y_true, y_pred)

	best_dict = {}
	best_dict['RMSE'] = float(rmse)
	best_dict['R2'] = float(r2)
	best_dict['RPIQ'] = float(rpiq)
	best_dict['MAE'] = float(mae)
	best_dict['MEC'] = float(mec)
	best_dict['CCC'] = float(ccc)

	# print(best_dict)
	cv_results_full['best_dict'] = best_dict

	# Load worst model for metrics calculation (but don't save the pth file)
	import os
	worst_model_path = f"results/RUN_{EXP_NAME}_{run_name}_worst.pth.tar"
	worst_temp_csv = f"results/RUN_{EXP_NAME}_{run_name}_worst_temp.csv"
	
	if os.path.exists(worst_model_path):
		try:
			load_checkpoint(model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY), filename=worst_model_path)
			model.eval()
			print("Worst Model loaded for metrics calculation")
			test_step_w_id(model=model, data_loader=test_dl_w_id, loss_fn=nn.L1Loss(), verbose=False, 
						   csv_file=worst_temp_csv, region_ids=test_region_ids_for_eval, label_mode=LABEL_MODE)
			
			# Compute metrics for worst model
			df_worst = pd.read_csv(worst_temp_csv)
			
			# 根据 label_mode 决定如何获取原尺度值
			if LABEL_MODE in ['baseline_raw_mse', 'log1p_mse', 'log1p_huber', 'log1p_huber_w']:
				if 'y_real_raw' in df_worst.columns and 'y_pred_raw' in df_worst.columns:
					y_true_worst = df_worst['y_real_raw']
					y_pred_worst = df_worst['y_pred_raw']
				else:
					y_true_worst = df_worst['y_real']
					y_pred_worst = df_worst['y_pred']
			else:
				y_true_worst = df_worst['y_real'] * OC_MAX
				y_pred_worst = df_worst['y_pred'] * OC_MAX

			rmse_worst, r2_worst, rpiq_worst, mae_worst, mec_worst, ccc_worst = evaluate_regression_metrics(y_true_worst, y_pred_worst)

			worst_dict = {}
			worst_dict['RMSE'] = float(rmse_worst)
			worst_dict['R2'] = float(r2_worst)
			worst_dict['RPIQ'] = float(rpiq_worst)
			worst_dict['MAE'] = float(mae_worst)
			worst_dict['MEC'] = float(mec_worst)
			worst_dict['CCC'] = float(ccc_worst)

			# print(worst_dict)
			cv_results_full['worst_dict'] = worst_dict

			# Clean up temporary worst CSV file
			if os.path.exists(worst_temp_csv):
				os.remove(worst_temp_csv)
		except Exception as e:
			print(f"Warning: Failed to load/evaluate worst model: {e}")
			print("Using best model metrics for worst_dict (worst model may be same as best if only one seed was run)")
			cv_results_full['worst_dict'] = best_dict.copy()
	else:
		print(f"Warning: Worst model file not found: {worst_model_path}")
		print("This may occur if only one seed was run. Using best model metrics for worst_dict.")
		cv_results_full['worst_dict'] = best_dict.copy()

	with open(f"results/RUN_{EXP_NAME}_{run_name}.json", "w") as fp:
		json.dump(to_serializable(cv_results_full), fp, indent=4)
