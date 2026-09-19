# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SoilNet is a hybrid Transformer-based framework with self-supervised learning for Soil Organic Carbon (SOC) prediction from satellite imagery and climate time-series data. Published in IEEE TGRS (2024). The training pipeline has two phases: (1) self-supervised contrastive learning (SimCLR), then (2) supervised fine-tuning.

## Commands

### Training

```bash
# Self-supervised pre-training (Phase 1)
python train_ssl.py --num_workers 8 --trbs 64 --lr 0.0001 --num_epochs 100 --lr_scheduler 'step' --dataset 'LUCAS' --use_srtm --use_lstm_branch --cnn_architecture 'ViT' --rnn_architecture 'Transformer' --seeds 1 42 86

# Supervised fine-tuning (Phase 2)
python train.py --dataset 'LUCAS' --num_workers 8 --load_simclr_model --trbs 64 --lr 0.0001 --num_epochs 100 --lr_scheduler 'step' --use_srtm --use_lstm_branch --seeds 1 42 86

# Training from scratch (no SSL pre-training)
python train.py --dataset 'LUCAS' --num_workers 8 --cnn_architecture 'ViT' --rnn_architecture 'Transformer' --trbs 64 --lr 0.0001 --num_epochs 100 --lr_scheduler 'step' --use_srtm --use_lstm_branch --seeds 1 42 86

# Full help
python train.py --help
python train_ssl.py --help
```

### Running Tests

```bash
# Individual test files (no pytest; they use if __name__ == "__main__" blocks or standalone scripts)
python test_semantic_aligned_fusion.py
python test_spectral_cnn.py
python test_scmrl_integration.py
python test_label_strategy.py
python test_climate_data.py
python soilnet/test_simple.py
```

### Environment Setup

```bash
conda env create -f requirements/pytorch_reqs.yml
```

Environment name: `pytorchGPU`. Python 3.9, PyTorch 1.13.1 with CUDA 11.7.

## Architecture

### Model Hierarchy (`soilnet/soil_net.py`)

```
SoilNet (image-only)
├── Visual encoder (CNN/ViT/spectral_cnn)
└── MultiHeadRegressor

SoilNetLSTM (image + climate, the primary model)
├── Visual encoder
├── RNN/LSTM/GRU/Transformer (climate branch)
├── [Optional] RegionEmbedding
├── [Optional] S-CMRL fusion (SemanticAlignedFusion)
└── MultiHeadRegressor (or Linear if S-CMRL)

SoilNetLSTMWithStatic (image + climate + static features, in soil_net_static.py)
├── Extends SoilNetLSTM
├── StaticBranch (MLP for numeric, nn.Embedding for categorical like LULC)
└── Static features fed to regressor or S-CMRL fusion

SoilNetJustLSTM (ablation: climate-only, no CNN)
└── Inherits SoilNetLSTM but skips CNN creation

SoilNetSimCLR / SoilNetSimCLRwRegHead (self-supervised pre-training)
└── Dual-branch encoder outputting embeddings for contrastive loss
```

### Visual Encoder Options (selected via `--cnn` flag)

- `resnet50`, `resnet101` — Modified torchvision ResNets with custom first conv for arbitrary input channels
- `vgg16` — VGG16 with optional GLAM attention
- `ViT` — Standard Vision Transformer (patch_size=8, embed_dim=768)
- `ViT-CoMer` — Hybrid CNN+ViT with CTI fusion (patch_size=16, embed_dim=768, depth=12). Uses `vit_comer.py`, NOT the vendored `ViT-CoMer/` directory
- `MPViT`, `HybridViT` — Multi-path ViT variants (embed_dim=768)
- `spectral_cnn` — MultiPreprocSpectralCNN (Tziolas et al., Geoderma 2024), bypasses spatial processing entirely

### Climate Encoder Options (selected via `--rnn` flag)

- `LSTM`, `GRU`, `RNN` — Standard RNN modules in `soilnet/submodules/rnn.py`
- `Transformer` — TSTransformerEncoderClassiregressor (d_model=512, n_heads=8, 6 layers)

Specifying `--rnn` automatically enables the climate branch (`--use_lstm_branch`).

### Fusion Strategies

- **Default concat**: `MultiHeadRegressor` projects each branch to hidden_size, concatenates, outputs scalar
- **S-CMRL** (`--use_scmrl_fusion`): Climate is "strong" modality; visual/static are "weak". Uses cross-modal residual attention with learnable alpha. InfoNCE alignment loss during training. Supports sequential or parallel mode (`--scmrl_parallel`)

### Label Strategies (`--label_mode`, for long-tail SOC ablation)

- `baseline_raw_mse` — Raw SOC scale + MSE loss
- `log1p_mse` — log1p(SOC) + MSE
- `log1p_huber` — log1p(SOC) + Huber loss
- `log1p_huber_w` — log1p(SOC) + weighted Huber (high-value sample weighting)

When `--label_mode` is not set, labels are normalized to [0,1] (default behavior).

## Key Files

| File | Purpose |
|------|---------|
| `config.py` | Default paths for data, climate CSVs, pre-trained models |
| `train.py` | Main supervised training script (extensive argparse) |
| `train_ssl.py` | Self-supervised contrastive training |
| `train_utils.py` | Training loops, loss functions, metrics, checkpoint save/load |
| `soilnet/soil_net.py` | Core model definitions |
| `soilnet/soil_net_static.py` | Static feature model extension |
| `soilnet/static_branch.py` | Static feature encoder (MLP + embedding) |
| `soilnet/submodules/cnn_feature_extractor.py` | All CNN/ResNet/VGG visual encoders |
| `soilnet/submodules/vit.py` | Standard ViT |
| `soilnet/submodules/vit_comer.py` | ViT-CoMer hybrid (CNN+ViT with CTI fusion) |
| `soilnet/submodules/regressor.py` | MultiHeadRegressor (multi-modal fusion head) |
| `soilnet/submodules/rnn.py` | LSTM/GRU/RNN modules |
| `soilnet/submodules/semantic_aligned_fusion.py` | S-CMRL fusion module |
| `soilnet/submodules/spectral_cnn.py` | MultiPreprocSpectralCNN |
| `soilnet/submodules/region_embedding.py` | Regional adaptation module |
| `dataset/dataset_loader.py` | LUCAS (European) dataset loader |
| `dataset/dataset_loader_us.py` | RaCA (US) dataset loader |
| `dataset/dataset_loader_china.py` | China dataset loader |
| `dataset/dataset_loader_china_static.py` | China dataset with static features |

## Data Layout

```
config.py paths (currently configured for China 3500 dataset):
  Images:  dataset/l8_images_CN_split/{train,test,val}/
  Labels:  dataset/CN-SOC-3500_new.csv
  Climate: dataset/Climate/climate_exports_indexed_2011_2015/output_filled_norm/
```

Each dataset folder contains `.tif` files named `{Point_ID}_{date}_{bandInfo}.tif`. The Point_ID is extracted from the filename to look up labels in the CSV and climate data in the climate CSVs.

## Dataset Selection Logic (`train.py`)

The `--dataset` flag controls which dataset loader is imported:
- `LUCAS` → `dataset/dataset_loader.py`
- `RaCA` → `dataset/dataset_loader_us.py`
- `CHINA` → `dataset/dataset_loader_china.py` (+ optional `dataset_loader_china_static.py`)

Each dataset has its own `OC_MAX` value used for label normalization: LUCAS=87, RaCA=4115, CHINA=60.

## Important Conventions

- **No test framework**: Tests are standalone Python scripts, not pytest/unittest. Run them directly with `python test_*.py`.
- **Bilingual comments**: Code comments and docstrings mix Chinese and English. Variable/class names are always English.
- **Device handling**: Uses `device = "cuda" if torch.cuda.is_available() else "cpu"` pattern throughout.
- **Checkpoint format**: Dict with `"state_dict"` and `"optimizer"` keys. The `save_checkpoint` function uses atomic write (temp file + rename).
- **Pretrained weight loading**: The `_load_pretrained_weights` method handles channel adaptation (3→14 channels via mean-replication), positional embedding interpolation, and key prefix stripping. Duplicated in `SoilNet` and `SoilNetLSTM`.
- **`soilnet/__init__.py`**: Adds `submodules/` to `sys.path`, so imports within the soilnet package use `from submodules.x import Y` (not `from soilnet.submodules.x`).
- **Results directory**: Training creates `results/` folder automatically. Model files, metrics JSON, and loss plots are saved there with timestamps.
