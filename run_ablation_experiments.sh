#!/bin/bash
# 批量运行四组消融实验
# 使用方式: bash run_ablation_experiments.sh

# 设置通用参数
NUM_EPOCHS=100
SEEDS="1 2 3"
CNN_ARCH="ViT-CoMer"
RNN_ARCH="Transformer"
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
BATCH_SIZE=32

echo "=========================================="
echo "开始运行标签策略消融实验"
echo "=========================================="
echo "配置:"
echo "  Epochs: $NUM_EPOCHS"
echo "  Seeds: $SEEDS"
echo "  CNN: $CNN_ARCH"
echo "  RNN: $RNN_ARCH"
echo "  LR: $LEARNING_RATE"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Batch Size: $BATCH_SIZE"
echo "=========================================="
echo ""

# 实验1: baseline_raw_mse
echo "[1/4] 运行实验: baseline_raw_mse"
python train.py \
    -e baseline_raw_mse \
    --label_mode baseline_raw_mse \
    -ne $NUM_EPOCHS \
    -seed $SEEDS \
    -lstm \
    -cnn $CNN_ARCH \
    -rnn $RNN_ARCH \
    -lr $LEARNING_RATE \
    -wd $WEIGHT_DECAY \
    -trbs $BATCH_SIZE \
    -tsbs $BATCH_SIZE

echo ""
echo "=========================================="
echo ""

# 实验2: log1p_mse
echo "[2/4] 运行实验: log1p_mse"
python train.py \
    -e log1p_mse \
    --label_mode log1p_mse \
    -ne $NUM_EPOCHS \
    -seed $SEEDS \
    -lstm \
    -cnn $CNN_ARCH \
    -rnn $RNN_ARCH \
    -lr $LEARNING_RATE \
    -wd $WEIGHT_DECAY \
    -trbs $BATCH_SIZE \
    -tsbs $BATCH_SIZE

echo ""
echo "=========================================="
echo ""

# 实验3: log1p_huber
echo "[3/4] 运行实验: log1p_huber"
python train.py \
    -e log1p_huber \
    --label_mode log1p_huber \
    --huber_beta 1.0 \
    -ne $NUM_EPOCHS \
    -seed $SEEDS \
    -lstm \
    -cnn $CNN_ARCH \
    -rnn $RNN_ARCH \
    -lr $LEARNING_RATE \
    -wd $WEIGHT_DECAY \
    -trbs $BATCH_SIZE \
    -tsbs $BATCH_SIZE

echo ""
echo "=========================================="
echo ""

# 实验4: log1p_huber_w
echo "[4/4] 运行实验: log1p_huber_w"
python train.py \
    -e log1p_huber_w \
    --label_mode log1p_huber_w \
    --huber_beta 1.0 \
    --tail_threshold 30.0 \
    --tail_weight 2.0 \
    -ne $NUM_EPOCHS \
    -seed $SEEDS \
    -lstm \
    -cnn $CNN_ARCH \
    -rnn $RNN_ARCH \
    -lr $LEARNING_RATE \
    -wd $WEIGHT_DECAY \
    -trbs $BATCH_SIZE \
    -tsbs $BATCH_SIZE

echo ""
echo "=========================================="
echo "所有实验完成!"
echo "=========================================="
echo ""
echo "结果文件位置:"
echo "  results/RUN_baseline_raw_mse_*.json"
echo "  results/RUN_log1p_mse_*.json"
echo "  results/RUN_log1p_huber_*.json"
echo "  results/RUN_log1p_huber_w_*.json"
echo ""
echo "运行分析脚本查看结果对比:"
echo "  python analyze_ablation_results.py"
echo ""



































