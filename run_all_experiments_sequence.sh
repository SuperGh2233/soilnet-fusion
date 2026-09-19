#!/bin/bash
# 按顺序运行所有四个消融实验

echo "=========================================="
echo "开始运行标签策略消融实验序列"
echo "=========================================="
echo ""

# 实验1: baseline_raw_mse
echo "[1/4] 运行实验: baseline_raw_mse"
python train.py \
    -e baseline_raw_mse \
    -d CHINA \
    -w 6 \
    -cnn ViT-CoMer \
    -rnn Transformer \
    -lstm \
    -trbs 32 \
    -ne 60 \
    -lr 0.0001 \
    -ls step \
    -stm \
    -srtm \
    --label_mode baseline_raw_mse \
    -seed 1

echo ""
echo "=========================================="
echo ""

# 实验2: log1p_mse (已完成，跳过)
echo "[2/4] 实验 log1p_mse 已完成，跳过"
echo ""

# 实验3: log1p_huber
echo "[3/4] 运行实验: log1p_huber"
python train.py \
    -e log1p_huber \
    -d CHINA \
    -w 6 \
    -cnn ViT-CoMer \
    -rnn Transformer \
    -lstm \
    -trbs 32 \
    -ne 60 \
    -lr 0.0001 \
    -ls step \
    -stm \
    -srtm \
    --label_mode log1p_huber \
    --huber_beta 1.0 \
    -seed 1

echo ""
echo "=========================================="
echo ""

# 实验4: log1p_huber_w
echo "[4/4] 运行实验: log1p_huber_w"
python train.py \
    -e log1p_huber_w \
    -d CHINA \
    -w 6 \
    -cnn ViT-CoMer \
    -rnn Transformer \
    -lstm \
    -trbs 32 \
    -ne 60 \
    -lr 0.0001 \
    -ls step \
    -stm \
    -srtm \
    --label_mode log1p_huber_w \
    --huber_beta 1.0 \
    --tail_threshold 30.0 \
    --tail_weight 2.0 \
    -seed 1

echo ""
echo "=========================================="
echo "所有实验完成!"
echo "=========================================="
echo ""
echo "运行分析脚本查看结果对比:"
echo "  python analyze_ablation_results.py"
echo ""


