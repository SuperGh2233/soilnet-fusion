#!/bin/bash
# 运行 baseline_raw_mse 作为真正的基线对比

echo "=========================================="
echo "运行基线实验: baseline_raw_mse"
echo "=========================================="
echo "这是真正的基线，用于对比 log1p 策略的效果"
echo "=========================================="
echo ""

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
echo "基线实验完成!"
echo "=========================================="
echo ""
echo "如果 baseline 效果更好，说明："
echo "  1. log1p 变换可能不适合这个数据集"
echo "  2. 可以尝试 log1p_huber_w（加权版本）"
echo "  3. 或者考虑其他改进方向（数据增强、模型调整等）"
echo ""

































