#!/bin/bash
# 运行下一个实验：log1p_huber

echo "=========================================="
echo "运行实验: log1p_huber"
echo "=========================================="
echo "配置:"
echo "  Label Mode: log1p_huber"
echo "  Huber Beta: 1.0"
echo "  Epochs: 60"
echo "  Seeds: 1"
echo "=========================================="
echo ""

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
echo "实验完成!"
echo "=========================================="
echo ""
echo "如果效果仍不理想，可以尝试:"
echo "  --label_mode log1p_huber_w --tail_threshold 30.0 --tail_weight 2.0"
echo ""


































