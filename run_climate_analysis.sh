#!/bin/bash
# 运行气候变量重要性分析

echo "=========================================="
echo "气候变量重要性分析"
echo "=========================================="

# 使用简化版（仅相关性分析，不需要模型）
python analyze_climate_importance_simple.py \
    --csv_path dataset/CN-SOC-3500_new.csv \
    --climate_folder dataset/Climate/climate_exports_indexed_2011_2015/output_filled_norm \
    --target_col OC \
    --output_dir climate_analysis \
    --temporal

echo ""
echo "=========================================="
echo "分析完成!"
echo "=========================================="
echo ""
echo "结果文件:"
echo "  - climate_analysis/climate_correlation_results.csv"
echo "  - climate_analysis/climate_correlation_analysis.png"
echo "  - climate_analysis/climate_temporal_patterns.png"
echo ""




























