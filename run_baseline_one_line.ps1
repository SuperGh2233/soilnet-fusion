# PowerShell 单行命令：运行基线实验
# 使用方式: .\run_baseline_one_line.ps1

python train.py -e baseline_raw_mse -d CHINA -w 6 -cnn ViT-CoMer -rnn Transformer -lstm -trbs 32 -ne 60 -lr 0.0001 -ls step -stm -srtm --label_mode baseline_raw_mse -seed 1

































