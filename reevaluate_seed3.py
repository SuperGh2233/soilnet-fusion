"""
重新评估Seed 3的模型（如果best.pth.tar确实是Seed 6的）
由于没有Seed 3的独立模型文件，我们需要：
1. 重新训练Seed 3（推荐）
2. 或者使用估算值（不准确）
"""
import json
import subprocess
import sys

json_path = 'results/RUN_11.03_add_static_D_2025_11_03_T_18_56.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

best_seed = data['Best Seed']
exp_name = '11.03_add_static_seed3_recovery'

print("=" * 60)
print("重新训练Seed 3以获取正确的best模型指标")
print("=" * 60)
print(f"\n目标：获取Seed {best_seed}在测试集上的真实指标")
print(f"\n由于当前best.pth.tar保存的是Seed 6的模型，我们需要重新训练Seed {best_seed}")
print(f"\n将运行以下命令（只训练Seed {best_seed}）:")
print(f"python train.py -e {exp_name} -d CHINA -w 8 -cnn ViT-CoMer -rnn Transformer")
print(f"  -trbs 32 -ne 60 -lr 0.0001 -ls step -srtm -lstm -stm -seed {best_seed}")
print("\n注意：这会重新训练Seed 3，可能需要一些时间")
print("=" * 60)

response = input("\n是否继续？(y/n): ")
if response.lower() != 'y':
    print("已取消")
    sys.exit(0)

# 构建训练命令
cmd = [
    'python', 'train.py',
    '-e', exp_name,
    '-d', 'CHINA',
    '-w', '8',
    '-cnn', 'ViT-CoMer',
    '-rnn', 'Transformer',
    '-trbs', '32',
    '-ne', '60',
    '-lr', '0.0001',
    '-ls', 'step',
    '-srtm',
    '-lstm',
    '-stm',
    '-seed', str(best_seed)
]

print(f"\n执行命令: {' '.join(cmd)}")
print("\n开始训练...")

try:
    result = subprocess.run(cmd, check=True)
    print("\n训练完成！")
    print(f"\n请检查结果文件，然后运行以下脚本更新JSON:")
    print(f"python update_json_with_seed3_results.py")
except subprocess.CalledProcessError as e:
    print(f"\n训练失败: {e}")
    sys.exit(1)
















