"""
综合区域实验脚本
集成区域fine-tuning、权重损失、可视化验证
"""

import os
import sys
import subprocess
import time

def run_experiment(experiment_name, command, description):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"实验: {experiment_name}")
    print(f"描述: {description}")
    print(f"命令: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ 实验完成 (耗时: {duration:.1f}秒)")
            return True
        else:
            print(f"❌ 实验失败 (耗时: {duration:.1f}秒)")
            print(f"错误信息: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 实验异常: {str(e)}")
        return False

def main():
    """主函数"""
    print("=== 综合区域实验套件 ===")
    print("包含: 区域fine-tuning + 权重损失 + 可视化验证")
    
    experiments = [
        {
            "name": "区域权重损失训练",
            "command": "python train.py -e EXP_RegionWeight_InvFreq -d CHINA -w 8 -cnn ViT-CoMer -rnn Transformer -trbs 32 -ne 80 -lr 0.00005 -ls plateau -srtm -lstm -ra -nr 6 -rs geographic -seed 1 2 3",
            "description": "使用反频率权重平衡区域样本差异"
        },
        {
            "name": "区域fine-tuning",
            "command": "python region_specific_finetuning.py",
            "description": "为每个地理分区单独进行fine-tuning"
        },
        {
            "name": "区域可视化分析",
            "command": "python regional_visualization.py",
            "description": "生成中国地图和区域误差分析图"
        },
        {
            "name": "区域权重分析",
            "command": "python region_weighted_loss.py",
            "description": "分析区域不平衡问题并计算权重"
        }
    ]
    
    results = {}
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n进度: {i}/{len(experiments)}")
        success = run_experiment(exp["name"], exp["command"], exp["description"])
        results[exp["name"]] = success
        
        if not success:
            print(f"⚠️  {exp['name']} 失败，继续下一个实验...")
    
    # 总结报告
    print(f"\n{'='*60}")
    print("实验总结报告")
    print(f"{'='*60}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"总实验数: {total_count}")
    print(f"成功数: {success_count}")
    print(f"失败数: {total_count - success_count}")
    print(f"成功率: {success_count/total_count*100:.1f}%")
    
    print(f"\n详细结果:")
    for name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n=== 下一步建议 ===")
    if success_count == total_count:
        print("🎉 所有实验都成功完成！")
        print("建议:")
        print("1. 查看生成的可视化图表")
        print("2. 分析区域fine-tuning结果")
        print("3. 比较不同权重策略的效果")
        print("4. 根据分析结果优化模型")
    else:
        print("⚠️  部分实验失败，建议:")
        print("1. 检查失败实验的错误信息")
        print("2. 确保所有依赖都已安装")
        print("3. 检查数据文件路径是否正确")
        print("4. 重新运行失败的实验")

if __name__ == "__main__":
    main()
