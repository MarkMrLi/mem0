#!/usr/bin/env python3
"""
Full-Context实验运行脚本

运行流程：
1. 使用full-context方式处理locomo数据
2. 使用evals.py评估结果
3. 使用generate_scores.py生成评分报告
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

def run_full_context_experiment(data_file, output_folder):
    """运行full-context实验"""
    print(f"🚀 开始Full-Context实验...")
    print(f"📁 数据文件: {data_file}")
    print(f"📁 输出目录: {output_folder}")
    
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)
    
    # 运行full-context实验
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"full_context_eval_{timestamp}"
    experiment_dir = os.path.join(output_folder, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    output_file = os.path.join(experiment_dir, "full_context_results.json")
    
    # 导入并运行full_context处理器
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.full_context import FullContextProcessor
    
    processor = FullContextProcessor(data_path=data_file)
    processor.process_all_conversations(output_file)
    
    print(f"✅ Full-Context实验完成，结果保存到: {output_file}")
    return experiment_dir, output_file

def run_evaluation(results_file, experiment_dir, max_workers=10):
    """运行评估"""
    print(f"\n📊 开始评估Full-Context结果...")
    
    # 使用evals.py评估结果
    eval_output = os.path.join(experiment_dir, "evaluation_metrics.json")
    
    cmd = [
        "python", "evals.py",
        "--input_file", results_file,
        "--output_file", eval_output,
        "--max_workers", str(max_workers)
    ]
    
    print(f"🔧 运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ 评估失败:")
        print(result.stderr)
        sys.exit(1)
    
    print(f"✅ 评估完成，结果保存到: {eval_output}")
    return eval_output

def generate_score_report(eval_metrics_file, experiment_dir):
    """生成评分报告"""
    print(f"\n📈 开始生成评分报告...")
    
    scores_csv = os.path.join(experiment_dir, "scores.csv")
    
    cmd = [
        "python", "generate_scores.py",
        "--input_file", eval_metrics_file,
        "--output_csv", scores_csv
    ]
    
    print(f"🔧 运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ 评分报告生成失败:")
        print(result.stderr)
        sys.exit(1)
    
    print(f"✅ 评分报告生成完成")
    print(result.stdout)
    return scores_csv

def main():
    parser = argparse.ArgumentParser(description="运行Full-Context实验和评估")
    parser.add_argument("--data_file", type=str, default="dataset/locomo10.json", 
                       help="Locomo数据文件路径")
    parser.add_argument("--output_folder", type=str, default="results/", 
                       help="输出目录")
    parser.add_argument("--max_workers", type=int, default=10, 
                       help="评估时的最大工作线程数")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🎯 Full-Context实验和评估流程")
    print("=" * 50)
    
    try:
        # Step 1: 运行full-context实验
        experiment_dir, results_file = run_full_context_experiment(
            args.data_file, args.output_folder
        )
        
        # Step 2: 运行评估
        eval_metrics_file = run_evaluation(
            results_file, experiment_dir, args.max_workers
        )
        
        # Step 3: 生成评分报告
        scores_csv = generate_score_report(eval_metrics_file, experiment_dir)
        
        print("\n" + "=" * 50)
        print("🎉 全部流程完成！")
        print(f"📁 实验目录: {experiment_dir}")
        print(f"📊 结果文件: {results_file}")
        print(f"📈 评估指标: {eval_metrics_file}")
        print(f"📋 评分报告: {scores_csv}")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 流程执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()