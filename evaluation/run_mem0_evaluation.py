#!/usr/bin/env python3
"""
整合的mem0评估流程脚本
将search、evals、generate_scores三个步骤合并为一个流程
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from dotenv import load_dotenv
from tqdm import tqdm

from src.memzero.search import MemorySearch
from metrics.llm_judge import evaluate_llm_judge
from metrics.utils import calculate_bleu_scores, calculate_metrics

load_dotenv()

# 类别映射表
CATEGORY_MAPPING = {
    "1": "multi-hop",
    "2": "temporal", 
    "3": "open-domain",
    "4": "single-hop",
    "5": "adversarial"
}


def process_item(item_data):
    """处理单个评估项目，来自evals.py的逻辑"""
    k, v = item_data
    local_results = defaultdict(list)

    for item in v:
        gt_answer = str(item["answer"])
        pred_answer = str(item["response"])
        category = str(item["category"])
        question = str(item["question"])

        # Skip category 5 (adversarial)
        if category == "5":
            continue

        metrics = calculate_metrics(pred_answer, gt_answer)
        bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)
        llm_score = evaluate_llm_judge(question, gt_answer, pred_answer)

        # 使用类别名称而不是数字
        category_name = CATEGORY_MAPPING.get(category, f"category_{category}")

        local_results[k].append(
            {
                "question": question,
                "answer": gt_answer,
                "response": pred_answer,
                "category": category,
                "category_name": category_name,
                "bleu_score": bleu_scores["bleu1"],
                "f1_score": metrics["f1"],
                "llm_score": llm_score,
            }
        )

    return local_results


def run_evaluation(data_file, output_folder, top_k=30, filter_memories=False, is_graph=False, max_workers=10):
    """运行完整的评估流程"""

    # 创建时间戳用于结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"mem0_eval_top{top_k}_filter{filter_memories}_graph{is_graph}_{timestamp}"

    # 创建实验结果目录
    experiment_dir = os.path.join(output_folder, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"🚀 开始完整的mem0评估流程")
    print(f"📁 实验结果将保存到: {experiment_dir}")

    # Step 1: 运行搜索
    print(f"\n🔍 Step 1: 运行记忆搜索...")
    search_results_file = os.path.join(experiment_dir, "search_results.json")

    memory_searcher = MemorySearch(
        output_path=search_results_file,
        top_k=top_k,
        filter_memories=filter_memories,
        is_graph=is_graph
    )
    memory_searcher.process_data_file(data_file)
    print(f"✅ 搜索完成，结果保存到: {search_results_file}")

    # Step 2: 运行评估
    print(f"\n📊 Step 2: 生成评估指标...")
    eval_metrics_file = os.path.join(experiment_dir, "evaluation_metrics.json")

    with open(search_results_file, "r") as f:
        search_data = json.load(f)

    results = defaultdict(list)
    results_lock = threading.Lock()

    # Use ThreadPoolExecutor with specified workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_item, item_data) for item_data in search_data.items()]

        for future in tqdm(ThreadPoolExecutor(max_workers=max_workers).map(lambda f: f.result(), futures),
                          total=len(futures), desc="处理评估"):
            local_results = future.result()
            with results_lock:
                for k, items in local_results.items():
                    results[k].extend(items)

    # Save evaluation metrics
    with open(eval_metrics_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ 评估完成，指标保存到: {eval_metrics_file}")

    # Step 3: 生成得分并保存CSV
    print(f"\n📈 Step 3: 生成得分报告...")
    scores_csv_file = os.path.join(experiment_dir, "scores.csv")

    # Flatten the data into a list of question items
    all_items = []
    for key in results:
        all_items.extend(results[key])

    # Convert to DataFrame and save as CSV
    import pandas as pd

    df = pd.DataFrame(all_items)

    # 按类别名称分组而不是数字
    category_results = df.groupby("category_name").agg({
        "bleu_score": "mean",
        "f1_score": "mean",
        "llm_score": "mean"
    }).round(4)

    # Add count of questions per category
    category_results["count"] = df.groupby("category_name").size()

    # Calculate overall means
    overall_means = df.agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)

    # Save results to CSV
    category_results.to_csv(scores_csv_file)
    print(f"✅ 得分报告保存到: {scores_csv_file}")

    # 打印结果到终端
    print("\n📊 评估结果:")
    print("\n各类别平均得分:")
    print(category_results)

    print("\n总体平均得分:")
    print(overall_means)

    # 创建实验元数据文件
    metadata = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "parameters": {
            "data_file": data_file,
            "top_k": top_k,
            "filter_memories": filter_memories,
            "is_graph": is_graph,
            "max_workers": max_workers
        },
        "files": {
            "search_results": search_results_file,
            "evaluation_metrics": eval_metrics_file,
            "scores_csv": scores_csv_file
        },
        "overall_scores": overall_means.to_dict()
    }

    metadata_file = os.path.join(experiment_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n🎯 实验完成！所有结果保存在: {experiment_dir}")
    print(f"📋 实验元数据: {metadata_file}")

    return experiment_dir


def main():
    parser = argparse.ArgumentParser(description="运行完整的mem0评估流程")
    parser.add_argument(
        "--data_file", type=str, default="dataset/locomo10.json",
        help="数据集文件路径"
    )
    parser.add_argument(
        "--output_folder", type=str, default="results/",
        help="输出文件夹路径"
    )
    parser.add_argument(
        "--top_k", type=int, default=30,
        help="检索的记忆数量"
    )
    parser.add_argument(
        "--filter_memories", action="store_true", default=False,
        help="是否过滤记忆"
    )
    parser.add_argument(
        "--is_graph", action="store_true", default=False,
        help="是否使用图谱搜索"
    )
    parser.add_argument(
        "--max_workers", type=int, default=10,
        help="最大工作线程数"
    )

    args = parser.parse_args()

    # 导入threading以支持ThreadPoolExecutor
    import threading

    run_evaluation(
        data_file=args.data_file,
        output_folder=args.output_folder,
        top_k=args.top_k,
        filter_memories=args.filter_memories,
        is_graph=args.is_graph,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()