#!/usr/bin/env python3
import argparse
import json
import os
import sys
import threading  # 1. 移动到这里，确保全局可用
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from dotenv import load_dotenv
from tqdm import tqdm

# 注意：请确保这些模块在你的路径中
try:
    from src.memzero.search import MemorySearch
    from metrics.llm_judge import evaluate_llm_judge
    from metrics.utils import calculate_bleu_scores, calculate_metrics
except ImportError as e:
    print(f"导入模块失败: {e}")

load_dotenv()

# 类别映射表
CATEGORY_MAPPING = {
    "1": "multi-hop",
    "2": "temporal",     # 修正：将空字符串改为 "temporal"
    "3": "open-domain",
    "4": "single-hop",
    "5": "adversarial"
}

def process_item(item_data):
    """处理单个评估项目"""
    k, v = item_data
    local_results = defaultdict(list)

    for item in v:
        gt_answer = str(item.get("answer", ""))
        pred_answer = str(item.get("response", ""))
        category = str(item.get("category", ""))
        question = str(item.get("question", ""))

        if category == "5":
            continue

        metrics = calculate_metrics(pred_answer, gt_answer)
        bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)
        llm_score = evaluate_llm_judge(question, gt_answer, pred_answer)

        category_name = CATEGORY_MAPPING.get(category, f"category_{category}")

        local_results[k].append({
            "question": question,
            "answer": gt_answer,
            "response": pred_answer,
            "category": category,
            "category_name": category_name,
            "bleu_score": bleu_scores["bleu1"],
            "f1_score": metrics["f1"],
            "llm_score": llm_score,
        })

    return local_results

def run_evaluation(data_file, output_folder, top_k=30, filter_memories=False, is_graph=False, max_workers=10):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"mem0_eval_top{top_k}_filter{filter_memories}_graph{is_graph}_{timestamp}"
    experiment_dir = os.path.join("/home/suma2/repo/mem0/evaluation/results/mem0_eval_top30_filterFalse_graphFalse_20260102_143036")
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"🚀 开始完整的mem0评估流程")
    print(f"📁 实验结果将保存到: {experiment_dir}")

    # Step 1: 运行搜索
    # print(f"\n🔍 Step 1: 运行记忆搜索...")
    search_results_file = os.path.join(experiment_dir, "search_results.json")
    # memory_searcher = MemorySearch(
    #     output_path=search_results_file,
    #     top_k=top_k,
    #     filter_memories=filter_memories,
    #     is_graph=is_graph
    # )
    # memory_searcher.process_data_file(data_file)
    # print(f"✅ 搜索完成")

    # Step 2: 运行评估
    print(f"\n📊 Step 2: 生成评估指标...")
    eval_metrics_file = os.path.join(experiment_dir, "evaluation_metrics.json")

    with open(search_results_file, "r") as f:
        search_data = json.load(f)

    results = defaultdict(list)
    results_lock = threading.Lock()

    # 优化后的多线程评估逻辑
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 将 items() 转换为列表进行迭代
        items_to_process = list(search_data.items())
        
        # 使用 list() 包装 map 以便立即开始执行，并在 tqdm 中显示进度
        for local_res in tqdm(executor.map(process_item, items_to_process), 
                             total=len(items_to_process), 
                             desc="处理评估"):
            with results_lock:
                for k, items in local_res.items():
                    results[k].extend(items)

    # 2. 修正语法错误：json.() -> json.dump()
    with open(eval_metrics_file, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"✅ 评估指标保存完成")

    # Step 3: 生成得分报告
    print(f"\n📈 Step 3: 生成得分报告...")
    scores_csv_file = os.path.join(experiment_dir, "scores.csv")

    all_items = []
    for key in results:
        all_items.extend(results[key])

    import pandas as pd
    df = pd.DataFrame(all_items)

    category_results = df.groupby("category_name").agg({
        "bleu_score": "mean",
        "f1_score": "mean",
        "llm_score": "mean"
    }).round(4)
    category_results["count"] = df.groupby("category_name").size()

    overall_means = df.agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)
    category_results.to_csv(scores_csv_file)

    # 保存元数据
    metadata = {
        "experiment_name": experiment_name,
        "parameters": {"top_k": top_k, "is_graph": is_graph},
        "overall_scores": overall_means.to_dict()
    }
    with open(os.path.join(experiment_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print("\n📊 评估完成！")
    print(category_results)
    return experiment_dir

def main():
    parser = argparse.ArgumentParser(description="运行完整的mem0评估流程")
    parser.add_argument("--data_file", type=str, default="dataset/locomo10.json")
    parser.add_argument("--output_folder", type=str, default="results/")
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--filter_memories", action="store_true")
    parser.add_argument("--is_graph", action="store_true")
    parser.add_argument("--max_workers", type=int, default=10)

    args = parser.parse_args()

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