#!/usr/bin/env python3
"""
分析locomo数据集中answer的长度统计
"""

import json
from collections import Counter

def analyze_answers(data_file="dataset/locomo10.json"):
    """分析answer长度"""
    print(f"📊 分析文件: {data_file}")
    
    with open(data_file, "r") as f:
        data = json.load(f)
    
    # 收集所有answer及其长度信息
    answer_lengths = []
    answer_details = []
    total_answers = 0
    
    for item in data:
        qa_pairs = item.get("qa", [])
        for qa in qa_pairs:
            answer = qa.get("answer", "")
            if answer:
                answer_str = str(answer)
                length = len(answer_str)
                answer_lengths.append(length)
                answer_details.append({
                    "answer": answer_str,
                    "length": length,
                    "category": qa.get("category", ""),
                    "question": qa.get("question", "")
                })
                total_answers += 1
    
    if not answer_lengths:
        print("❌ 没有找到answer数据")
        return
    
    # 计算统计信息
    max_length = max(answer_lengths)
    min_length = min(answer_lengths)
    avg_length = sum(answer_lengths) / len(answer_lengths)
    median_length = sorted(answer_lengths)[len(answer_lengths) // 2]
    
    # 找到最长和最短的answer
    longest_answers = [detail for detail in answer_details if detail["length"] == max_length]
    shortest_answers = [detail for detail in answer_details if detail["length"] == min_length]
    
    # 长度分布
    length_distribution = Counter(answer_lengths)
    
    print("=" * 50)
    print("📈 Answer长度统计")
    print("=" * 50)
    print(f"📊 总answer数量: {total_answers}")
    print(f"📏 最长answer长度: {max_length} 字符")
    print(f"📏 最短answer长度: {min_length} 字符")
    print(f"📏 平均长度: {avg_length:.1f} 字符")
    print(f"📏 中位数长度: {median_length} 字符")
    
    print("\n" + "=" * 50)
    print(f"🏆 最长的answer (长度: {max_length}):")
    print("=" * 50)
    for i, detail in enumerate(longest_answers[:3], 1):  # 只显示前3个
        print(f"\n{i}. Question: {detail['question'][:100]}...")
        print(f"   Category: {detail['category']}")
        print(f"   Answer: {detail['answer']}")
    
    print("\n" + "=" * 50)
    print(f"📝 最短的answer (长度: {min_length}):")
    print("=" * 50)
    for i, detail in enumerate(shortest_answers[:3], 1):  # 只显示前3个
        print(f"\n{i}. Question: {detail['question'][:100]}...")
        print(f"   Category: {detail['category']}")
        print(f"   Answer: {detail['answer']}")
    
    print("\n" + "=" * 50)
    print("📊 长度分布 (按字符数分组):")
    print("=" * 50)
    
    # 按长度范围分组
    length_ranges = [
        (0, 10, "0-10"),
        (11, 20, "11-20"), 
        (21, 30, "21-30"),
        (31, 50, "31-50"),
        (51, 100, "51-100"),
        (101, 200, "101-200"),
        (201, float('inf'), "200+")
    ]
    
    for min_len, max_len, label in length_ranges:
        if max_len == float('inf'):
            count = sum(1 for length in answer_lengths if length >= min_len)
        else:
            count = sum(1 for length in answer_lengths if min_len <= length <= max_len)
        percentage = (count / total_answers) * 100
        bar = "█" * int(percentage / 2)
        print(f"{label:>8}: {count:>4} ({percentage:>5.1f}%) {bar}")
    
    # Top 10 最长的answers
    print("\n" + "=" * 50)
    print("🏆 Top 10 最长的answers:")
    print("=" * 50)
    top_10 = sorted(answer_details, key=lambda x: x["length"], reverse=True)[:10]
    for i, detail in enumerate(top_10, 1):
        print(f"\n{i}. 长度: {detail['length']} 字符")
        print(f"   Category: {detail['category']}")
        print(f"   Question: {detail['question'][:80]}...")
        print(f"   Answer: {detail['answer'][:100]}{'...' if len(detail['answer']) > 100 else ''}")
    
    return {
        "total": total_answers,
        "max": max_length,
        "min": min_length,
        "avg": avg_length,
        "median": median_length,
        "longest_answers": longest_answers,
        "all_lengths": answer_lengths
    }

if __name__ == "__main__":
    import sys
    import os
    
    # 添加evaluation目录到路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # 分析locomo数据集
    analyze_answers("dataset/locomo10.json")