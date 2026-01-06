#!/usr/bin/env python3
"""
分析Full-Context实验的Token使用情况
"""

import json
import argparse
from collections import defaultdict

def analyze_token_usage(results_file):
    """分析token使用情况"""
    print(f"📊 分析文件: {results_file}")
    
    with open(results_file, "r") as f:
        data = json.load(f)
    
    # 收集所有token使用数据
    token_data = []
    category_stats = defaultdict(list)
    conversation_stats = defaultdict(list)
    
    for conv_id, questions in data.items():
        for qa in questions:
            item = {
                "conv_id": conv_id,
                "category": qa.get("category", ""),
                "input_tokens": qa.get("input_tokens", 0),
                "output_tokens": qa.get("output_tokens", 0),
                "total_tokens": qa.get("total_tokens", 0),
                "context_chars": qa.get("context_chars", 0),
                "context_length": qa.get("context_length", 0),
                "response_time": qa.get("response_time", 0),
                "question": qa.get("question", ""),
                "response": qa.get("response", "")
            }
            token_data.append(item)
            category_stats[item["category"]].append(item)
            conversation_stats[conv_id].append(item)
    
    if not token_data:
        print("❌ 没有找到token使用数据")
        return
    
    # 总体统计
    total_input = sum(item["input_tokens"] for item in token_data)
    total_output = sum(item["output_tokens"] for item in token_data)
    total_all = sum(item["total_tokens"] for item in token_data)
    total_chars = sum(item["context_chars"] for item in token_data)
    
    avg_input = total_input / len(token_data)
    avg_output = total_output / len(token_data)
    avg_total = total_all / len(token_data)
    avg_chars = total_chars / len(token_data)
    
    print("=" * 60)
    print("📊 Token使用总体统计")
    print("=" * 60)
    print(f"📈 总问题数: {len(token_data)}")
    print(f"🔤 总输入tokens: {total_input:,}")
    print(f"🔤 总输出tokens: {total_output:,}")
    print(f"🔤 总tokens: {total_all:,}")
    print(f"📝 平均输入tokens: {avg_input:.1f}")
    print(f"📝 平均输出tokens: {avg_output:.1f}")
    print(f"📝 平均总tokens: {avg_total:.1f}")
    print(f"📄 平均上下文字符数: {avg_chars:.1f}")
    
    if avg_chars > 0 and avg_input > 0:
        chars_per_token = avg_chars / avg_input
        print(f"📊 字符/token比例: {chars_per_token:.2f}")
    
    # 极值统计
    max_input = max(token_data, key=lambda x: x["input_tokens"])
    min_input = min(token_data, key=lambda x: x["input_tokens"])
    max_output = max(token_data, key=lambda x: x["output_tokens"])
    min_output = min(token_data, key=lambda x: x["output_tokens"])
    max_total = max(token_data, key=lambda x: x["total_tokens"])
    min_total = min(token_data, key=lambda x: x["total_tokens"])
    
    print("\n" + "=" * 60)
    print("🏆 Token极值统计")
    print("=" * 60)
    print(f"📈 最大输入tokens: {max_input['input_tokens']:,}")
    print(f"   Question: {max_input['question'][:80]}...")
    print(f"📉 最小输入tokens: {min_input['input_tokens']:,}")
    print(f"   Question: {min_input['question'][:80]}...")
    print(f"📈 最大输出tokens: {max_output['output_tokens']:,}")
    print(f"   Question: {max_output['question'][:80]}...")
    print(f"📉 最小输出tokens: {min_output['output_tokens']:,}")
    print(f"   Question: {min_output['question'][:80]}...")
    
    # 按类别统计
    print("\n" + "=" * 60)
    print("📂 按Category统计")
    print("=" * 60)
    
    category_mapping = {
        "1": "multi-hop",
        "2": "temporal",
        "3": "open-domain", 
        "4": "single-hop",
        "5": "adversarial"
    }
    
    for category, items in sorted(category_stats.items()):
        cat_name = category_mapping.get(category, f"category_{category}")
        cat_total_input = sum(item["input_tokens"] for item in items)
        cat_total_output = sum(item["output_tokens"] for item in items)
        cat_avg_total = cat_total_input + cat_total_output / len(items)
        
        print(f"\n📊 {cat_name} (category: {category})")
        print(f"   问题数: {len(items)}")
        print(f"   平均输入tokens: {cat_total_input / len(items):.1f}")
        print(f"   平均输出tokens: {cat_total_output / len(items):.1f}")
        print(f"   平均总tokens: {(cat_total_input + cat_total_output) / len(items):.1f}")
    
    # Token分布
    print("\n" + "=" * 60)
    print("📊 Token使用分布")
    print("=" * 60)
    
    input_ranges = [
        (0, 1000, "0-1K"),
        (1000, 2000, "1K-2K"),
        (2000, 4000, "2K-4K"),
        (4000, 8000, "4K-8K"),
        (8000, 16000, "8K-16K"),
        (16000, float('inf'), "16K+")
    ]
    
    print("输入tokens分布:")
    for min_tokens, max_tokens, label in input_ranges:
        if max_tokens == float('inf'):
            count = sum(1 for item in token_data if item["input_tokens"] >= min_tokens)
        else:
            count = sum(1 for item in token_data if min_tokens <= item["input_tokens"] < max_tokens)
        percentage = (count / len(token_data)) * 100
        bar = "█" * int(percentage / 2)
        print(f"{label:>8}: {count:>4} ({percentage:>5.1f}%) {bar}")
    
    print("\n输出tokens分布:")
    output_ranges = [
        (0, 50, "0-50"),
        (50, 100, "50-100"),
        (100, 200, "100-200"),
        (200, 500, "200-500"),
        (500, float('inf'), "500+")
    ]
    
    for min_tokens, max_tokens, label in output_ranges:
        if max_tokens == float('inf'):
            count = sum(1 for item in token_data if item["output_tokens"] >= min_tokens)
        else:
            count = sum(1 for item in token_data if min_tokens <= item["output_tokens"] < max_tokens)
        percentage = (count / len(token_data)) * 100
        bar = "█" * int(percentage / 2)
        print(f"{label:>8}: {count:>4} ({percentage:>5.1f}%) {bar}")
    
    # 效率分析
    print("\n" + "=" * 60)
    print("⚡ 效率分析")
    print("=" * 60)
    
    avg_response_time = sum(item["response_time"] for item in token_data) / len(token_data)
    tokens_per_second = total_all / sum(item["response_time"] for item in token_data)
    
    print(f"⏱️  平均响应时间: {avg_response_time:.3f}秒")
    print(f"🚀 总处理速度: {tokens_per_second:.1f} tokens/秒")
    
    # 成本估算 (假设GPT-4价格)
    print("\n" + "=" * 60)
    print("💰 成本估算 (GPT-4定价参考)")
    print("=" * 60)
    
    # GPT-4定价: 输入 $0.03/1K tokens, 输出 $0.06/1K tokens
    input_cost = (total_input / 1000) * 0.03
    output_cost = (total_output / 1000) * 0.06
    total_cost = input_cost + output_cost
    
    print(f"💵 输入成本: ${input_cost:.4f}")
    print(f"💵 输出成本: ${output_cost:.4f}")
    print(f"💵 总成本: ${total_cost:.4f}")
    print(f"💵 平均每问题成本: ${total_cost / len(token_data):.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析Full-Context实验的Token使用情况")
    parser.add_argument("--input_file", type=str, default="results/full_context_results.json",
                       help="Full-Context实验结果文件")
    
    args = parser.parse_args()
    
    analyze_token_usage(args.input_file)