import argparse
import json

import pandas as pd

# 类别映射表
CATEGORY_MAPPING = {
    "1": "multi-hop",
    "2": "temporal", 
    "3": "open-domain",
    "4": "single-hop",
    "5": "adversarial"
}


def generate_scores(input_file="evaluation_metrics.json", output_csv=None, print_terminal=True):
    """生成评估得分，支持CSV输出和终端打印"""

    # Load the evaluation metrics data
    with open(input_file, "r") as f:
        data = json.load(f)

    # Flatten the data into a list of question items
    all_items = []
    for key in data:
        all_items.extend(data[key])

    # Convert to DataFrame
    df = pd.DataFrame(all_items)

    # 如果数据中没有category_name列，添加它
    if "category_name" not in df.columns:
        df["category_name"] = df["category"].apply(
            lambda x: CATEGORY_MAPPING.get(str(x), f"category_{x}")
        )

    # 按类别名称分组
    category_result = df.groupby("category_name").agg({
        "bleu_score": "mean",
        "f1_score": "mean",
        "llm_score": "mean"
    }).round(4)

    # Add count of questions per category
    category_result["count"] = df.groupby("category_name").size()

    # Calculate overall means
    overall_means = df.agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)

    # 输出CSV文件
    if output_csv:
        category_result.to_csv(output_csv)
        print(f"📊 得分已保存到CSV文件: {output_csv}")

        # 也可以保存完整的数据
        full_data_csv = output_csv.replace(".csv", "_full_data.csv")
        df.to_csv(full_data_csv, index=False)
        print(f"📊 完整数据已保存到CSV文件: {full_data_csv}")

    # 打印到终端
    if print_terminal:
        print("\n📊 Mean Scores Per Category:")
        print(category_result)

        print("\n📈 Overall Mean Scores:")
        print(overall_means)

    return category_result, overall_means


def main():
    parser = argparse.ArgumentParser(description="生成评估得分报告")
    parser.add_argument(
        "--input_file", type=str, default="evaluation_metrics.json",
        help="评估指标JSON文件路径"
    )
    parser.add_argument(
        "--output_csv", type=str, default="scores.csv",
        help="输出CSV文件路径"
    )
    parser.add_argument(
        "--no_print", action="store_true", default=False,
        help="不打印到终端"
    )

    args = parser.parse_args()

    generate_scores(
        input_file=args.input_file,
        output_csv=args.output_csv,
        print_terminal=not args.no_print
    )


if __name__ == "__main__":
    main()
