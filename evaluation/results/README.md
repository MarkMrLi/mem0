# 评估结果目录组织

这个目录包含了mem0评估系统的所有实验结果。

## 📁 目录结构

```
results/
├── README.md                           # 本文件
├── mem0_eval_top30_filterFalse_graphFalse_20250101_120000/  # 具体实验目录
│   ├── metadata.json                   # 实验元数据
│   ├── search_results.json             # 搜索结果
│   ├── evaluation_metrics.json         # 评估指标
│   └── scores.csv                      # 得分报告
└── mem0_eval_top30_filterFalse_graphTrue_20250101_130000/   # 另一个实验
    ├── metadata.json
    ├── search_results.json
    ├── evaluation_metrics.json
    └── scores.csv
```

## 🗂️ 实验目录命名规则

每个实验目录按以下格式命名：
```
mem0_eval_top{k}_filter{filter_memories}_graph{is_graph}_{timestamp}
```

- `top{k}`: 检索的记忆数量
- `filter{filter_memories}`: 是否过滤记忆 (True/False)
- `graph{is_graph}`: 是否使用图谱搜索 (True/False)
- `{timestamp}`: 实验开始时间 (YYYYMMDD_HHMMSS)

## 📄 文件说明

### metadata.json
包含实验的完整元数据：
```json
{
  "experiment_name": "实验名称",
  "timestamp": "时间戳",
  "parameters": {
    "data_file": "使用的数据文件",
    "top_k": 30,
    "filter_memories": false,
    "is_graph": false,
    "max_workers": 10
  },
  "files": {
    "search_results": "搜索结果文件路径",
    "evaluation_metrics": "评估指标文件路径",
    "scores_csv": "得分CSV文件路径"
  },
  "overall_scores": {
    "bleu_score": 0.xxxx,
    "f1_score": 0.xxxx,
    "llm_score": 0.xxxx
  }
}
```

### search_results.json
包含每个问题的搜索和回答结果，用于后续分析。

### evaluation_metrics.json
包含详细的评估指标，包括BLEU、F1和LLM得分。

### scores.csv
CSV格式的得分报告，方便生成图表：
- 按类别分组的平均得分
- 每个类别的问题数量
- 总体平均得分

## 🚀 使用方法

### 运行新的完整评估流程
```bash
# 基础评估
make run-mem0-eval

# 使用图谱搜索
make run-mem0-eval-graph
```

### 自定义参数
```bash
python run_mem0_evaluation.py \
  --data_file dataset/locomo10.json \
  --output_folder results/ \
  --top_k 30 \
  --filter_memories \
  --is_graph \
  --max_workers 10
```

## 📊 数据分析

所有实验结果都以CSV格式保存在各自的实验目录中，可以轻松导入到Excel、Google Sheets或数据分析工具（如pandas）中生成图表。

示例：
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取得分数据
df = pd.read_csv("results/mem0_eval_top30_filterFalse_graphFalse_20250101_120000/scores.csv", index_col=0)

# 生成图表
df.plot(kind='bar', y=['bleu_score', 'f1_score', 'llm_score'])
plt.savefig('comparison.png')
```

## 🧹 清理旧结果

要清理旧的实验结果：
```bash
# 删除所有实验结果
rm -rf results/mem0_eval_*

# 删除特定实验
rm -rf results/mem0_eval_top30_filterFalse_graphFalse_20250101_120000
```