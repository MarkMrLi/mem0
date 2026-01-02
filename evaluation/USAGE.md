# Mem0 评估系统使用指南

优化后的mem0评估系统，简化和自动化了整个评估流程。

## 🎯 主要改进

1. **✅ 使用HTTP请求代替memclient**: 现在直接使用requests库与本地mem0实例通信
2. **🔄 整合流程**: 将search、evals、generate_scores三步合并为一个流程
3. **📊 CSV输出**: 自动生成CSV格式的得分报告，方便后续分析和制图
4. **📁 改进目录结构**: 每个实验都有独立的目录，便于管理和对比

## 🚀 快速开始

### 1. 配置环境变量

在项目根目录创建 `.env` 文件：

```bash
# Mem0本地实例配置
MEM0_BASE_URL=http://127.0.0.1:7000

# OpenAI配置（用于生成答案和评估）
OPENAI_API_KEY=your-openai-api-key
MODEL=gpt-4o-mini
```

### 2. 启动本地Mem0实例

确保你的mem0实例在 `http://127.0.0.1:7000` 运行。

参考示例：`demo_client.py`

### 3. 运行评估

#### 方式一：使用新的整合流程（推荐）

```bash
# 基础评估 - 一步到位！
make run-mem0-eval

# 使用图谱搜索
make run-mem0-eval-graph
```

这将自动完成：
- 🔍 搜索记忆并生成答案
- 📊 计算评估指标（BLEU, F1, LLM judge）
- 📈 生成CSV得分报告
- 📁 创建时间戳命名的实验目录

#### 方式二：传统分步执行（向后兼容）

```bash
# 1. 添加记忆
make run-mem0-add

# 2. 搜索并生成答案
make run-mem0-search

# 3. 评估结果
python evals.py --input_file results/mem0_results_top_30_filter_False_graph_False.json --output_file evaluation_metrics.json

# 4. 生成得分（现在支持CSV）
python generate_scores.py --output_csv my_scores.csv
```

## 📂 实验结果组织

每次运行都会在 `results/` 目录下创建一个新的实验目录：

```
results/
└── mem0_eval_top30_filterFalse_graphFalse_20250101_143022/
    ├── metadata.json              # 实验配置和总体得分
    ├── search_results.json        # 详细搜索结果
    ├── evaluation_metrics.json    # 评估指标
    └── scores.csv                 # CSV格式得分报告
```

## 🔧 自定义参数

使用Python脚本直接运行以获得更多控制：

```bash
python run_mem0_evaluation.py \
  --data_file dataset/locomo10.json \
  --output_folder results/ \
  --top_k 50 \
  --filter_memories \
  --is_graph \
  --max_workers 20
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_file` | `dataset/locomo10.json` | 数据集文件路径 |
| `--output_folder` | `results/` | 结果输出目录 |
| `--top_k` | `30` | 检索的记忆数量 |
| `--filter_memories` | `False` | 是否过滤记忆 |
| `--is_graph` | `False` | 是否使用图谱搜索 |
| `--max_workers` | `10` | 并发工作线程数 |

## 📊 分析结果

### 🏷️ 类别映射说明

系统现在使用直观的类别名称而不是数字：

| 数字 | 类别名称 | 说明 |
|------|----------|------|
| 1 | `multi-hop` | 多跳推理问题 |
| 2 | `temporal` | 时间相关问馗 |
| 3 | `open-domain` | 开放域问题 |
| 4 | `single-hop` | 单跳问题 |
| 5 | `adversarial` | 对抗性问题（评估中跳过）**

### CSV文件使用

生成的CSV文件现在直接使用类别名称而不是数字，更加直观：

**新的CSV格式**：
```csv
category_name,bleu_score,f1_score,llm_score,count
multi-hop,0.2345,0.3456,0.7890,15
temporal,0.3456,0.4567,0.8901,12
open-domain,0.4567,0.5678,0.9012,8
single-hop,0.5678,0.6789,0.9123,20
```

生成的CSV文件可以直接在Excel、Google Sheets中打开，或用Python分析：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取实验结果
df = pd.read_csv("results/mem0_eval_top30_filterFalse_graphFalse_20250101_143022/scores.csv", index_col=0)

# 显示得分
print(df)

# 生成对比图 - 现在有意义的标签
df[['bleu_score', 'f1_score', 'llm_score']].plot(kind='bar')
plt.ylabel('Score')
plt.title('Mem0 Evaluation Results by Question Type')
plt.tight_layout()
plt.savefig('evaluation_results.png')
plt.show()
```

### 对比多个实验

```python
import pandas as pd
import os

# 读取所有实验
results = []
for exp_dir in os.listdir('results'):
    if exp_dir.startswith('mem0_eval_'):
        csv_file = f'results/{exp_dir}/scores.csv'
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, index_col=0)
            df['experiment'] = exp_dir
            results.append(df)

# 合并对比
all_results = pd.concat(results)
print(all_results)
```

## 🛠️ 技术改进详解

### 1. HTTP请求代替memclient

**之前**: 使用 `MemoryClient` 需要API密钥和组织ID
```python
from mem0 import MemoryClient
client = MemoryClient(api_key=..., org_id=..., project_id=...)
```

**现在**: 直接使用requests与本地实例通信
```python
import requests
response = requests.post(f"{base_url}/memories", json=data)
```

### 2. 整合的评估流程

**之前**: 需要手动运行多个命令
```bash
make run-mem0-add
make run-mem0-search
python evals.py --input_file ... --output_file ...
python generate_scores.py
```

**现在**: 一个命令完成所有步骤
```bash
make run-mem0-eval
```

### 3. CSV输出

**之前**: 得分只打印到终端
```
Mean Scores Per Category:
         bleu_score  f1_score  llm_score
category
1           0.1234    0.2345     0.3456
```

**现在**: 自动保存CSV，方便制图
```csv
category,bleu_score,f1_score,llm_score,count
1,0.1234,0.2345,0.3456,10
```

## 📝 环境要求

确保安装了所需的依赖：

```bash
pip install requests python-dotenv openai pandas tqdm jinja2
```

## 🐛 故障排除

### 问题1: 连接本地mem0实例失败

**解决方案**: 确保mem0实例正在运行
```bash
# 检查环境变量
echo $MEM0_BASE_URL

# 测试连接
curl http://127.0.0.1:7000/health
```

### 问题2: 找不到数据文件

**解决方案**: 确保数据文件在正确位置
```bash
ls dataset/locomo10.json
```

### 问题3: OpenAI API错误

**解决方案**: 检查API密钥配置
```bash
# 检查.env文件
cat .env | grep OPENAI_API_KEY
```

## 📚 更多资源

- [完整README](README.md)
- [结果目录说明](results/README.md)
- [原始论文](https://arxiv.org/abs/2504.19413)