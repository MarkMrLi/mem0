# Full-Context实验使用说明

## 概述
Full-Context实验旨在测试将locomo对话中所有sessions连接成完整上下文的性能表现。

## 实验设计
- **核心思路**: 将每个conversation中的所有sessions连接成完整的prompt上下文
- **处理方式**: 使用完整上下文回答所有qa问题
- **输出格式**: 完全兼容现有evals.py和generate_scores.py的评估流程

## 文件说明

### 新增文件
- `evaluation/src/full_context.py`: Full-Context处理器核心实现
- `evaluation/run_full_context_eval.py`: 一键运行实验和评估的便捷脚本

### 修改文件
- `evaluation/src/utils.py`: 添加"full_context"到TECHNIQUES列表
- `evaluation/run_experiments.py`: 集成full_context处理逻辑

## 使用方法

### 方法1: 一键运行完整流程（推荐）
```bash
cd /home/marklee/repo/mem0/evaluation
python run_full_context_eval.py --data_file dataset/locomo10.json --output_folder results/
```

### 方法2: 分步运行
```bash
# Step 1: 运行full-context实验
cd /home/marklee/repo/mem0/evaluation
python run_experiments.py --technique_type full_context --output_folder results/

# Step 2: 评估结果
python evals.py --input_file results/full_context_results.json --output_file evaluation_metrics.json

# Step 3: 生成评分报告
python generate_scores.py --input_file evaluation_metrics.json --output_csv scores.csv
```

## 参数说明

### run_full_context_eval.py参数
- `--data_file`: Locomo数据文件路径 (默认: dataset/locomo10.json)
- `--output_folder`: 输出目录 (默认: results/)
- `--max_workers`: 评估时的最大工作线程数 (默认: 10)

### run_experiments.py参数
- `--technique_type`: 实验类型，设置为"full_context"
- `--output_folder`: 输出目录

## 输出结果

实验完成后会生成以下文件：
```
results/
└── full_context_eval_<timestamp>/
    ├── full_context_results.json       # Full-Context处理结果
    ├── evaluation_metrics.json         # 评估指标
    ├── scores.csv                      # 评分报告
    └── scores_full_data.csv            # 完整评分数据
```

## 兼容性说明

该实验完全兼容现有的评估框架：
- ✅ 复用 `evaluation/evals.py` 进行评估
- ✅ 复用 `evaluation/generate_scores.py` 生成评分
- ✅ 输出格式与现有实验（mem0、rag等）一致
- ✅ 支持类别映射和评分计算

## 核心实现细节

### Full-Context处理器
- **上下文构建**: 遍历所有sessions，按时间顺序组织完整对话历史
- **回答生成**: 基于完整上下文使用LLM回答问题
- **格式标准化**: 输出格式与现有系统完全兼容

### 评估流程
- **评估指标**: BLEU、F1、LLM Judge评分
- **类别支持**: 支持multi-hop、temporal、open-domain、single-hop等类别
- **并行处理**: 支持多线程评估提高效率

## 预期优势
1. **完整上下文**: 利用所有对话信息，避免信息丢失
2. **性能基准**: 作为理论上限，与其他方法比较
3. **简单直观**: 实现简单，易于理解和调试
4. **完全兼容**: 无需修改现有评估框架

## 注意事项
1. 确保已设置环境变量（MODEL、API密钥等）
2. Full-context方法可能消耗较多tokens
3. 对于长对话，可能遇到上下文长度限制
4. 建议先用小数据集测试流程