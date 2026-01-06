# LongMemEval Integration Guide

## Overview

This guide explains how to use the LongMemEval integration for mem0 evaluation. LongMemEval is a comprehensive benchmark for evaluating long-term memory systems in conversational AI.

## Dataset Structure

LongMemEval datasets contain conversations with the following structure:

- `question_id`: Unique identifier for each question
- `question_type`: Type of question (temporal-reasoning, knowledge-update, multi-session, etc.)
- `question`: The question content
- `answer`: Expected answer
- `question_date`: Date of the question
- `haystack_session_ids`: List of history session IDs
- `haystack_dates`: Timestamps of history sessions
- `haystack_sessions`: Actual conversation history
- `answer_session_ids`: Evidence session IDs for evaluation

## Setup

### Prerequisites

1. **Environment Variables**: Set up your `.env` file with required API keys:
```bash
# mem0 Configuration
MEM0_BASE_URL="http://127.0.0.1:7000"  # Your mem0 server URL

# LLM Configuration  
VLLM_BASE_URL="your-vllm-base-url"     # Your LLM API URL
MODEL="gpt-4o-mini"                     # Model to use

# Optional: Embedding model
EMBEDDING_MODEL="text-embedding-3-small"
```

2. **Install Dependencies**: Ensure all required packages are installed:
```bash
pip install openai httpx jinja2 tiktoken tqdm python-dotenv
```

### Dataset Preparation

Place your LongMemEval dataset files in the `dataset/` directory:
- `longmemeval_s.json` - Small dataset (for testing)
- `longmemeval_m.json` - Medium dataset  
- `longmemeval_l.json` - Large dataset

## Usage

### Basic Usage

Run evaluation with default settings:

```bash
python run_longmemeval_eval.py --dataset dataset/longmemeval_s.json
```

### Advanced Usage

#### Custom Top-K Values

Test specific top_k values:

```bash
python run_longmemeval_eval.py --dataset dataset/longmemeval_s.json --top_k 5 10 15
```

#### Limited Testing

Test with a limited number of conversations:

```bash
python run_longmemeval_eval.py --dataset dataset/longmemeval_s.json --limit 5
```

#### Custom Output Directory

Specify where to save results:

```bash
python run_longmemeval_eval.py --dataset dataset/longmemeval_s.json --output results/my_eval
```

#### High Concurrency

Increase concurrent requests for faster processing:

```bash
python run_longmemeval_eval.py --dataset dataset/longmemeval_s.json --max_concurrent 5
```

## Evaluation Process

For each conversation in the dataset:

1. **Memory Addition Phase**
   - Adds all conversation sessions to mem0
   - Measures: Time, token consumption, throughput

2. **Search & Response Phase**  
   - Searches memories with multiple top_k values (1, 5, 10, 20, 30)
   - Generates responses using retrieved memories
   - Measures: Search time, response time, token consumption

3. **Performance Evaluation**
   - Aggregates metrics across all conversations
   - Generates comparison reports by top_k values
   - Analyzes performance by question types

## Results

### Output Structure

Results are saved in timestamped directories:

```
results/longmemeval/longmemeval_eval_20250106_143000/
├── complete_results.json              # Full evaluation results
├── performance_summary.json           # Aggregated performance metrics
├── performance_report.txt             # Human-readable report
└── intermediate_results_*.json        # Intermediate checkpoints
```

### Performance Metrics

#### Memory Addition Phase
- **total_time_ms**: Total time to add all memories
- **total_tokens**: Estimated token count
- **throughput_tokens_per_second**: Processing speed

#### Search & Response Phase (per top_k)
- **search_time_ms**: Time to retrieve memories
- **response_time_ms**: Time to generate response
- **total_time_ms**: Combined search + response time
- **token_counts**: Input/output/total tokens

### Report Analysis

The performance report provides:

1. **Overall Summary**: Aggregate statistics across all conversations
2. **Top-K Comparison**: Performance comparison across different top_k values
3. **Question Type Analysis**: Performance breakdown by question types
4. **Efficiency Metrics**: Time and token efficiency rankings

## Integration with Existing Evaluation

The LongMemEval integration works alongside the existing LOCOMO evaluation:

```bash
# Run LOCOMO evaluation (existing)
python run_experiments.py --technique_type mem0

# Run LongMemEval evaluation (new)
python run_longmemeval_eval.py --dataset dataset/longmemeval_s.json
```

## Troubleshooting

### Common Issues

1. **Connection Errors**: Ensure mem0 server is running at `MEM0_BASE_URL`
2. **API Errors**: Verify LLM API credentials in `.env`
3. **Dataset Format**: Check that dataset follows LongMemEval structure
4. **Memory Issues**: Reduce `max_concurrent` if facing memory constraints

### Performance Optimization

- **Faster Processing**: Increase `max_concurrent` (if resources allow)
- **Quick Testing**: Use `--limit` to test with fewer conversations
- **Reduced Top-K Values**: Test fewer top_k values to speed up evaluation

## API Reference

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | Required | Path to LongMemEval JSON file |
| `--output` | str | results/longmemeval | Output directory |
| `--top_k` | int[] | [1,5,10,20,30] | Top-K values to test |
| `--max_concurrent` | int | 3 | Max concurrent requests |
| `--limit` | int | None | Limit conversations (testing) |

### Python API

You can also use the components programmatically:

```python
from src.longmemeval import (
    LongMemEvalDataConverter,
    LongMemEvalMemoryAdder, 
    LongMemEvalMemorySearch,
    PerformanceEvaluator
)

# Load and convert dataset
converter = LongMemEvalDataConverter()
data = converter.load_and_convert_json("dataset/longmemeval_s.json")

# Process conversations
adder = LongMemEvalMemoryAdder()
searcher = LongMemEvalMemorySearch()
evaluator = PerformanceEvaluator()

for conversation in data:
    add_metrics = adder.add_conversation(conversation)
    search_results = searcher.search_and_respond(
        conversation["user_id"],
        conversation["query"]
    )
    evaluator.evaluate_conversation(
        conversation, add_metrics, search_results
    )
```