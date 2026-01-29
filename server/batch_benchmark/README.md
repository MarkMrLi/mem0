# Mem0 Add 阶段 Batch LLM 调用性能验证 MVP

## 目标

验证在 mem0 的 `add` 操作中，通过优化 LLM 调用调度，利用 vLLM 的 **prefix caching** 机制带来的性能提升。

## 背景

### Mem0 Add 操作流程

每次 `add` 操作包含两次 LLM 调用：

```
┌─────────────────┐     ┌─────────────────┐
│   Extraction    │ --> │     Update      │
│ (提取 facts)    │     │ (更新 memory)   │
└─────────────────┘     └─────────────────┘
```

### 问题

原生 mem0 行为是**完全串行**的：

```
E1 → U1 → E2 → U2 → E3 → U3 → ...
```

这种模式下，共享 system prompt 的 Extraction 请求被其他类型的请求（Update）隔开，导致 vLLM 的 prefix cache 无法有效利用。

### 优化方案

通过**按阶段批量调度**，让共享 prefix 的请求连续执行：

```
[E1, E2, E3, ...] → [U1, U2, U3, ...]
```

## 测试场景设计

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           测试场景对比                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  基线 (mem0 原生行为):                                                       │
│  ┌────┐   ┌────┐   ┌────┐   ┌────┐   ┌────┐   ┌────┐                       │
│  │ E1 │──▶│ U1 │──▶│ E2 │──▶│ U2 │──▶│ E3 │──▶│ U3 │──▶ ...               │
│  └────┘   └────┘   └────┘   └────┘   └────┘   └────┘                       │
│     └── 完全串行，prefix cache 无法有效利用 ──┘                              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  优化1 (并发 HTTP 请求):                                                     │
│  ┌────┬────┬────┐         ┌────┬────┬────┐                                 │
│  │ E1 │ E2 │ E3 │ ──────▶ │ U1 │ U2 │ U3 │                                 │
│  └────┴────┴────┘         └────┴────┴────┘                                 │
│     └── 按阶段批量，vLLM 自动 batch + prefix cache ──┘                       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  优化2 (离线 LLM.generate()):                                                │
│  ┌─────────────────────┐    ┌─────────────────────┐                        │
│  │ LLM.generate(       │    │ LLM.generate(       │                        │
│  │   [E1, E2, E3, ...] │───▶│   [U1, U2, U3, ...] │                        │
│  │ )                   │    │ )                   │                        │
│  └─────────────────────┘    └─────────────────────┘                        │
│     └── 离线批量推理，最优性能 ──┘                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 测试策略

| 策略 | 模式 | 说明 |
|------|------|------|
| `mem0_native` | E1→U1→E2→U2→... | 基线，模拟 mem0 原生串行行为 |
| `concurrent_http` | [E1,E2,...] → [U1,U2,...] | 并发 HTTP 请求，按阶段批量 |
| `offline_batch` | LLM.generate() | 离线批量推理（服务器端运行） |

## 文件结构

```
server/batch_benchmark/
├── .env.example           # 配置模板
├── config.py              # 环境变量加载
├── llm_client.py          # Async OpenAI 客户端
├── token_analyzer.py      # Token 成本分析
├── strategies.py          # 测试策略实现
├── benchmark.py           # 主入口（HTTP 方式）
├── offline_batch.py       # 离线批量推理（服务器端）
├── vllm_metrics.py        # vLLM metrics 解析
└── README.md              # 本文件
```

## 快速开始

### 1. 配置

```bash
cd server/batch_benchmark
cp .env.example .env
# 编辑 .env 设置你的 vLLM 服务地址
```

### 2. 运行 HTTP 方式 Benchmark

```bash
# 运行所有策略对比
python benchmark.py --strategy all --num-requests 20

# 只运行 mem0_native 基线
python benchmark.py --strategy mem0_native --num-requests 10

# 只运行 concurrent_http 优化
python benchmark.py --strategy concurrent_http --num-requests 10

# 只测试 extraction 阶段（更好地隔离 prefix cache 效果）
python benchmark.py --mode extraction-only --num-requests 20
```

### 3. 运行离线批量 Benchmark（服务器端）

```bash
# 在 vLLM 服务器上运行
python offline_batch.py \
    --model /home/llz/model/qwen3-30b-a3b-instruct-2507 \
    --prompts ../prompts_storage.json \
    --num-requests 20 \
    --mode compare
```

## 输出示例

```
Strategy Comparison (N=20 requests)
══════════════════════════════════════════════════════════════
Strategy          │ Time (s) │ Cache Hit % │ Speedup │ Tokens
──────────────────┼──────────┼─────────────┼─────────┼────────
mem0_native       │   45.2   │    ~30%     │  1.00x  │ 120,000
concurrent_http   │   12.5   │    ~85%     │  3.6x   │ 120,000
──────────────────────────────────────────────────────────────
```

## 关键指标

1. **总执行时间**: 端到端完成所有请求的时间
2. **Cache Hit Rate**: vLLM prefix cache 命中率（从 `/metrics` 端点获取）
3. **Speedup**: 相对于基线的加速比

## vLLM Metrics

Benchmark 自动从 vLLM 的 `/metrics` 端点获取 prefix cache 统计：

```
vllm:prefix_cache_queries_total  # 查询的 token 数
vllm:prefix_cache_hits_total     # 命中的 token 数
```

Cache Hit Rate = hits / queries × 100%

## 配置选项

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM API 地址 |
| `VLLM_API_KEY` | `EMPTY` | API Key |
| `VLLM_MODEL` | - | 模型路径 |
| `VLLM_METRICS_URL` | 从 BASE_URL 推导 | Metrics 端点地址 |
| `VLLM_METRICS_ENABLED` | `true` | 是否启用 metrics 采集 |

## 依赖

```
openai>=1.0.0
httpx
python-dotenv
tiktoken  # 可选，用于 token 分析
```

离线批量模式额外需要：
```
vllm
transformers
```

## 预期结果

基于 extraction prompts 共享 ~2000 tokens 的 system prompt：

| 场景 | mem0_native Cache Hit | concurrent_http Cache Hit | 预期加速 |
|------|----------------------|---------------------------|---------|
| 10 请求 | ~20-30% | ~80-90% | 2-3x |
| 20 请求 | ~15-25% | ~85-95% | 3-4x |
| 50 请求 | ~10-20% | ~90-95% | 4-5x |

实际结果取决于：
- vLLM 版本和配置
- GPU 显存和 KV cache 大小
- 并发请求数量
- 请求到达间隔
