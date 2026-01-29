# Mem0 Add 阶段 Batch LLM 调用性能验证 MVP

## 背景

在多并发场景下，Mem0 的 `add` 操作需要频繁调用 LLM。本项目旨在验证通过 batch 合并 LLM 调用，能够带来多少性能提升和成本减少。

## 问题分析

### Mem0 Add 操作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mem0 Add 操作流程                              │
├─────────────────────────────────────────────────────────────────┤
│  Input: N 条消息 (来自 N 次 add 调用)                             │
│                                                                   │
│  ┌──────────────────┐    ┌──────────────────────────────────┐   │
│  │ LLM Call 1       │    │ LLM Call 2                       │   │
│  │ Fact Extraction  │ -> │ Memory Update Decision           │   │
│  │ (独立，可并行)     │    │ (依赖已有 memory 状态)            │   │
│  └──────────────────┘    └──────────────────────────────────┘   │
│                                                                   │
│  原始：N 次 extraction + N 次 update = 2N 次串行 LLM 调用        │
│  优化后：                                                         │
│    - 同用户：1 batch extraction + N 次串行 update                │
│    - 多用户：1 batch extraction + 1 batch update (完全并行)      │
└─────────────────────────────────────────────────────────────────┘
```

### 数据特征 (来自 prompts_storage.json)

| 阶段 | 数量 | System Prompt | 特征 |
|------|------|---------------|------|
| Extraction | 100 | 1 个共享 (~2000 tokens) | 仅 user content 不同，可以高效 batch |
| Update | 78 | 无 system prompt | 包含动态 memory 状态，依赖前序结果 |

## 优化策略对比

```
┌────────────────────────────────────────────────────────────────────────┐
│ 策略对比                                                                 │
├─────────────┬──────────────────────┬───────────────────────────────────┤
│ 策略        │ Extraction           │ Update                            │
├─────────────┼──────────────────────┼───────────────────────────────────┤
│ Sequential  │ N 次独立请求          │ N 次独立请求 (每次依赖前一次结果)  │
│             │ ↓ N × system prompt  │ ↓ 必须串行                         │
├─────────────┼──────────────────────┼───────────────────────────────────┤
│ Concurrent  │ N 次并发请求          │ 单用户: 串行 / 多用户: 并发        │
│ (asyncio)   │ (vLLM 自动 batch)    │ ↓ vLLM 自动 batch                 │
├─────────────┼──────────────────────┼───────────────────────────────────┤
│ Prompt      │ 1 次请求             │ 保持原样                           │
│ Batching    │ 合并多个 user input  │ ↓ 无法合并 (每个有不同 context)    │
│             │ ↓ 1 × system prompt  │                                    │
└─────────────┴──────────────────────┴───────────────────────────────────┘
```

## MVP 设计

### 文件结构

```
server/batch_benchmark/
├── .env.example           # 配置模板
├── config.py              # 加载 .env 配置
├── llm_client.py          # Async OpenAI 客户端
├── token_analyzer.py      # Token 成本分析 (使用 tiktoken)
├── prompt_batcher.py      # Prompt 合并逻辑
├── strategies.py          # 测试策略实现
├── benchmark.py           # 主入口
└── README.md              # 本文件
```

### 测试策略

| 策略名称 | Extraction | Update | 适用场景 |
|---------|------------|--------|---------|
| `sequential` | 串行 N 次 | 串行 N 次 | 基线 |
| `concurrent_extraction` | 并发 N 次 | 串行 N 次 | 单用户 |
| `batched_extraction` | 合并 1 次 | 串行 N 次 | 单用户 (省 token) |
| `full_concurrent` | 并发 N 次 | 并发 N 次 | 多用户 |
| `full_batched` | 合并 1 次 | 并发/合并 | 多用户 (省 token) |

### Prompt 合并策略

#### 原始格式 (N 个独立请求)

```json
[
    {"role": "system", "content": "<2000 tokens system prompt>"},
    {"role": "user", "content": "Input: <conversation 1>"}
]
[
    {"role": "system", "content": "<2000 tokens system prompt>"},
    {"role": "user", "content": "Input: <conversation 2>"}
]
```

#### 合并后格式 (1 个请求)

```json
[
    {"role": "system", "content": "<2000 tokens system prompt + 批量处理指令>"},
    {"role": "user", "content": "Process the following items:\n\n[Item 1]:\n<conversation 1>\n\n[Item 2]:\n<conversation 2>"}
]
```

#### 输出格式

```json
{
    "results": [
        {"item_id": 1, "facts": ["fact1", "fact2"]},
        {"item_id": 2, "facts": ["fact3"]}
    ]
}
```

## 测试流程

### Phase 1: Token 分析 (无需调用 LLM)

- 计算 sequential 总 token 数
- 计算 batched 总 token 数
- 输出节省比例

### Phase 2: 模拟测试 (验证逻辑正确性)

- 使用固定延迟模拟 LLM 调用
- 验证 prompt 合并逻辑
- 验证结果解析逻辑

### Phase 3: 实际调用测试 (测性能)

- Sequential baseline
- Concurrent (vLLM auto-batch)
- Prompt batching
- 生成对比报告

## 预期成本节省

基于 `prompts_storage.json` 的数据 (15 个请求)：

| 项目 | Sequential | Batched | 节省 |
|------|-----------|---------|------|
| System Prompt | 15 × ~2000 = 30,000 | 1 × ~2100 = 2,100 | **93%** |
| User Content | 15 × ~100 = 1,500 | 1 × ~1,600 = 1,600 | -7% |
| **Total Input** | **31,500** | **3,700** | **88%** |

## 预期结果

### 场景 1: 单用户 (10 adds, sequential update phase)

| Strategy | Time (s) | Speedup | Calls |
|----------|----------|---------|-------|
| Sequential | 25.3 | 1.00x | 20 |
| Concurrent Extraction | 14.2 | 1.78x | 11 |
| Batched Extraction | 12.5 | 2.02x | 11 |

### 场景 2: 多用户 (10 adds, parallel update phase)

| Strategy | Time (s) | Speedup | Calls |
|----------|----------|---------|-------|
| Sequential | 24.8 | 1.00x | 20 |
| Full Concurrent | 6.1 | 4.07x | 2 |
| Full Batched | 5.2 | 4.77x | 2 |

## 配置

复制 `.env.example` 为 `.env` 并配置：

```bash
cp .env.example .env
```

配置项：

```
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL=your-model-name
```

## 运行

```bash
# 仅运行 Token 分析 (不需要 LLM)
python benchmark.py --phase token

# 运行模拟测试
python benchmark.py --phase mock

# 运行完整测试
python benchmark.py --phase full

# 指定测试规模
python benchmark.py --num-requests 15
```

## 关键技术点

1. **Prompt 合并格式**：需要修改 system prompt 添加批量处理指令，并设计 item 分隔符
2. **结果解析**：从单个 JSON 响应中提取多个结果
3. **错误处理**：单个 item 失败不影响其他 item
4. **Update 阶段依赖**：单用户场景下必须串行

## 依赖

```
openai>=1.0.0
tiktoken
python-dotenv
asyncio
```
