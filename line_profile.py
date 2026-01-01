import time
import json
from line_profiler import LineProfiler
from mem0 import Memory

# ---- 配置 ----
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {"embedding_model_dims": 768},
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.1:latest",
            "temperature": 0,
            "max_tokens": 2000,
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            "ollama_base_url": "http://localhost:11434",
        },
    },
}

results = {}

def measure_with_line_profiler(fn, name, *args, **kwargs):
    """使用 line_profiler 进行逐行性能分析"""
    
    # 创建分析器
    profiler = LineProfiler()
    
    # 包装函数进行分析
    profiled_fn = profiler(fn)
    
    # 执行函数
    t0 = time.perf_counter()
    output = profiled_fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0

    # 获取分析结果
    import io
    stream = io.StringIO()
    profiler.print_stats(stream=stream)
    profiling_results = stream.getvalue()

    results[name] = {
        "elapsed_sec": elapsed,
        "profiling_stats": profiling_results
    }

    return output

# ---- 初始化 Memory ----
print("初始化 Memory...")
m = Memory.from_config(config)

# ---- CRUD 测试 ----
print("添加记忆...")
measure_with_line_profiler(lambda: m.add("I am an engineer", user_id="u1"), "add_memory")

print("搜索记忆...")
memories = measure_with_line_profiler(lambda: m.search("What do you know about me?", user_id="u1"), "search_memory")
print(f"找到 {len(memories['results'])} 条记忆")

if memories["results"]:
    id = memories["results"][0]["id"]
    print("-------------------------------------------")

    print("更新记忆...")
    measure_with_line_profiler(lambda: m.update(memory_id=id, data="I am a software engineer"), "update_memory")

    print("搜索更新后的记忆...")
    memories = measure_with_line_profiler(lambda: m.search("What do you know about me?", user_id="u1"), "search_updated_memory")
    print(f"找到 {len(memories['results'])} 条记忆")
    print("-------------------------------------------")

    print("删除记忆...")
    id = memories["results"][0]["id"]
    measure_with_line_profiler(lambda: m.delete(id), "delete_memory")

    print("添加新记忆...")
    measure_with_line_profiler(lambda: m.add("I like playing basketball", user_id="u1"), "add_new_memory")

    print("搜索新记忆...")
    memories = measure_with_line_profiler(lambda: m.search("What do you know about me?", user_id="u1"), "search_new_memory")
    print(f"找到 {len(memories['results'])} 条记忆")
    print("-------------------------------------------")

# ---- 保存报告 ----
with open("line_profiler_report.json", "w") as f:
    json.dump(results, f, indent=4)

# ---- 打印详细分析报告 ----
print("\n=== Line Profiler 详细分析报告 ===")
for operation_name, data in results.items():
    print(f"\n--- {operation_name} ---")
    print(f"执行时间: {data['elapsed_sec']:.4f} 秒")
    print("逐行分析结果:")
    print(data['profiling_stats'])
    print("=" * 80)

print("\nLine Profiler 报告已保存: line_profiler_report.json")