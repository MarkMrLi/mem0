import time
import psutil
import GPUtil
import json
import cProfile
import pstats
import io
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

process = psutil.Process()
results = {}

def measure_with_cprofile(fn, name, *args, **kwargs):
    """使用 cProfile 进行性能分析 + 资源监控"""
    
    # 资源监控
    start_cpu = process.cpu_percent(interval=None)
    start_mem = process.memory_info().rss
    start_gpu = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else None

    # cProfile 分析
    profiler = cProfile.Profile()
    profiler.enable()
    
    t0 = time.perf_counter()
    output = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    
    profiler.disable()

    # 获取 profiling 结果
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # 打印前20个最耗时的函数
    
    profiling_results = s.getvalue()

    # 资源监控结束
    end_cpu = process.cpu_percent(interval=None)
    end_mem = process.memory_info().rss
    end_gpu = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else None

    results[name] = {
        "elapsed_sec": elapsed,
        "cpu_percent_delta": end_cpu - start_cpu,
        "ram_mb_delta": (end_mem - start_mem) / 1024 / 1024,
        "gpu_mb_delta": (end_gpu - start_gpu) if start_gpu and end_gpu else None,
        "profiling_stats": profiling_results
    }

    return output

def measure_simple(fn, name, *args, **kwargs):
    """简化的测量函数（只测量时间和资源，不包含 cProfile）"""
    start_cpu = process.cpu_percent(interval=None)
    start_mem = process.memory_info().rss
    start_gpu = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else None

    t0 = time.perf_counter()
    output = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0

    end_cpu = process.cpu_percent(interval=None)
    end_mem = process.memory_info().rss
    end_gpu = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else None

    results[name] = {
        "elapsed_sec": elapsed,
        "cpu_percent_delta": end_cpu - start_cpu,
        "ram_mb_delta": (end_mem - start_mem) / 1024 / 1024,
        "gpu_mb_delta": (end_gpu - start_gpu) if start_gpu and end_gpu else None,
    }

    return output

# ---- 初始化 Memory ----
print("初始化 Memory...")
m = measure_with_cprofile(lambda: Memory.from_config(config), "init_memory")

# ---- CRUD 测试 ----
print("添加记忆...")
measure_with_cprofile(lambda: m.add("I am an engineer", user_id="u1"), "add_memory")

print("搜索记忆...")
memories = measure_with_cprofile(lambda: m.search("What do you know about me?", user_id="u1"), "search_memory")
print(memories)
id = memories["results"][0]["id"]
print("-------------------------------------------")

print("更新记忆...")
measure_with_cprofile(lambda: m.update(memory_id=id, data="I am an software engineer"), "update_memory")

print("搜索更新后的记忆...")
memories = measure_with_cprofile(lambda: m.search("What do you know about me?", user_id="u1"), "search_new_memory")
print(memories)
print("-------------------------------------------")

print("删除记忆...")
id = memories["results"][0]["id"]
measure_with_cprofile(lambda: m.delete(id), "delete_memory")

print("添加新记忆...")
measure_with_cprofile(lambda: m.add("I like playing basketball", user_id="u1"), "add_new_memory")

print("搜索新记忆...")
memories = measure_with_cprofile(lambda: m.search("What do you know about me?", user_id="u1"), "search_new_memory")
print(memories)
print("-------------------------------------------")

# ---- 保存报告 ----
with open("perf_report.json", "w") as f:
    json.dump(results, f, indent=4)

# ---- 保存详细的 profiling 报告 ----
print("\n=== cProfile 详细分析报告 ===")
for operation_name, data in results.items():
    print(f"\n--- {operation_name} ---")
    print(f"执行时间: {data['elapsed_sec']:.4f} 秒")
    print(f"内存变化: {data['ram_mb_delta']:.2f} MB")
    print(f"CPU 变化: {data['cpu_percent_delta']:.2f}%")
    if data.get('gpu_mb_delta') is not None:
        print(f"GPU 内存变化: {data['gpu_mb_delta']:.2f} MB")
    
    if 'profiling_stats' in data:
        print("函数调用分析:")
        print(data['profiling_stats'])

# 保存完整的 profiling 数据到单独文件
with open("cprofile_detailed.txt", "w") as f:
    for operation_name, data in results.items():
        f.write(f"\n{'='*50}\n")
        f.write(f"操作: {operation_name}\n")
        f.write(f"{'='*50}\n")
        f.write(f"执行时间: {data['elapsed_sec']:.4f} 秒\n")
        f.write(f"内存变化: {data['ram_mb_delta']:.2f} MB\n")
        f.write(f"CPU 变化: {data['cpu_percent_delta']:.2f}%\n")
        if data.get('gpu_mb_delta') is not None:
            f.write(f"GPU 内存变化: {data['gpu_mb_delta']:.2f} MB\n")
        
        if 'profiling_stats' in data:
            f.write("\n函数调用分析:\n")
            f.write(data['profiling_stats'])

print("Performance report saved → perf_report.json")
print("Detailed cProfile report saved → cprofile_detailed.txt")