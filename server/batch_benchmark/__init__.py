"""
Mem0 Add Batch Benchmark

A benchmark tool for measuring the impact of vLLM prefix caching
when batching LLM calls by phase in the mem0 add operation.

Modules:
- config: Configuration loading from .env
- llm_client: Async OpenAI-compatible client
- token_analyzer: Token cost analysis
- strategies: Benchmark strategy implementations
- vllm_metrics: vLLM metrics parsing
- benchmark: Main entry point (HTTP)
- offline_batch: Offline batch benchmark (server-side)
"""

from .config import (
    get_llm_config,
    get_benchmark_config,
    get_vllm_metrics_config,
    LLMConfig,
    BenchmarkConfig,
    VLLMMetricsConfig,
)
from .llm_client import LLMClient, LLMResponse
from .token_analyzer import (
    analyze_extraction_prompts,
    analyze_update_prompts,
    run_token_analysis,
    TokenAnalysis,
)
from .strategies import (
    Mem0NativeStrategy,
    ConcurrentHttpStrategy,
    ExtractionOnlyStrategy,
    StrategyResult,
    compare_results,
)
from .vllm_metrics import (
    VLLMMetrics,
    MetricsTracker,
    fetch_metrics,
    fetch_metrics_sync,
)

__all__ = [
    # Config
    "get_llm_config",
    "get_benchmark_config",
    "get_vllm_metrics_config",
    "LLMConfig",
    "BenchmarkConfig",
    "VLLMMetricsConfig",
    # Client
    "LLMClient",
    "LLMResponse",
    # Token Analysis
    "analyze_extraction_prompts",
    "analyze_update_prompts",
    "run_token_analysis",
    "TokenAnalysis",
    # Strategies
    "Mem0NativeStrategy",
    "ConcurrentHttpStrategy",
    "ExtractionOnlyStrategy",
    "StrategyResult",
    "compare_results",
    # Metrics
    "VLLMMetrics",
    "MetricsTracker",
    "fetch_metrics",
    "fetch_metrics_sync",
]
