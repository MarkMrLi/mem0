"""
Configuration loader for batch benchmark.
Loads settings from .env file or environment variables.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Try to load dotenv, but don't fail if not installed
try:
    from dotenv import load_dotenv

    # Load .env from the same directory as this file
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"Loaded .env from: {env_path}")
    else:
        print(f"Warning: .env file not found at {env_path}")
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")


@dataclass
class LLMConfig:
    """LLM API configuration."""

    base_url: str
    api_key: str
    model: str
    timeout: int = 120


@dataclass
class BenchmarkConfig:
    """Benchmark execution configuration."""

    num_requests: int = 15
    timeout: int = 120
    prompts_file: str = "../prompts_storage.json"


@dataclass
class VLLMMetricsConfig:
    """vLLM metrics endpoint configuration."""

    metrics_url: str
    enabled: bool = True


def get_llm_config() -> LLMConfig:
    """Load LLM configuration from environment variables."""
    return LLMConfig(
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        model=os.getenv("VLLM_MODEL", "default-model"),
        timeout=int(os.getenv("BENCHMARK_TIMEOUT", "120")),
    )


def get_benchmark_config() -> BenchmarkConfig:
    """Load benchmark configuration from environment variables."""
    return BenchmarkConfig(
        num_requests=int(os.getenv("BENCHMARK_NUM_REQUESTS", "15")),
        timeout=int(os.getenv("BENCHMARK_TIMEOUT", "120")),
        prompts_file=os.getenv("PROMPTS_FILE", str(Path(__file__).parent.parent / "prompts_storage.json")),
    )


def get_vllm_metrics_config() -> VLLMMetricsConfig:
    """Load vLLM metrics configuration from environment variables."""
    metrics_url = os.getenv("VLLM_METRICS_URL", "")

    # If not explicitly set, derive from base URL
    if not metrics_url:
        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        # Remove /v1 suffix and add /metrics
        if base_url.endswith("/v1"):
            metrics_url = base_url[:-3] + "/metrics"
        else:
            metrics_url = base_url.rstrip("/") + "/metrics"

    enabled = os.getenv("VLLM_METRICS_ENABLED", "true").lower() in ("true", "1", "yes")

    return VLLMMetricsConfig(
        metrics_url=metrics_url,
        enabled=enabled,
    )


def validate_config(llm_config: LLMConfig) -> bool:
    """Validate that the configuration is complete."""
    if not llm_config.base_url:
        print("Error: VLLM_BASE_URL is not set")
        return False
    if not llm_config.model or llm_config.model == "default-model":
        print("Warning: VLLM_MODEL is not set, using default")
    return True
