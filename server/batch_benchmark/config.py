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


def validate_config(llm_config: LLMConfig) -> bool:
    """Validate that the configuration is complete."""
    if not llm_config.base_url:
        print("Error: VLLM_BASE_URL is not set")
        return False
    if not llm_config.model or llm_config.model == "default-model":
        print("Warning: VLLM_MODEL is not set, using default")
    return True
