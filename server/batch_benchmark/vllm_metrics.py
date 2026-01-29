"""
vLLM Metrics Parser

Fetches and parses Prometheus metrics from vLLM's /metrics endpoint.
Focuses on prefix cache hit rate for benchmark comparison.

Key metrics:
- vllm:prefix_cache_queries_total: Total tokens queried from cache
- vllm:prefix_cache_hits_total: Total tokens that were cache hits
- vllm:prompt_tokens_total: Total prompt tokens processed
- vllm:generation_tokens_total: Total generation tokens processed
"""

import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Dict, Optional

# Try to import aiohttp for async operations
try:
    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


@dataclass
class VLLMMetrics:
    """Snapshot of vLLM metrics at a point in time."""

    # Prefix cache metrics
    prefix_cache_queries: float = 0.0
    prefix_cache_hits: float = 0.0

    # Token metrics
    prompt_tokens: float = 0.0
    generation_tokens: float = 0.0

    # Request metrics
    num_requests_running: float = 0.0
    num_requests_waiting: float = 0.0

    # KV cache usage
    kv_cache_usage_pct: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Calculate prefix cache hit rate as percentage."""
        if self.prefix_cache_queries == 0:
            return 0.0
        return (self.prefix_cache_hits / self.prefix_cache_queries) * 100

    def __sub__(self, other: "VLLMMetrics") -> "VLLMMetrics":
        """Calculate delta between two metric snapshots."""
        return VLLMMetrics(
            prefix_cache_queries=self.prefix_cache_queries - other.prefix_cache_queries,
            prefix_cache_hits=self.prefix_cache_hits - other.prefix_cache_hits,
            prompt_tokens=self.prompt_tokens - other.prompt_tokens,
            generation_tokens=self.generation_tokens - other.generation_tokens,
            num_requests_running=self.num_requests_running,  # Keep current value
            num_requests_waiting=self.num_requests_waiting,  # Keep current value
            kv_cache_usage_pct=self.kv_cache_usage_pct,  # Keep current value
        )

    def summary(self, title: str = "vLLM Metrics") -> str:
        """Generate a human-readable summary."""
        return f"""
{title}
{"=" * 50}
Prefix Cache:
  Queries:      {self.prefix_cache_queries:,.0f} tokens
  Hits:         {self.prefix_cache_hits:,.0f} tokens
  Hit Rate:     {self.cache_hit_rate:.1f}%

Tokens Processed:
  Prompt:       {self.prompt_tokens:,.0f}
  Generation:   {self.generation_tokens:,.0f}

Engine State:
  KV Cache:     {self.kv_cache_usage_pct * 100:.1f}% used
  Running:      {self.num_requests_running:.0f} requests
  Waiting:      {self.num_requests_waiting:.0f} requests
{"=" * 50}
"""


def parse_prometheus_metrics(text: str) -> Dict[str, float]:
    """
    Parse Prometheus text format metrics.

    Handles lines like:
    vllm:prefix_cache_hits_total{engine="0",model_name="..."} 2.781644112e+09
    """
    metrics = {}

    # Pattern to match metric lines (ignoring comments and type declarations)
    # Matches: metric_name{labels} value or metric_name value
    pattern = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\{?[^}]*\}?\s+([0-9eE.+-]+)$")

    for line in text.strip().split("\n"):
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith("#"):
            continue

        match = pattern.match(line)
        if match:
            metric_name = match.group(1)
            try:
                value = float(match.group(2))
                # Store the first occurrence of each metric name
                if metric_name not in metrics:
                    metrics[metric_name] = value
            except ValueError:
                continue

    return metrics


def metrics_from_dict(data: Dict[str, float]) -> VLLMMetrics:
    """Create VLLMMetrics from parsed Prometheus metrics dict."""
    return VLLMMetrics(
        prefix_cache_queries=data.get("vllm:prefix_cache_queries_total", 0.0),
        prefix_cache_hits=data.get("vllm:prefix_cache_hits_total", 0.0),
        prompt_tokens=data.get("vllm:prompt_tokens_total", 0.0),
        generation_tokens=data.get("vllm:generation_tokens_total", 0.0),
        num_requests_running=data.get("vllm:num_requests_running", 0.0),
        num_requests_waiting=data.get("vllm:num_requests_waiting", 0.0),
        kv_cache_usage_pct=data.get("vllm:kv_cache_usage_perc", 0.0),
    )


async def fetch_metrics(metrics_url: str, timeout: float = 10.0) -> Optional[VLLMMetrics]:
    """
    Fetch and parse metrics from vLLM's /metrics endpoint.

    Args:
        metrics_url: Full URL to the metrics endpoint (e.g., http://localhost:8887/metrics)
        timeout: Request timeout in seconds

    Returns:
        VLLMMetrics object or None if fetch failed
    """
    if HAS_AIOHTTP:
        try:
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_obj) as session:
                async with session.get(metrics_url) as response:
                    response.raise_for_status()
                    text = await response.text()
                    raw_metrics = parse_prometheus_metrics(text)
                    return metrics_from_dict(raw_metrics)
        except aiohttp.ClientError as e:
            print(f"Warning: Failed to fetch metrics from {metrics_url}: {e}")
            return None
        except Exception as e:
            print(f"Warning: Error parsing metrics: {e}")
            return None
    else:
        # Fallback to sync version wrapped in async
        return fetch_metrics_sync(metrics_url, timeout)


def fetch_metrics_sync(metrics_url: str, timeout: float = 10.0) -> Optional[VLLMMetrics]:
    """
    Synchronous version of fetch_metrics using urllib.

    Works without any external dependencies.
    """
    try:
        req = urllib.request.Request(metrics_url)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            text = response.read().decode("utf-8")
            raw_metrics = parse_prometheus_metrics(text)
            return metrics_from_dict(raw_metrics)

    except urllib.error.HTTPError as e:
        print(f"Warning: Failed to fetch metrics from {metrics_url}: HTTP {e.code}")
        return None
    except urllib.error.URLError as e:
        print(f"Warning: Failed to connect to {metrics_url}: {e.reason}")
        return None
    except Exception as e:
        print(f"Warning: Error parsing metrics: {e}")
        return None


class MetricsTracker:
    """
    Track metrics before and after a benchmark run.

    Usage:
        tracker = MetricsTracker(metrics_url)
        await tracker.start()
        # ... run benchmark ...
        await tracker.stop()
        print(tracker.summary())
    """

    def __init__(self, metrics_url: str, enabled: bool = True):
        self.metrics_url = metrics_url
        self.enabled = enabled
        self.before: Optional[VLLMMetrics] = None
        self.after: Optional[VLLMMetrics] = None
        self._delta: Optional[VLLMMetrics] = None

    async def start(self) -> Optional[VLLMMetrics]:
        """Capture metrics before benchmark."""
        if not self.enabled:
            return None
        self.before = await fetch_metrics(self.metrics_url)
        return self.before

    async def stop(self) -> Optional[VLLMMetrics]:
        """Capture metrics after benchmark and calculate delta."""
        if not self.enabled:
            return None
        self.after = await fetch_metrics(self.metrics_url)
        if self.before and self.after:
            self._delta = self.after - self.before
        return self.after

    @property
    def delta(self) -> Optional[VLLMMetrics]:
        """Get the delta between before and after metrics."""
        return self._delta

    def summary(self) -> str:
        """Generate summary of the benchmark run's metrics impact."""
        if not self.enabled:
            return "Metrics tracking disabled"

        if not self.delta:
            return "No metrics captured (start/stop not called or fetch failed)"

        return self.delta.summary("Benchmark Run Metrics Delta")

    def get_cache_hit_rate(self) -> Optional[float]:
        """Get the cache hit rate for this benchmark run."""
        if self.delta:
            return self.delta.cache_hit_rate
        return None


# Quick test when run directly
if __name__ == "__main__":
    import asyncio
    import sys

    async def main():
        url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8887/metrics"
        print(f"Fetching metrics from: {url}")

        metrics = await fetch_metrics(url)
        if metrics:
            print(metrics.summary())
        else:
            print("Failed to fetch metrics")

    asyncio.run(main())
