"""
Benchmark strategies for comparing different LLM call scheduling patterns.

Implements two main strategies:
1. Mem0NativeStrategy: E1→U1→E2→U2→... (sequential, simulates mem0 native behavior)
2. ConcurrentHttpStrategy: [E1,E2,...] concurrent → [U1,U2,...] concurrent

The goal is to measure the impact of vLLM's prefix caching when requests
with shared prefixes are batched together.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Support both direct run and package import
try:
    from .llm_client import LLMClient, LLMResponse
except ImportError:
    from llm_client import LLMClient, LLMResponse


@dataclass
class StrategyResult:
    """Result from running a benchmark strategy."""

    strategy_name: str
    total_time_ms: float
    num_requests: int
    num_llm_calls: int

    # Token usage
    total_input_tokens: int
    total_output_tokens: int

    # Latency stats
    latencies_ms: List[float] = field(default_factory=list)

    # Results
    results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Cache metrics (if available)
    cache_hit_rate: Optional[float] = None

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def requests_per_second(self) -> float:
        return self.num_requests / (self.total_time_ms / 1000) if self.total_time_ms > 0 else 0

    def summary(self) -> str:
        cache_info = f"  Cache Hit Rate:  {self.cache_hit_rate:.1f}%\n" if self.cache_hit_rate is not None else ""
        return f"""
Strategy: {self.strategy_name}
{"=" * 50}
Total Time:        {self.total_time_ms:.1f} ms ({self.total_time_ms / 1000:.2f} s)
Num Requests:      {self.num_requests}
Num LLM Calls:     {self.num_llm_calls}
Requests/Second:   {self.requests_per_second:.2f}

Token Usage:
  Input Tokens:    {self.total_input_tokens:,}
  Output Tokens:   {self.total_output_tokens:,}
  Total Tokens:    {self.total_tokens:,}

Latency (ms):
  Average:         {self.avg_latency_ms:.1f}
  Min:             {min(self.latencies_ms) if self.latencies_ms else 0:.1f} (per call)
  Max:             {max(self.latencies_ms) if self.latencies_ms else 0:.1f} (per call)
{cache_info}
Errors:            {len(self.errors)}
{"=" * 50}
"""


class BaseStrategy:
    """Base class for benchmark strategies."""

    name: str = "base"

    def __init__(self, client: LLMClient):
        self.client = client

    async def run(
        self,
        extraction_prompts: List[List[Dict[str, str]]],
        update_prompts: List[List[Dict[str, str]]],
    ) -> StrategyResult:
        """
        Run the full add workflow with this strategy.

        Args:
            extraction_prompts: List of message arrays for extraction phase
            update_prompts: List of message arrays for update phase

        Returns:
            Combined result for both phases
        """
        raise NotImplementedError


class Mem0NativeStrategy(BaseStrategy):
    """
    Simulates mem0's native sequential behavior.

    For N add requests:
    - Processes each request completely before moving to the next
    - Pattern: E1→U1→E2→U2→E3→U3→...

    This is the baseline for comparison. vLLM's prefix cache won't be
    effective here because requests with shared prefixes are separated
    by different prompts.
    """

    name = "mem0_native"

    async def run(
        self,
        extraction_prompts: List[List[Dict[str, str]]],
        update_prompts: List[List[Dict[str, str]]],
    ) -> StrategyResult:
        start_time = time.perf_counter()

        responses: List[LLMResponse] = []
        n_requests = len(extraction_prompts)
        n_updates = len(update_prompts)

        # Interleave extraction and update calls to simulate mem0 native behavior
        # E1→U1→E2→U2→...
        for i in range(n_requests):
            # Extraction call
            extraction_response = await self.client.chat_completion(
                messages=extraction_prompts[i],
                response_format={"type": "json_object"},
            )
            responses.append(extraction_response)

            # Update call (if available)
            if i < n_updates:
                update_response = await self.client.chat_completion(
                    messages=update_prompts[i],
                    response_format={"type": "json_object"},
                )
                responses.append(update_response)

        total_time = (time.perf_counter() - start_time) * 1000

        return StrategyResult(
            strategy_name=self.name,
            total_time_ms=total_time,
            num_requests=n_requests,
            num_llm_calls=len(responses),
            total_input_tokens=sum(r.input_tokens for r in responses),
            total_output_tokens=sum(r.output_tokens for r in responses),
            latencies_ms=[r.latency_ms for r in responses],
            results=[{"content": r.content, "success": r.success} for r in responses],
            errors=[r.error for r in responses if r.error],
        )


class ConcurrentHttpStrategy(BaseStrategy):
    """
    Batches requests by phase using concurrent HTTP calls.

    For N add requests:
    - First, run all extraction requests concurrently
    - Then, run all update requests concurrently
    - Pattern: [E1,E2,E3,...] → [U1,U2,U3,...]

    vLLM will automatically batch concurrent requests with shared prefixes,
    enabling prefix cache hits on the shared system prompt.
    """

    name = "concurrent_http"

    def __init__(self, client: LLMClient, max_concurrent: int = 50):
        super().__init__(client)
        self.max_concurrent = max_concurrent

    async def run(
        self,
        extraction_prompts: List[List[Dict[str, str]]],
        update_prompts: List[List[Dict[str, str]]],
    ) -> StrategyResult:
        start_time = time.perf_counter()

        all_responses: List[LLMResponse] = []

        # Phase 1: All extractions concurrently
        extraction_responses = await self.client.batch_chat_completion(
            messages_list=extraction_prompts,
            response_format={"type": "json_object"},
            max_concurrent=self.max_concurrent,
        )
        all_responses.extend(extraction_responses)

        # Phase 2: All updates concurrently
        if update_prompts:
            update_responses = await self.client.batch_chat_completion(
                messages_list=update_prompts,
                response_format={"type": "json_object"},
                max_concurrent=self.max_concurrent,
            )
            all_responses.extend(update_responses)

        total_time = (time.perf_counter() - start_time) * 1000

        return StrategyResult(
            strategy_name=self.name,
            total_time_ms=total_time,
            num_requests=len(extraction_prompts),
            num_llm_calls=len(all_responses),
            total_input_tokens=sum(r.input_tokens for r in all_responses),
            total_output_tokens=sum(r.output_tokens for r in all_responses),
            latencies_ms=[r.latency_ms for r in all_responses],
            results=[{"content": r.content, "success": r.success} for r in all_responses],
            errors=[r.error for r in all_responses if r.error],
        )


class ExtractionOnlyStrategy(BaseStrategy):
    """
    Run only extraction phase - useful for isolating prefix cache effects.

    Extraction prompts share the same ~2000 token system prompt, making
    them ideal for measuring prefix cache effectiveness.
    """

    name = "extraction_only"

    def __init__(self, client: LLMClient, sequential: bool = False, max_concurrent: int = 50):
        super().__init__(client)
        self.sequential = sequential
        self.max_concurrent = max_concurrent

    async def run(
        self,
        extraction_prompts: List[List[Dict[str, str]]],
        update_prompts: List[List[Dict[str, str]]],  # Ignored
    ) -> StrategyResult:
        start_time = time.perf_counter()

        if self.sequential:
            # Sequential execution
            responses: List[LLMResponse] = []
            for prompt_msgs in extraction_prompts:
                response = await self.client.chat_completion(
                    messages=prompt_msgs,
                    response_format={"type": "json_object"},
                )
                responses.append(response)
        else:
            # Concurrent execution
            responses = await self.client.batch_chat_completion(
                messages_list=extraction_prompts,
                response_format={"type": "json_object"},
                max_concurrent=self.max_concurrent,
            )

        total_time = (time.perf_counter() - start_time) * 1000
        mode = "sequential" if self.sequential else "concurrent"

        return StrategyResult(
            strategy_name=f"{self.name}_{mode}",
            total_time_ms=total_time,
            num_requests=len(extraction_prompts),
            num_llm_calls=len(responses),
            total_input_tokens=sum(r.input_tokens for r in responses),
            total_output_tokens=sum(r.output_tokens for r in responses),
            latencies_ms=[r.latency_ms for r in responses],
            results=[{"content": r.content, "success": r.success} for r in responses],
            errors=[r.error for r in responses if r.error],
        )


def compare_results(
    baseline: StrategyResult,
    optimized: StrategyResult,
) -> str:
    """Generate comparison report between two strategy results."""

    speedup = baseline.total_time_ms / optimized.total_time_ms if optimized.total_time_ms > 0 else 0
    token_diff = optimized.total_tokens - baseline.total_tokens
    token_pct = (token_diff / baseline.total_tokens) * 100 if baseline.total_tokens > 0 else 0

    cache_comparison = ""
    if baseline.cache_hit_rate is not None and optimized.cache_hit_rate is not None:
        cache_diff = optimized.cache_hit_rate - baseline.cache_hit_rate
        cache_comparison = f"""
Cache Hit Rate:
  {baseline.strategy_name}:   {baseline.cache_hit_rate:.1f}%
  {optimized.strategy_name}:  {optimized.cache_hit_rate:.1f}%
  Improvement:        +{cache_diff:.1f}%
"""

    return f"""
Comparison: {baseline.strategy_name} vs {optimized.strategy_name}
{"=" * 60}

Time:
  {baseline.strategy_name}:   {baseline.total_time_ms:.1f} ms ({baseline.total_time_ms / 1000:.2f} s)
  {optimized.strategy_name}:  {optimized.total_time_ms:.1f} ms ({optimized.total_time_ms / 1000:.2f} s)
  Speedup:            {speedup:.2f}x

LLM Calls:
  {baseline.strategy_name}:   {baseline.num_llm_calls}
  {optimized.strategy_name}:  {optimized.num_llm_calls}

Tokens:
  {baseline.strategy_name}:   {baseline.total_tokens:,}
  {optimized.strategy_name}:  {optimized.total_tokens:,}
  Difference:         {token_diff:+,} ({token_pct:+.1f}%)
{cache_comparison}
{"=" * 60}
"""
