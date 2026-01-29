"""
Benchmark strategies for comparing different LLM call patterns.
Implements sequential, concurrent, and batched approaches.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Support both direct run and package import
try:
    from .llm_client import LLMClient, LLMResponse
    from .prompt_batcher import (
        BatchedPrompt,
        create_batched_extraction_prompt,
        parse_batched_response,
        unbatch_results,
    )
except ImportError:
    from llm_client import LLMClient, LLMResponse
    from prompt_batcher import (
        BatchedPrompt,
        create_batched_extraction_prompt,
        parse_batched_response,
        unbatch_results,
    )


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
  Min:             {min(self.latencies_ms):.1f} (per call)
  Max:             {max(self.latencies_ms):.1f} (per call)

Errors:            {len(self.errors)}
{"=" * 50}
"""


class BaseStrategy:
    """Base class for benchmark strategies."""

    name: str = "base"

    def __init__(self, client: LLMClient):
        self.client = client

    async def run_extraction(
        self,
        prompts: List[List[Dict[str, str]]],
    ) -> StrategyResult:
        """Run extraction phase with this strategy."""
        raise NotImplementedError

    async def run_update(
        self,
        prompts: List[List[Dict[str, str]]],
    ) -> StrategyResult:
        """Run update phase with this strategy."""
        raise NotImplementedError


class SequentialStrategy(BaseStrategy):
    """
    Sequential strategy: Execute requests one by one.
    This is the baseline for comparison.
    """

    name = "sequential"

    async def run_extraction(
        self,
        prompts: List[List[Dict[str, str]]],
    ) -> StrategyResult:
        start_time = time.perf_counter()

        responses: List[LLMResponse] = []
        for prompt_msgs in prompts:
            response = await self.client.chat_completion(
                messages=prompt_msgs,
                response_format={"type": "json_object"},
            )
            responses.append(response)

        total_time = (time.perf_counter() - start_time) * 1000

        return StrategyResult(
            strategy_name=self.name,
            total_time_ms=total_time,
            num_requests=len(prompts),
            num_llm_calls=len(prompts),
            total_input_tokens=sum(r.input_tokens for r in responses),
            total_output_tokens=sum(r.output_tokens for r in responses),
            latencies_ms=[r.latency_ms for r in responses],
            results=[{"content": r.content, "success": r.success} for r in responses],
            errors=[r.error for r in responses if r.error],
        )

    async def run_update(
        self,
        prompts: List[List[Dict[str, str]]],
    ) -> StrategyResult:
        # Update phase is the same as extraction for sequential
        return await self.run_extraction(prompts)


class ConcurrentStrategy(BaseStrategy):
    """
    Concurrent strategy: Execute all requests in parallel using asyncio.
    vLLM will automatically batch these requests.
    """

    name = "concurrent"

    def __init__(self, client: LLMClient, max_concurrent: int = 10):
        super().__init__(client)
        self.max_concurrent = max_concurrent

    async def run_extraction(
        self,
        prompts: List[List[Dict[str, str]]],
    ) -> StrategyResult:
        start_time = time.perf_counter()

        responses = await self.client.batch_chat_completion(
            messages_list=prompts,
            response_format={"type": "json_object"},
            max_concurrent=self.max_concurrent,
        )

        total_time = (time.perf_counter() - start_time) * 1000

        return StrategyResult(
            strategy_name=self.name,
            total_time_ms=total_time,
            num_requests=len(prompts),
            num_llm_calls=len(prompts),  # Still N calls, but concurrent
            total_input_tokens=sum(r.input_tokens for r in responses),
            total_output_tokens=sum(r.output_tokens for r in responses),
            latencies_ms=[r.latency_ms for r in responses],
            results=[{"content": r.content, "success": r.success} for r in responses],
            errors=[r.error for r in responses if r.error],
        )

    async def run_update(
        self,
        prompts: List[List[Dict[str, str]]],
    ) -> StrategyResult:
        return await self.run_extraction(prompts)


class BatchedExtractionStrategy(BaseStrategy):
    """
    Batched extraction strategy: Combine multiple prompts into single requests.
    Reduces token usage by sharing system prompts.
    """

    name = "batched_extraction"

    def __init__(self, client: LLMClient, max_items_per_batch: int = 15):
        super().__init__(client)
        self.max_items_per_batch = max_items_per_batch

    async def run_extraction(
        self,
        prompts: List[List[Dict[str, str]]],
    ) -> StrategyResult:
        start_time = time.perf_counter()

        # Create batched prompts
        batched_prompts = create_batched_extraction_prompt(
            prompts,
            max_items_per_batch=self.max_items_per_batch,
        )

        # Execute batched requests
        responses: List[LLMResponse] = []
        all_batch_results = []

        for batched_prompt in batched_prompts:
            response = await self.client.chat_completion(
                messages=batched_prompt.messages,
                response_format={"type": "json_object"},
            )
            responses.append(response)

            # Parse batched response
            batch_results = parse_batched_response(
                response.content,
                batched_prompt,
            )
            all_batch_results.append(batch_results)

        # Unbatch results to original order
        final_results = unbatch_results(all_batch_results, len(prompts))

        total_time = (time.perf_counter() - start_time) * 1000

        return StrategyResult(
            strategy_name=self.name,
            total_time_ms=total_time,
            num_requests=len(prompts),
            num_llm_calls=len(batched_prompts),
            total_input_tokens=sum(r.input_tokens for r in responses),
            total_output_tokens=sum(r.output_tokens for r in responses),
            latencies_ms=[r.latency_ms for r in responses],
            results=final_results,
            errors=[r.error for r in responses if r.error],
        )

    async def run_update(
        self,
        prompts: List[List[Dict[str, str]]],
    ) -> StrategyResult:
        # Update phase runs sequentially (dependency on previous results)
        sequential = SequentialStrategy(self.client)
        result = await sequential.run_update(prompts)
        result.strategy_name = f"{self.name}_update"
        return result


class SingleUserScenario:
    """
    Single user scenario: One user making multiple add requests.
    Update phase must be sequential due to memory state dependencies.
    """

    def __init__(self, client: LLMClient):
        self.client = client

    async def run_sequential(
        self,
        extraction_prompts: List[List[Dict[str, str]]],
        update_prompts: List[List[Dict[str, str]]],
    ) -> Dict[str, StrategyResult]:
        """Run with fully sequential approach."""
        strategy = SequentialStrategy(self.client)

        extraction_result = await strategy.run_extraction(extraction_prompts)
        update_result = await strategy.run_update(update_prompts)

        return {
            "extraction": extraction_result,
            "update": update_result,
        }

    async def run_batched_extraction(
        self,
        extraction_prompts: List[List[Dict[str, str]]],
        update_prompts: List[List[Dict[str, str]]],
    ) -> Dict[str, StrategyResult]:
        """Run with batched extraction, sequential update."""
        extraction_strategy = BatchedExtractionStrategy(self.client)
        update_strategy = SequentialStrategy(self.client)

        extraction_result = await extraction_strategy.run_extraction(extraction_prompts)
        update_result = await update_strategy.run_update(update_prompts)

        return {
            "extraction": extraction_result,
            "update": update_result,
        }


class MultiUserScenario:
    """
    Multi-user scenario: Multiple users making add requests concurrently.
    Both extraction and update phases can be parallelized.
    """

    def __init__(self, client: LLMClient):
        self.client = client

    async def run_sequential(
        self,
        extraction_prompts: List[List[Dict[str, str]]],
        update_prompts: List[List[Dict[str, str]]],
    ) -> Dict[str, StrategyResult]:
        """Run with fully sequential approach (baseline)."""
        strategy = SequentialStrategy(self.client)

        extraction_result = await strategy.run_extraction(extraction_prompts)
        update_result = await strategy.run_update(update_prompts)

        return {
            "extraction": extraction_result,
            "update": update_result,
        }

    async def run_full_concurrent(
        self,
        extraction_prompts: List[List[Dict[str, str]]],
        update_prompts: List[List[Dict[str, str]]],
    ) -> Dict[str, StrategyResult]:
        """Run with full concurrency (both phases parallel)."""
        strategy = ConcurrentStrategy(self.client)

        extraction_result = await strategy.run_extraction(extraction_prompts)
        update_result = await strategy.run_update(update_prompts)

        return {
            "extraction": extraction_result,
            "update": update_result,
        }

    async def run_full_batched(
        self,
        extraction_prompts: List[List[Dict[str, str]]],
        update_prompts: List[List[Dict[str, str]]],
    ) -> Dict[str, StrategyResult]:
        """Run with batched extraction and concurrent update."""
        extraction_strategy = BatchedExtractionStrategy(self.client)
        update_strategy = ConcurrentStrategy(self.client)

        extraction_result = await extraction_strategy.run_extraction(extraction_prompts)
        update_result = await update_strategy.run_update(update_prompts)

        return {
            "extraction": extraction_result,
            "update": update_result,
        }


def compare_results(
    baseline: Dict[str, StrategyResult],
    optimized: Dict[str, StrategyResult],
    baseline_name: str = "Sequential",
    optimized_name: str = "Optimized",
) -> str:
    """Generate comparison report between two strategy results."""

    baseline_total = baseline["extraction"].total_time_ms + baseline["update"].total_time_ms
    optimized_total = optimized["extraction"].total_time_ms + optimized["update"].total_time_ms

    baseline_tokens = baseline["extraction"].total_tokens + baseline["update"].total_tokens
    optimized_tokens = optimized["extraction"].total_tokens + optimized["update"].total_tokens

    speedup = baseline_total / optimized_total if optimized_total > 0 else 0
    token_savings = (1 - optimized_tokens / baseline_tokens) * 100 if baseline_tokens > 0 else 0

    return f"""
Comparison: {baseline_name} vs {optimized_name}
{"=" * 60}

Time:
  {baseline_name}:   {baseline_total:.1f} ms
  {optimized_name}:  {optimized_total:.1f} ms
  Speedup:           {speedup:.2f}x

LLM Calls:
  {baseline_name}:   {baseline["extraction"].num_llm_calls + baseline["update"].num_llm_calls}
  {optimized_name}:  {optimized["extraction"].num_llm_calls + optimized["update"].num_llm_calls}

Tokens:
  {baseline_name}:   {baseline_tokens:,}
  {optimized_name}:  {optimized_tokens:,}
  Savings:           {token_savings:.1f}%

{"=" * 60}
"""
