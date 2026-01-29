#!/usr/bin/env python3
"""
Mem0 Add Batch Benchmark - Main Entry Point

Benchmarks different scheduling strategies for LLM calls in the mem0 add operation.
Measures the impact of vLLM prefix caching when batching requests by phase.

Strategies:
- mem0_native: E1→U1→E2→U2→... (sequential, baseline)
- concurrent_http: [E1,E2,...] → [U1,U2,...] (batched by phase)

Usage:
    python benchmark.py --strategy all --num-requests 20
    python benchmark.py --strategy mem0_native --num-requests 10
    python benchmark.py --strategy concurrent_http --num-requests 10
    python benchmark.py --mode extraction-only --num-requests 20
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import get_llm_config, get_benchmark_config, get_vllm_metrics_config, validate_config
from llm_client import LLMClient
from token_analyzer import load_prompts
from strategies import (
    Mem0NativeStrategy,
    ConcurrentHttpStrategy,
    ExtractionOnlyStrategy,
    StrategyResult,
    compare_results,
)
from vllm_metrics import MetricsTracker, VLLMMetrics


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def load_test_prompts(
    prompts_file: str,
    num_requests: int,
) -> Tuple[List[List[Dict[str, str]]], List[List[Dict[str, str]]]]:
    """Load and slice prompts for testing."""
    try:
        extraction_prompts, update_prompts = load_prompts(prompts_file)
    except FileNotFoundError:
        print(f"Error: Prompts file not found: {prompts_file}")
        sys.exit(1)

    # Limit to num_requests
    extraction_prompts = extraction_prompts[:num_requests]
    update_prompts = update_prompts[: min(num_requests, len(update_prompts))]

    return extraction_prompts, update_prompts


async def run_strategy_with_metrics(
    strategy,
    extraction_prompts: List[List[Dict[str, str]]],
    update_prompts: List[List[Dict[str, str]]],
    metrics_tracker: Optional[MetricsTracker],
) -> StrategyResult:
    """Run a strategy and capture before/after metrics."""

    # Capture metrics before
    if metrics_tracker:
        await metrics_tracker.start()

    # Run the strategy
    result = await strategy.run(extraction_prompts, update_prompts)

    # Capture metrics after
    if metrics_tracker:
        await metrics_tracker.stop()
        cache_hit_rate = metrics_tracker.get_cache_hit_rate()
        if cache_hit_rate is not None:
            result.cache_hit_rate = cache_hit_rate

    return result


async def run_benchmark(
    prompts_file: str,
    num_requests: int,
    strategies: List[str],
    mode: str,
    max_concurrent: int,
) -> Dict[str, StrategyResult]:
    """Run the benchmark with specified strategies."""

    print_header("Mem0 Add Batch Benchmark")

    # Load config
    llm_config = get_llm_config()
    metrics_config = get_vllm_metrics_config()

    if not validate_config(llm_config):
        print("Error: Invalid LLM configuration. Please check your .env file.")
        return {}

    print(f"LLM Endpoint:     {llm_config.base_url}")
    print(f"Model:            {llm_config.model}")
    print(f"Metrics URL:      {metrics_config.metrics_url}")
    print(f"Metrics Enabled:  {metrics_config.enabled}")
    print(f"Mode:             {mode}")
    print(f"Strategies:       {', '.join(strategies)}")
    print(f"Num Requests:     {num_requests}")
    print(f"Max Concurrent:   {max_concurrent}")

    # Load prompts
    extraction_prompts, update_prompts = load_test_prompts(prompts_file, num_requests)
    print(f"\nLoaded {len(extraction_prompts)} extraction prompts")
    print(f"Loaded {len(update_prompts)} update prompts")

    # Create client
    client = LLMClient(config=llm_config, mock_mode=False)

    results: Dict[str, StrategyResult] = {}

    # Run each strategy
    for strategy_name in strategies:
        print(f"\n{'=' * 50}")
        print(f"Running strategy: {strategy_name}")
        print("=" * 50)

        # Create metrics tracker for this run
        tracker = MetricsTracker(
            metrics_url=metrics_config.metrics_url,
            enabled=metrics_config.enabled,
        )

        try:
            if mode == "extraction-only":
                # Run extraction-only mode
                sequential = strategy_name == "mem0_native"
                strategy = ExtractionOnlyStrategy(
                    client,
                    sequential=sequential,
                    max_concurrent=max_concurrent,
                )
                result = await run_strategy_with_metrics(
                    strategy,
                    extraction_prompts,
                    [],  # No update prompts
                    tracker,
                )
            elif strategy_name == "mem0_native":
                strategy = Mem0NativeStrategy(client)
                result = await run_strategy_with_metrics(
                    strategy,
                    extraction_prompts,
                    update_prompts,
                    tracker,
                )
            elif strategy_name == "concurrent_http":
                strategy = ConcurrentHttpStrategy(client, max_concurrent=max_concurrent)
                result = await run_strategy_with_metrics(
                    strategy,
                    extraction_prompts,
                    update_prompts,
                    tracker,
                )
            else:
                print(f"Unknown strategy: {strategy_name}")
                continue

            results[strategy_name] = result
            print(result.summary())

            # Print metrics delta if available
            if tracker.delta:
                print(tracker.summary())

        except Exception as e:
            print(f"Error running {strategy_name}: {e}")
            import traceback

            traceback.print_exc()

    return results


def print_comparison_report(results: Dict[str, StrategyResult]) -> None:
    """Print comparison between strategies."""

    if len(results) < 2:
        return

    print_header("Strategy Comparison")

    # If we have both strategies, compare them
    if "mem0_native" in results and "concurrent_http" in results:
        print(compare_results(results["mem0_native"], results["concurrent_http"]))

    # Print summary table
    print("\nSummary Table:")
    print("-" * 80)
    print(f"{'Strategy':<20} {'Time (s)':<12} {'Cache Hit %':<14} {'Speedup':<10} {'Tokens':<12}")
    print("-" * 80)

    baseline_time = None
    for name, result in results.items():
        if baseline_time is None:
            baseline_time = result.total_time_ms
            speedup = "1.00x"
        else:
            speedup = f"{baseline_time / result.total_time_ms:.2f}x"

        cache_rate = f"{result.cache_hit_rate:.1f}%" if result.cache_hit_rate is not None else "N/A"

        print(f"{name:<20} {result.total_time_ms / 1000:<12.2f} {cache_rate:<14} {speedup:<10} {result.total_tokens:,}")

    print("-" * 80)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mem0 Add Batch Benchmark - Measure vLLM prefix caching impact",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--strategy",
        choices=["mem0_native", "concurrent_http", "all"],
        default="all",
        help="Which strategy to run (default: all)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "extraction-only"],
        default="full",
        help="Test mode: full (E+U) or extraction-only (default: full)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=20,
        help="Number of requests to test (default: 20)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        help="Max concurrent requests for concurrent strategy (default: 50)",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Path to prompts_storage.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )

    args = parser.parse_args()

    # Determine prompts file path
    if args.prompts_file:
        prompts_file = args.prompts_file
    else:
        prompts_file = str(Path(__file__).parent.parent / "prompts_storage.json")

    # Determine strategies to run
    if args.strategy == "all":
        strategies = ["mem0_native", "concurrent_http"]
    else:
        strategies = [args.strategy]

    # Run benchmark
    results = await run_benchmark(
        prompts_file=prompts_file,
        num_requests=args.num_requests,
        strategies=strategies,
        mode=args.mode,
        max_concurrent=args.max_concurrent,
    )

    # Print comparison
    if results:
        print_comparison_report(results)

    # Save results if requested
    if args.output and results:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_requests": args.num_requests,
                "mode": args.mode,
                "strategies": strategies,
            },
            "results": {
                name: {
                    "total_time_ms": r.total_time_ms,
                    "num_requests": r.num_requests,
                    "num_llm_calls": r.num_llm_calls,
                    "total_input_tokens": r.total_input_tokens,
                    "total_output_tokens": r.total_output_tokens,
                    "cache_hit_rate": r.cache_hit_rate,
                    "errors": r.errors,
                }
                for name, r in results.items()
            },
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    asyncio.run(main())
