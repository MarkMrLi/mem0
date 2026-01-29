#!/usr/bin/env python3
"""
Mem0 Add Batch Benchmark - Main Entry Point

Benchmarks different strategies for batching LLM calls in the mem0 add operation.
Compares sequential, concurrent, and prompt-batched approaches.

Usage:
    python benchmark.py --phase token      # Token analysis only (no LLM calls)
    python benchmark.py --phase mock       # Mock mode (simulated LLM calls)
    python benchmark.py --phase full       # Full benchmark with real LLM calls
    python benchmark.py --num-requests 15  # Specify number of requests to test
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import get_llm_config, get_benchmark_config, validate_config
from llm_client import LLMClient
from token_analyzer import (
    load_prompts,
    analyze_extraction_prompts,
    analyze_update_prompts,
    run_token_analysis,
)
from strategies import (
    SingleUserScenario,
    MultiUserScenario,
    compare_results,
)


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def run_token_analysis_phase(prompts_file: str, num_requests: int) -> Dict[str, Any]:
    """
    Phase 1: Analyze token usage without calling LLM.

    This phase provides cost estimates for different strategies.
    """
    print_header("Phase 1: Token Cost Analysis")

    try:
        extraction_prompts, update_prompts = load_prompts(prompts_file)
    except FileNotFoundError:
        print(f"Error: Prompts file not found: {prompts_file}")
        return {}

    print(f"Loaded {len(extraction_prompts)} extraction prompts")
    print(f"Loaded {len(update_prompts)} update prompts")
    print(f"Analyzing first {num_requests} requests...\n")

    # Analyze extraction prompts
    extraction_analysis = analyze_extraction_prompts(
        extraction_prompts[:num_requests],
        batch_size=num_requests,
    )
    print(extraction_analysis)

    # Analyze update prompts
    update_analysis = analyze_update_prompts(
        update_prompts[:num_requests],
        batch_size=num_requests,
    )
    print("\nUpdate Prompts Analysis:")
    print(f"  Count: {update_analysis['count']}")
    print(f"  Total Tokens: {update_analysis['total_tokens']:,}")
    print(f"  Avg Tokens per Request: {update_analysis['avg_tokens']:.0f}")
    print(f"  Note: {update_analysis.get('note', 'N/A')}")

    return {
        "extraction": extraction_analysis,
        "update": update_analysis,
    }


async def run_mock_phase(prompts_file: str, num_requests: int) -> Dict[str, Any]:
    """
    Phase 2: Run with mock LLM calls to verify logic.

    Uses simulated delays to test the benchmark framework.
    """
    print_header("Phase 2: Mock Benchmark (Simulated LLM Calls)")

    try:
        extraction_prompts, update_prompts = load_prompts(prompts_file)
    except FileNotFoundError:
        print(f"Error: Prompts file not found: {prompts_file}")
        return {}

    # Limit to num_requests
    extraction_prompts = extraction_prompts[:num_requests]
    update_prompts = update_prompts[: min(num_requests, len(update_prompts))]

    print(f"Testing with {len(extraction_prompts)} extraction prompts")
    print(f"Testing with {len(update_prompts)} update prompts")
    print(f"Mock delay: 100ms per call\n")

    # Create mock client
    from config import LLMConfig

    mock_config = LLMConfig(
        base_url="http://mock",
        api_key="mock",
        model="mock-model",
    )
    client = LLMClient(config=mock_config, mock_mode=True, mock_delay_ms=100)

    results = {}

    # Test Single User Scenario
    print("\n--- Single User Scenario ---\n")
    single_user = SingleUserScenario(client)

    print("Running sequential strategy...")
    sequential_result = await single_user.run_sequential(extraction_prompts, update_prompts)
    print(sequential_result["extraction"].summary())

    print("Running batched extraction strategy...")
    batched_result = await single_user.run_batched_extraction(extraction_prompts, update_prompts)
    print(batched_result["extraction"].summary())

    print(compare_results(sequential_result, batched_result, "Sequential", "Batched"))

    results["single_user"] = {
        "sequential": sequential_result,
        "batched": batched_result,
    }

    # Test Multi User Scenario
    print("\n--- Multi User Scenario ---\n")
    multi_user = MultiUserScenario(client)

    print("Running sequential strategy...")
    sequential_result = await multi_user.run_sequential(extraction_prompts, update_prompts)
    print(sequential_result["extraction"].summary())

    print("Running full concurrent strategy...")
    concurrent_result = await multi_user.run_full_concurrent(extraction_prompts, update_prompts)
    print(concurrent_result["extraction"].summary())

    print("Running full batched strategy...")
    batched_result = await multi_user.run_full_batched(extraction_prompts, update_prompts)
    print(batched_result["extraction"].summary())

    print(compare_results(sequential_result, concurrent_result, "Sequential", "Concurrent"))
    print(compare_results(sequential_result, batched_result, "Sequential", "Full Batched"))

    results["multi_user"] = {
        "sequential": sequential_result,
        "concurrent": concurrent_result,
        "batched": batched_result,
    }

    return results


async def run_full_phase(prompts_file: str, num_requests: int) -> Dict[str, Any]:
    """
    Phase 3: Run full benchmark with real LLM calls.

    Requires vLLM or OpenAI-compatible server to be running.
    """
    print_header("Phase 3: Full Benchmark (Real LLM Calls)")

    # Load config
    llm_config = get_llm_config()
    if not validate_config(llm_config):
        print("Error: Invalid LLM configuration. Please check your .env file.")
        return {}

    print(f"LLM Endpoint: {llm_config.base_url}")
    print(f"Model: {llm_config.model}")

    try:
        extraction_prompts, update_prompts = load_prompts(prompts_file)
    except FileNotFoundError:
        print(f"Error: Prompts file not found: {prompts_file}")
        return {}

    # Limit to num_requests
    extraction_prompts = extraction_prompts[:num_requests]
    update_prompts = update_prompts[: min(num_requests, len(update_prompts))]

    print(f"\nTesting with {len(extraction_prompts)} extraction prompts")
    print(f"Testing with {len(update_prompts)} update prompts\n")

    # Create real client
    client = LLMClient(config=llm_config, mock_mode=False)

    results = {}

    # Test Single User Scenario
    print("\n--- Single User Scenario ---\n")
    single_user = SingleUserScenario(client)

    print("Running sequential strategy...")
    try:
        sequential_result = await single_user.run_sequential(extraction_prompts, update_prompts)
        print(sequential_result["extraction"].summary())
    except Exception as e:
        print(f"Error in sequential strategy: {e}")
        sequential_result = None

    print("Running batched extraction strategy...")
    try:
        batched_result = await single_user.run_batched_extraction(extraction_prompts, update_prompts)
        print(batched_result["extraction"].summary())
    except Exception as e:
        print(f"Error in batched strategy: {e}")
        batched_result = None

    if sequential_result and batched_result:
        print(compare_results(sequential_result, batched_result, "Sequential", "Batched"))
        results["single_user"] = {
            "sequential": sequential_result,
            "batched": batched_result,
        }

    # Test Multi User Scenario
    print("\n--- Multi User Scenario ---\n")
    multi_user = MultiUserScenario(client)

    print("Running sequential strategy...")
    try:
        sequential_result = await multi_user.run_sequential(extraction_prompts, update_prompts)
        print(sequential_result["extraction"].summary())
    except Exception as e:
        print(f"Error in sequential strategy: {e}")
        sequential_result = None

    print("Running full concurrent strategy...")
    try:
        concurrent_result = await multi_user.run_full_concurrent(extraction_prompts, update_prompts)
        print(concurrent_result["extraction"].summary())
    except Exception as e:
        print(f"Error in concurrent strategy: {e}")
        concurrent_result = None

    print("Running full batched strategy...")
    try:
        batched_result = await multi_user.run_full_batched(extraction_prompts, update_prompts)
        print(batched_result["extraction"].summary())
    except Exception as e:
        print(f"Error in batched strategy: {e}")
        batched_result = None

    if sequential_result and concurrent_result:
        print(compare_results(sequential_result, concurrent_result, "Sequential", "Concurrent"))
    if sequential_result and batched_result:
        print(compare_results(sequential_result, batched_result, "Sequential", "Full Batched"))

    results["multi_user"] = {
        "sequential": sequential_result,
        "concurrent": concurrent_result,
        "batched": batched_result,
    }

    return results


def generate_report(
    token_analysis: Dict[str, Any],
    mock_results: Dict[str, Any],
    full_results: Dict[str, Any],
    output_file: str = None,
) -> str:
    """Generate a comprehensive benchmark report."""

    report = []
    report.append("=" * 70)
    report.append("  Mem0 Add Batch Benchmark Report")
    report.append(f"  Generated: {datetime.now().isoformat()}")
    report.append("=" * 70)

    if token_analysis:
        report.append("\n## Token Cost Analysis\n")
        if "extraction" in token_analysis:
            ea = token_analysis["extraction"]
            report.append(f"Extraction Phase ({ea.sequential_total_input:,} -> {ea.batched_total_input:,} tokens):")
            report.append(f"  Savings: {ea.total_savings_pct:.1f}% ({ea.tokens_saved:,} tokens)")

    report.append("\n## Benchmark Summary\n")
    report.append("See detailed output above for full results.")

    report_text = "\n".join(report)

    if output_file:
        with open(output_file, "w") as f:
            f.write(report_text)
        print(f"\nReport saved to: {output_file}")

    return report_text


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mem0 Add Batch Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        choices=["token", "mock", "full", "all"],
        default="mock",
        help="Which phase to run (default: mock)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=15,
        help="Number of requests to test (default: 15)",
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
        help="Output file for report",
    )

    args = parser.parse_args()

    # Determine prompts file path
    if args.prompts_file:
        prompts_file = args.prompts_file
    else:
        prompts_file = str(Path(__file__).parent.parent / "prompts_storage.json")

    print_header("Mem0 Add Batch Benchmark")
    print(f"Phase: {args.phase}")
    print(f"Num Requests: {args.num_requests}")
    print(f"Prompts File: {prompts_file}")

    token_analysis = {}
    mock_results = {}
    full_results = {}

    if args.phase in ["token", "all"]:
        token_analysis = run_token_analysis_phase(prompts_file, args.num_requests)

    if args.phase in ["mock", "all"]:
        mock_results = await run_mock_phase(prompts_file, args.num_requests)

    if args.phase in ["full", "all"]:
        full_results = await run_full_phase(prompts_file, args.num_requests)

    # Generate report
    report = generate_report(
        token_analysis,
        mock_results,
        full_results,
        args.output,
    )

    print("\n" + report)
    print("\nBenchmark complete!")


if __name__ == "__main__":
    asyncio.run(main())
