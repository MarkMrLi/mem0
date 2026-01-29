#!/usr/bin/env python3
"""
Offline Batch Benchmark - Server-side Script

This script runs directly on the vLLM server where the model is loaded.
It uses vLLM's offline LLM.generate() API for optimal batching performance.

Requirements:
- Run this on the same machine where vLLM is installed
- The model must be loadable by vLLM

Usage:
    python offline_batch.py --prompts prompts_storage.json --num-requests 20
    python offline_batch.py --prompts prompts_storage.json --mode extraction-only
    python offline_batch.py --prompts prompts_storage.json --mode compare

Output:
    - Prints timing and results to stdout
    - Optionally saves results to JSON file
"""

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BatchResult:
    """Result from a batch run."""

    strategy_name: str
    total_time_ms: float
    num_requests: int
    prompt_tokens: int
    completion_tokens: int
    results: List[str]

    @property
    def requests_per_second(self) -> float:
        return self.num_requests / (self.total_time_ms / 1000) if self.total_time_ms > 0 else 0

    def summary(self) -> str:
        return f"""
Strategy: {self.strategy_name}
{"=" * 50}
Total Time:        {self.total_time_ms:.1f} ms ({self.total_time_ms / 1000:.2f} s)
Num Requests:      {self.num_requests}
Requests/Second:   {self.requests_per_second:.2f}

Tokens:
  Prompt:          {self.prompt_tokens:,}
  Completion:      {self.completion_tokens:,}
  Total:           {self.prompt_tokens + self.completion_tokens:,}
{"=" * 50}
"""


def load_prompts(prompts_file: str) -> Tuple[List[List[Dict[str, str]]], List[List[Dict[str, str]]]]:
    """Load prompts from prompts_storage.json format."""
    with open(prompts_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    extraction_prompts = data.get("extraction", [])
    update_prompts = data.get("update", [])

    return extraction_prompts, update_prompts


def messages_to_prompt(messages: List[Dict[str, str]], tokenizer) -> str:
    """Convert OpenAI-style messages to a prompt string using the tokenizer's chat template."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_sequential_batch(
    llm,
    tokenizer,
    extraction_prompts: List[List[Dict[str, str]]],
    update_prompts: List[List[Dict[str, str]]],
    sampling_params,
) -> BatchResult:
    """
    Simulate mem0 native sequential behavior.

    Pattern: E1→U1→E2→U2→...
    Each request is processed individually (batch size = 1).
    """
    from vllm import SamplingParams

    start_time = time.perf_counter()

    all_outputs = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    n_requests = len(extraction_prompts)
    n_updates = len(update_prompts)

    for i in range(n_requests):
        # Extraction
        prompt = messages_to_prompt(extraction_prompts[i], tokenizer)
        outputs = llm.generate([prompt], sampling_params)
        all_outputs.append(outputs[0].outputs[0].text)
        total_prompt_tokens += len(outputs[0].prompt_token_ids)
        total_completion_tokens += len(outputs[0].outputs[0].token_ids)

        # Update (if available)
        if i < n_updates:
            prompt = messages_to_prompt(update_prompts[i], tokenizer)
            outputs = llm.generate([prompt], sampling_params)
            all_outputs.append(outputs[0].outputs[0].text)
            total_prompt_tokens += len(outputs[0].prompt_token_ids)
            total_completion_tokens += len(outputs[0].outputs[0].token_ids)

    total_time = (time.perf_counter() - start_time) * 1000

    return BatchResult(
        strategy_name="offline_sequential",
        total_time_ms=total_time,
        num_requests=n_requests,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        results=all_outputs,
    )


def run_batched_by_phase(
    llm,
    tokenizer,
    extraction_prompts: List[List[Dict[str, str]]],
    update_prompts: List[List[Dict[str, str]]],
    sampling_params,
) -> BatchResult:
    """
    Batched by phase approach.

    Pattern: [E1,E2,E3,...] → [U1,U2,U3,...]
    All extractions run as one batch, then all updates as another batch.
    This maximizes prefix cache utilization.
    """
    start_time = time.perf_counter()

    all_outputs = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Phase 1: All extractions
    if extraction_prompts:
        extraction_texts = [messages_to_prompt(p, tokenizer) for p in extraction_prompts]
        outputs = llm.generate(extraction_texts, sampling_params)

        for output in outputs:
            all_outputs.append(output.outputs[0].text)
            total_prompt_tokens += len(output.prompt_token_ids)
            total_completion_tokens += len(output.outputs[0].token_ids)

    # Phase 2: All updates
    if update_prompts:
        update_texts = [messages_to_prompt(p, tokenizer) for p in update_prompts]
        outputs = llm.generate(update_texts, sampling_params)

        for output in outputs:
            all_outputs.append(output.outputs[0].text)
            total_prompt_tokens += len(output.prompt_token_ids)
            total_completion_tokens += len(output.outputs[0].token_ids)

    total_time = (time.perf_counter() - start_time) * 1000

    return BatchResult(
        strategy_name="offline_batched",
        total_time_ms=total_time,
        num_requests=len(extraction_prompts),
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        results=all_outputs,
    )


def run_extraction_only(
    llm,
    tokenizer,
    extraction_prompts: List[List[Dict[str, str]]],
    sampling_params,
    sequential: bool = False,
) -> BatchResult:
    """
    Run only extraction phase for focused prefix cache testing.

    Args:
        sequential: If True, run one at a time. If False, run as batch.
    """
    start_time = time.perf_counter()

    all_outputs = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    if sequential:
        # One at a time
        for prompt_msgs in extraction_prompts:
            prompt = messages_to_prompt(prompt_msgs, tokenizer)
            outputs = llm.generate([prompt], sampling_params)
            all_outputs.append(outputs[0].outputs[0].text)
            total_prompt_tokens += len(outputs[0].prompt_token_ids)
            total_completion_tokens += len(outputs[0].outputs[0].token_ids)
    else:
        # All at once
        prompts = [messages_to_prompt(p, tokenizer) for p in extraction_prompts]
        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            all_outputs.append(output.outputs[0].text)
            total_prompt_tokens += len(output.prompt_token_ids)
            total_completion_tokens += len(output.outputs[0].token_ids)

    total_time = (time.perf_counter() - start_time) * 1000
    mode = "sequential" if sequential else "batched"

    return BatchResult(
        strategy_name=f"extraction_only_{mode}",
        total_time_ms=total_time,
        num_requests=len(extraction_prompts),
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        results=all_outputs,
    )


def print_comparison(results: List[BatchResult]) -> None:
    """Print comparison table."""
    if len(results) < 2:
        return

    print("\n" + "=" * 70)
    print("  Comparison Summary")
    print("=" * 70)

    baseline = results[0]

    print(f"\n{'Strategy':<25} {'Time (s)':<12} {'Speedup':<10} {'Req/s':<10}")
    print("-" * 60)

    for result in results:
        speedup = baseline.total_time_ms / result.total_time_ms if result.total_time_ms > 0 else 0
        print(
            f"{result.strategy_name:<25} {result.total_time_ms / 1000:<12.2f} {speedup:<10.2f}x {result.requests_per_second:<10.1f}"
        )

    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Offline Batch Benchmark using vLLM LLM.generate()",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or name (e.g., /home/llz/model/qwen3-30b-a3b-instruct-2507)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to prompts_storage.json",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=20,
        help="Number of requests to test (default: 20)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "extraction-only", "compare"],
        default="compare",
        help="Test mode (default: compare)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens to generate (default: 1024)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        default=True,
        help="Enable prefix caching (default: True)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )

    args = parser.parse_args()

    # Import vLLM (only available on server)
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: vLLM is not installed. This script must run on the vLLM server.")
        print("Install with: pip install vllm")
        return

    print("=" * 70)
    print("  Offline Batch Benchmark")
    print("=" * 70)
    print(f"\nModel:            {args.model}")
    print(f"Prompts file:     {args.prompts}")
    print(f"Num requests:     {args.num_requests}")
    print(f"Mode:             {args.mode}")
    print(f"Prefix caching:   {args.enable_prefix_caching}")

    # Load prompts
    extraction_prompts, update_prompts = load_prompts(args.prompts)
    extraction_prompts = extraction_prompts[: args.num_requests]
    update_prompts = update_prompts[: min(args.num_requests, len(update_prompts))]

    print(f"\nLoaded {len(extraction_prompts)} extraction prompts")
    print(f"Loaded {len(update_prompts)} update prompts")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Initialize LLM
    print("Loading model...")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
        trust_remote_code=True,
    )

    # Sampling params
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0,
    )

    results: List[BatchResult] = []

    # Run benchmarks based on mode
    if args.mode == "extraction-only":
        print("\n--- Running extraction-only (sequential) ---")
        result = run_extraction_only(llm, tokenizer, extraction_prompts, sampling_params, sequential=True)
        print(result.summary())
        results.append(result)

        print("\n--- Running extraction-only (batched) ---")
        result = run_extraction_only(llm, tokenizer, extraction_prompts, sampling_params, sequential=False)
        print(result.summary())
        results.append(result)

    elif args.mode == "full":
        print("\n--- Running batched by phase ---")
        result = run_batched_by_phase(llm, tokenizer, extraction_prompts, update_prompts, sampling_params)
        print(result.summary())
        results.append(result)

    elif args.mode == "compare":
        print("\n--- Running sequential (simulating mem0 native) ---")
        result = run_sequential_batch(llm, tokenizer, extraction_prompts, update_prompts, sampling_params)
        print(result.summary())
        results.append(result)

        print("\n--- Running batched by phase ---")
        result = run_batched_by_phase(llm, tokenizer, extraction_prompts, update_prompts, sampling_params)
        print(result.summary())
        results.append(result)

    # Print comparison
    print_comparison(results)

    # Save results if requested
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model": args.model,
                "num_requests": args.num_requests,
                "mode": args.mode,
                "enable_prefix_caching": args.enable_prefix_caching,
            },
            "results": [
                {
                    "strategy": r.strategy_name,
                    "total_time_ms": r.total_time_ms,
                    "num_requests": r.num_requests,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "requests_per_second": r.requests_per_second,
                }
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
