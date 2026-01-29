"""
Token cost analyzer for batch benchmark.
Analyzes token usage and potential savings from prompt batching.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Try to use tiktoken for accurate token counting
try:
    import tiktoken

    _encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        """Count tokens using tiktoken."""
        return len(_encoding.encode(text))
except ImportError:

    def count_tokens(text: str) -> int:
        """Approximate token count (4 chars per token)."""
        return len(text) // 4


@dataclass
class TokenAnalysis:
    """Token usage analysis results."""

    # Sequential approach
    sequential_system_tokens: int
    sequential_user_tokens: int
    sequential_total_input: int

    # Batched approach
    batched_system_tokens: int
    batched_user_tokens: int
    batched_overhead_tokens: int
    batched_total_input: int

    # Savings
    system_prompt_savings_pct: float
    total_savings_pct: float
    tokens_saved: int

    def __str__(self) -> str:
        return f"""
Token Cost Analysis:
{"=" * 60}
Sequential Approach:
  System Prompt Tokens: {self.sequential_system_tokens:,}
  User Content Tokens:  {self.sequential_user_tokens:,}
  Total Input Tokens:   {self.sequential_total_input:,}

Batched Approach:
  System Prompt Tokens: {self.batched_system_tokens:,}
  User Content Tokens:  {self.batched_user_tokens:,}
  Overhead Tokens:      {self.batched_overhead_tokens:,}
  Total Input Tokens:   {self.batched_total_input:,}

Savings:
  System Prompt Savings: {self.system_prompt_savings_pct:.1f}%
  Total Input Savings:   {self.total_savings_pct:.1f}%
  Tokens Saved:          {self.tokens_saved:,}
{"=" * 60}
"""


def load_prompts(prompts_file: str) -> Tuple[List[Any], List[Any]]:
    """
    Load extraction and update prompts from JSON file.

    Returns:
        Tuple of (extraction_prompts, update_prompts)
    """
    with open(prompts_file, "r") as f:
        data = json.load(f)

    return data.get("extraction_prompts", []), data.get("update_prompts", [])


def analyze_extraction_prompts(
    prompts: List[List[Dict[str, str]]],
    batch_size: int = None,
) -> TokenAnalysis:
    """
    Analyze token usage for extraction prompts.

    Args:
        prompts: List of prompt message lists
        batch_size: Number of prompts to analyze (None = all)

    Returns:
        TokenAnalysis with detailed breakdown
    """
    if batch_size:
        prompts = prompts[:batch_size]

    n = len(prompts)
    if n == 0:
        return TokenAnalysis(
            sequential_system_tokens=0,
            sequential_user_tokens=0,
            sequential_total_input=0,
            batched_system_tokens=0,
            batched_user_tokens=0,
            batched_overhead_tokens=0,
            batched_total_input=0,
            system_prompt_savings_pct=0,
            total_savings_pct=0,
            tokens_saved=0,
        )

    # Extract system prompt (should be the same for all)
    system_prompt = ""
    user_contents = []

    for prompt_msgs in prompts:
        for msg in prompt_msgs:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            elif msg.get("role") == "user":
                user_contents.append(msg.get("content", ""))

    # Count tokens
    system_tokens = count_tokens(system_prompt)
    user_tokens_list = [count_tokens(content) for content in user_contents]
    total_user_tokens = sum(user_tokens_list)

    # Sequential: N copies of system prompt + N user prompts
    sequential_system = n * system_tokens
    sequential_user = total_user_tokens
    sequential_total = sequential_system + sequential_user

    # Batched: 1 system prompt + combined user prompts + overhead
    # Overhead includes batch instruction additions and item separators
    batch_instruction_overhead = count_tokens(
        "\n\nYou will receive multiple inputs. Process each one and return results "
        'in a JSON array format: {"results": [{"item_id": 1, "facts": [...]}, ...]}\n'
    )
    item_separator_overhead = count_tokens("[Item 1]:\n") * n

    batched_system = system_tokens + batch_instruction_overhead
    batched_user = total_user_tokens
    batched_overhead = item_separator_overhead
    batched_total = batched_system + batched_user + batched_overhead

    # Calculate savings
    system_savings_pct = (1 - batched_system / sequential_system) * 100 if sequential_system > 0 else 0
    total_savings_pct = (1 - batched_total / sequential_total) * 100 if sequential_total > 0 else 0
    tokens_saved = sequential_total - batched_total

    return TokenAnalysis(
        sequential_system_tokens=sequential_system,
        sequential_user_tokens=sequential_user,
        sequential_total_input=sequential_total,
        batched_system_tokens=batched_system,
        batched_user_tokens=batched_user,
        batched_overhead_tokens=batched_overhead,
        batched_total_input=batched_total,
        system_prompt_savings_pct=system_savings_pct,
        total_savings_pct=total_savings_pct,
        tokens_saved=tokens_saved,
    )


def analyze_update_prompts(
    prompts: List[List[Dict[str, str]]],
    batch_size: int = None,
) -> Dict[str, Any]:
    """
    Analyze token usage for update prompts.

    Note: Update prompts have dynamic content (current memory state),
    so batching is more complex and may not provide the same savings.

    Returns:
        Dict with analysis results
    """
    if batch_size:
        prompts = prompts[:batch_size]

    n = len(prompts)
    if n == 0:
        return {"count": 0, "total_tokens": 0, "avg_tokens": 0}

    # Update prompts typically don't have a shared system prompt
    # Each contains the full instruction + current memory state
    token_counts = []
    for prompt_msgs in prompts:
        total = sum(count_tokens(msg.get("content", "")) for msg in prompt_msgs)
        token_counts.append(total)

    return {
        "count": n,
        "total_tokens": sum(token_counts),
        "avg_tokens": sum(token_counts) / n,
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "note": "Update prompts contain dynamic memory state, batching savings limited",
    }


def run_token_analysis(prompts_file: str, num_requests: int = 15) -> Dict[str, Any]:
    """
    Run complete token analysis on prompts file.

    Args:
        prompts_file: Path to prompts_storage.json
        num_requests: Number of requests to analyze

    Returns:
        Dict with complete analysis results
    """
    extraction_prompts, update_prompts = load_prompts(prompts_file)

    extraction_analysis = analyze_extraction_prompts(extraction_prompts, num_requests)
    update_analysis = analyze_update_prompts(update_prompts, num_requests)

    return {
        "extraction": extraction_analysis,
        "update": update_analysis,
        "summary": {
            "num_requests": num_requests,
            "extraction_savings_pct": extraction_analysis.total_savings_pct,
            "extraction_tokens_saved": extraction_analysis.tokens_saved,
        },
    }
