"""
Prompt batcher for combining multiple LLM requests into a single batched request.
Reduces token usage by sharing system prompts across multiple inputs.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# Batch instruction to add to system prompt
BATCH_INSTRUCTION = """

IMPORTANT: You will receive multiple inputs, each marked with [Item N]. 
Process each input independently and return results in the following JSON format:
{
    "results": [
        {"item_id": 1, "facts": [...]},
        {"item_id": 2, "facts": [...]},
        ...
    ]
}
Return ONLY the JSON object, no additional text."""


@dataclass
class BatchedPrompt:
    """A batched prompt combining multiple requests."""

    messages: List[Dict[str, str]]
    item_count: int
    original_indices: List[int]  # Track original order for result mapping


@dataclass
class BatchedResult:
    """Result from a batched request, parsed into individual results."""

    item_id: int
    original_index: int
    facts: List[str]
    success: bool
    error: Optional[str] = None


def create_batched_extraction_prompt(
    prompts: List[List[Dict[str, str]]],
    max_items_per_batch: int = 20,
) -> List[BatchedPrompt]:
    """
    Combine multiple extraction prompts into batched prompts.

    Args:
        prompts: List of message lists, each with system and user messages
        max_items_per_batch: Maximum items per batch (to stay within context limits)

    Returns:
        List of BatchedPrompt objects
    """
    if not prompts:
        return []

    # Extract shared system prompt and individual user contents
    system_prompt = ""
    user_contents = []

    for prompt_msgs in prompts:
        for msg in prompt_msgs:
            if msg.get("role") == "system" and not system_prompt:
                system_prompt = msg.get("content", "")
            elif msg.get("role") == "user":
                user_contents.append(msg.get("content", ""))

    # Create batched prompts
    batched_prompts = []

    for batch_start in range(0, len(user_contents), max_items_per_batch):
        batch_end = min(batch_start + max_items_per_batch, len(user_contents))
        batch_contents = user_contents[batch_start:batch_end]
        batch_indices = list(range(batch_start, batch_end))

        # Combine user contents with item markers
        combined_user_content = "\n\n".join([f"[Item {i + 1}]:\n{content}" for i, content in enumerate(batch_contents)])

        # Create batched messages
        messages = [
            {"role": "system", "content": system_prompt + BATCH_INSTRUCTION},
            {"role": "user", "content": combined_user_content},
        ]

        batched_prompts.append(
            BatchedPrompt(
                messages=messages,
                item_count=len(batch_contents),
                original_indices=batch_indices,
            )
        )

    return batched_prompts


def parse_batched_response(
    response_content: str,
    batched_prompt: BatchedPrompt,
) -> List[BatchedResult]:
    """
    Parse a batched response into individual results.

    Args:
        response_content: JSON response from LLM
        batched_prompt: The original batched prompt (for index mapping)

    Returns:
        List of BatchedResult objects
    """
    results = []

    try:
        data = json.loads(response_content)

        # Handle both {"results": [...]} and direct [...] format
        if isinstance(data, dict) and "results" in data:
            items = data["results"]
        elif isinstance(data, list):
            items = data
        else:
            # Single result format
            items = [data]

        # Map results to original indices
        for item in items:
            item_id = item.get("item_id", 0)

            # item_id is 1-indexed in our format
            if 1 <= item_id <= len(batched_prompt.original_indices):
                original_index = batched_prompt.original_indices[item_id - 1]
            else:
                # Try to infer from position
                idx = items.index(item)
                if idx < len(batched_prompt.original_indices):
                    original_index = batched_prompt.original_indices[idx]
                else:
                    original_index = -1

            results.append(
                BatchedResult(
                    item_id=item_id,
                    original_index=original_index,
                    facts=item.get("facts", []),
                    success=True,
                )
            )

        # Check for missing items
        found_indices = {r.original_index for r in results}
        for idx in batched_prompt.original_indices:
            if idx not in found_indices:
                results.append(
                    BatchedResult(
                        item_id=-1,
                        original_index=idx,
                        facts=[],
                        success=False,
                        error="Missing from response",
                    )
                )

    except json.JSONDecodeError as e:
        # If JSON parsing fails, return error results for all items
        for idx in batched_prompt.original_indices:
            results.append(
                BatchedResult(
                    item_id=-1,
                    original_index=idx,
                    facts=[],
                    success=False,
                    error=f"JSON parse error: {str(e)}",
                )
            )

    # Sort by original index
    results.sort(key=lambda r: r.original_index)

    return results


def unbatch_results(
    batched_results: List[List[BatchedResult]],
    total_items: int,
) -> List[Dict[str, Any]]:
    """
    Flatten batched results back to original order.

    Args:
        batched_results: List of result lists from each batch
        total_items: Total number of original items

    Returns:
        List of result dicts in original order
    """
    # Create result array
    results = [None] * total_items

    for batch_results in batched_results:
        for result in batch_results:
            if 0 <= result.original_index < total_items:
                results[result.original_index] = {
                    "facts": result.facts,
                    "success": result.success,
                    "error": result.error,
                }

    # Fill any missing with errors
    for i, r in enumerate(results):
        if r is None:
            results[i] = {
                "facts": [],
                "success": False,
                "error": "Result not found",
            }

    return results


def estimate_batched_tokens(
    prompts: List[List[Dict[str, str]]],
    max_items_per_batch: int = 20,
) -> Dict[str, int]:
    """
    Estimate token usage for batched vs sequential approach.

    Returns:
        Dict with token estimates
    """
    # Support both direct run and package import
    try:
        from .token_analyzer import count_tokens
    except ImportError:
        from token_analyzer import count_tokens

    if not prompts:
        return {"sequential": 0, "batched": 0, "savings": 0}

    # Get system prompt tokens
    system_prompt = ""
    user_contents = []

    for prompt_msgs in prompts:
        for msg in prompt_msgs:
            if msg.get("role") == "system" and not system_prompt:
                system_prompt = msg.get("content", "")
            elif msg.get("role") == "user":
                user_contents.append(msg.get("content", ""))

    n = len(user_contents)
    system_tokens = count_tokens(system_prompt)
    user_tokens = sum(count_tokens(c) for c in user_contents)

    # Sequential: N * system + user
    sequential = n * system_tokens + user_tokens

    # Batched: ceil(N / batch_size) * (system + batch_instruction) + user + overhead
    num_batches = (n + max_items_per_batch - 1) // max_items_per_batch
    batch_instruction_tokens = count_tokens(BATCH_INSTRUCTION)
    item_marker_tokens = count_tokens("[Item 1]:\n") * n

    batched = num_batches * (system_tokens + batch_instruction_tokens) + user_tokens + item_marker_tokens

    return {
        "sequential": sequential,
        "batched": batched,
        "savings": sequential - batched,
        "savings_pct": (1 - batched / sequential) * 100 if sequential > 0 else 0,
    }
