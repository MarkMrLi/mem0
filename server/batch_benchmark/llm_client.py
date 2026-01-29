"""
Async OpenAI-compatible LLM client for batch benchmark.
Supports both real API calls and mock mode for testing.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Disable proxy for localhost connections
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

# OpenAI is optional - only needed for real API calls
try:
    from openai import AsyncOpenAI
    import httpx

    HAS_OPENAI = True
except ImportError:
    AsyncOpenAI = None
    httpx = None
    HAS_OPENAI = False

# Support both direct run and package import
try:
    from .config import LLMConfig
except ImportError:
    from config import LLMConfig


@dataclass
class LLMResponse:
    """Structured response from LLM call."""

    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    success: bool
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMClient:
    """Async OpenAI-compatible client with metrics collection."""

    config: LLMConfig
    mock_mode: bool = False
    mock_delay_ms: float = 500.0
    _client: Any = field(default=None, init=False)

    def __post_init__(self):
        if not self.mock_mode:
            if not HAS_OPENAI:
                raise ImportError("openai package is required for real API calls. Install with: pip install openai")

            # Create httpx client with proxy disabled for localhost
            http_client = httpx.AsyncClient(
                proxy=None,  # Explicitly disable proxy
                verify=True,
            )

            self._client = AsyncOpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                http_client=http_client,
            )

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            response_format: Optional response format (e.g., {"type": "json_object"})

        Returns:
            LLMResponse with content, token counts, and latency
        """
        start_time = time.perf_counter()

        if self.mock_mode:
            return await self._mock_completion(messages, start_time)

        try:
            params = {
                "model": self.config.model,
                "messages": messages,
            }
            if response_format:
                params["response_format"] = response_format

            response = await self._client.chat.completions.create(**params)

            latency_ms = (time.perf_counter() - start_time) * 1000

            return LLMResponse(
                content=response.choices[0].message.content or "",
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                latency_ms=latency_ms,
                success=True,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return LLMResponse(
                content="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )

    async def _mock_completion(
        self,
        messages: List[Dict[str, str]],
        start_time: float,
    ) -> LLMResponse:
        """Generate a mock response for testing."""
        await asyncio.sleep(self.mock_delay_ms / 1000)

        # Generate mock response based on message content
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        # Check if this is a batched request (contains multiple items)
        if "[Item " in user_content:
            # Count items and generate mock batched response
            import re

            items = re.findall(r"\[Item (\d+)\]", user_content)
            results = [{"item_id": int(i), "facts": [f"Mock fact for item {i}"]} for i in items]
            mock_content = json.dumps({"results": results})
        else:
            # Single item response
            mock_content = json.dumps({"facts": ["Mock extracted fact"]})

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Estimate tokens (rough approximation)
        input_tokens = sum(len(m.get("content", "")) // 4 for m in messages)
        output_tokens = len(mock_content) // 4

        return LLMResponse(
            content=mock_content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            success=True,
        )

    async def batch_chat_completion(
        self,
        messages_list: List[List[Dict[str, str]]],
        response_format: Optional[Dict[str, str]] = None,
        max_concurrent: int = 10,
    ) -> List[LLMResponse]:
        """
        Send multiple chat completion requests concurrently.

        Args:
            messages_list: List of message lists
            response_format: Optional response format
            max_concurrent: Maximum concurrent requests

        Returns:
            List of LLMResponse objects in the same order as input
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_request(messages: List[Dict[str, str]]) -> LLMResponse:
            async with semaphore:
                return await self.chat_completion(messages, response_format)

        tasks = [limited_request(messages) for messages in messages_list]
        return await asyncio.gather(*tasks)
