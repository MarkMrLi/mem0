"""
Memory adder for LongMemEval with performance tracking.
"""
import os
import time
import asyncio
import httpx
from typing import List, Dict, Any, Tuple
from tqdm import tqdm


class LongMemEvalMemoryAdder:
    """Add LongMemEval conversations to mem0 with performance tracking."""
    
    def __init__(self, base_url: str = None, batch_size: int = 5):
        """
        Initialize LongMemEval memory adder.
        
        Args:
            base_url: mem0 API base URL
            batch_size: Number of messages to add per batch
        """
        self.base_url = base_url or os.getenv("MEM0_BASE_URL", "http://127.0.0.1:7000")
        self.batch_size = batch_size
    
    def _count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Estimate token count for messages (rough estimation).
        
        Args:
            messages: List of messages
            
        Returns:
            Estimated token count
        """
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        # Rough estimation: 1 token ≈ 4 characters
        return total_chars // 4
    
    async def _async_add_messages(
        self, 
        client: httpx.AsyncClient,
        user_id: str, 
        messages: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        semaphore: asyncio.Semaphore,
        retries: int = 3
    ) -> Tuple[bool, float, int]:
        """
        Async add messages to mem0 with retry logic.
        
        Args:
            client: HTTP client
            user_id: User identifier
            messages: Messages to add
            metadata: Additional metadata
            semaphore: Semaphore for concurrency control
            retries: Number of retry attempts
            
        Returns:
            Tuple of (success, time_taken, token_count)
        """
        async with semaphore:
            data = {
                "messages": messages if isinstance(messages, list) else [messages],
                "user_id": user_id,
                "metadata": metadata
            }
            
            token_count = self._count_tokens(messages)
            start_time = time.time()
            
            for attempt in range(retries):
                try:
                    response = await client.post(
                        f"{self.base_url}/memories",
                        json=data,
                        timeout=300.0
                    )
                    response.raise_for_status()
                    time_taken = time.time() - start_time
                    
                    return True, time_taken, token_count
                    
                except Exception as e:
                    if attempt < retries - 1:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    else:
                        print(f"Failed to add memories for {user_id}: {e}")
                        return False, 0, token_count
    
    async def add_conversation_async(
        self,
        conversation_data: Dict[str, Any],
        max_concurrent: int = 3,
        progress_bar: bool = True
    ) -> Dict[str, Any]:
        """
        Add a single conversation to mem0 asynchronously.
        
        Args:
            conversation_data: Converted conversation data
            max_concurrent: Maximum concurrent requests
            progress_bar: Whether to show progress bar
            
        Returns:
            Performance metrics dictionary
        """
        user_id = conversation_data["user_id"]
        messages = conversation_data["messages"]
        conversation_id = conversation_data["conversation_id"]
        
        # Split messages into batches
        message_batches = []
        for i in range(0, len(messages), self.batch_size):
            batch = messages[i:i + self.batch_size]
            message_batches.append(batch)
        
        # Prepare metadata
        base_metadata = conversation_data.get("metadata", {})
        
        # Create async tasks
        semaphore = asyncio.Semaphore(max_concurrent)
        limits = httpx.Limits(
            max_keepalive_connections=max_concurrent,
            max_connections=max_concurrent + 2
        )
        
        total_time = 0
        total_tokens = 0
        successful_batches = 0
        failed_batches = 0
        
        async with httpx.AsyncClient(limits=limits, timeout=None) as client:
            tasks = []
            for batch_idx, batch in enumerate(message_batches):
                batch_metadata = {
                    **base_metadata,
                    "batch_index": batch_idx,
                    "total_batches": len(message_batches)
                }
                task = self._async_add_messages(
                    client, user_id, batch, batch_metadata, semaphore
                )
                tasks.append(task)
            
            # Execute with progress bar
            if progress_bar:
                iterator = tqdm(
                    asyncio.as_completed(tasks),
                    total=len(tasks),
                    desc=f"Adding {conversation_id}"
                )
            else:
                iterator = asyncio.as_completed(tasks)
            
            for task in iterator:
                success, time_taken, token_count = await task
                
                if success:
                    total_time += time_taken
                    total_tokens += token_count
                    successful_batches += 1
                else:
                    failed_batches += 1
        
        return {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "add_phase": {
                "total_time_ms": round(total_time * 1000, 2),
                "total_tokens": total_tokens,
                "messages_count": len(messages),
                "batches_count": len(message_batches),
                "successful_batches": successful_batches,
                "failed_batches": failed_batches,
                "average_time_per_batch_ms": round((total_time / len(message_batches)) * 1000, 2) if message_batches else 0,
                "throughput_tokens_per_second": round(total_tokens / total_time, 2) if total_time > 0 else 0
            }
        }
    
    def add_conversation(
        self,
        conversation_data: Dict[str, Any],
        max_concurrent: int = 3
    ) -> Dict[str, Any]:
        """
        Add a single conversation to mem0 (synchronous wrapper).
        
        Args:
            conversation_data: Converted conversation data
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Performance metrics dictionary
        """
        return asyncio.run(
            self.add_conversation_async(conversation_data, max_concurrent)
        )
    
    def add_multiple_conversations(
        self,
        conversations: List[Dict[str, Any]],
        max_concurrent: int = 3,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Add multiple conversations sequentially.
        
        Args:
            conversations: List of converted conversation data
            max_concurrent: Maximum concurrent requests per conversation
            show_progress: Whether to show overall progress
            
        Returns:
            List of performance metrics for each conversation
        """
        results = []
        
        if show_progress:
            iterator = tqdm(conversations, desc="Processing conversations")
        else:
            iterator = conversations
        
        for conversation in iterator:
            result = self.add_conversation(conversation, max_concurrent)
            results.append(result)
        
        return results