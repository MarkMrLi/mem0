"""
Memory searcher for LongMemEval with multi-top_k support and performance tracking.
"""
import os
import time
import json
import requests
import tiktoken
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from jinja2 import Template


ANSWER_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to conversation memories that contain timestamped information relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.), 
   calculate the actual date based on the memory timestamp. For example, if a memory from 
   4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example, 
   convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
   timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories. Do not confuse character names mentioned in 
   memories with the actual users who created those memories.
8. The answer should be less than 5-6 words.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Memories:

{{memories}}

Question: {{question}}

Answer:
"""


class LongMemEvalMemorySearch:
    """Search LongMemEval memories with multi-top_k support and performance tracking."""
    
    def __init__(self, base_url: str = None, model: str = None, embedding_model: str = None):
        """
        Initialize LongMemEval memory searcher.
        
        Args:
            base_url: mem0 API base URL
            model: LLM model for response generation
            embedding_model: Embedding model for token estimation
        """
        self.base_url = base_url or os.getenv("MEM0_BASE_URL", "http://127.0.0.1:7000")
        self.model = model or os.getenv("MODEL", "gpt-4o-mini")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        # Initialize clients
        self.openai_client = OpenAI(base_url=os.getenv("VLLM_BASE_URL"))
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def _search_memories(
        self, 
        user_id: str, 
        query: str, 
        top_k: int,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Search memories from mem0 API.
        
        Args:
            user_id: User identifier
            query: Search query
            top_k: Number of memories to retrieve
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
            
        Returns:
            Tuple of (memories, search_time)
        """
        start_time = time.time()
        retries = 0
        
        while retries < max_retries:
            try:
                data = {
                    "query": query,
                    "user_id": user_id,
                    "top_k": top_k
                }
                
                response = requests.post(
                    f"{self.base_url}/search", 
                    json=data, 
                    timeout=300
                )
                response.raise_for_status()
                memories_data = response.json()
                break
                
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)
        
        search_time = time.time() - start_time
        
        # Process memories
        results_data = memories_data.get("results", [])
        memories = [
            {
                "memory": memory["memory"],
                "timestamp": memory.get("metadata", {}).get("timestamp", ""),
                "score": round(memory["score"], 2),
            }
            for memory in results_data
        ]
        
        return memories, search_time
    
    def _generate_response(
        self, 
        query: str, 
        memories: List[Dict[str, Any]],
        max_retries: int = 3
    ) -> Tuple[str, float, Dict[str, int]]:
        """
        Generate response based on retrieved memories.
        
        Args:
            query: User query
            memories: Retrieved memories
            max_retries: Maximum retry attempts
            
        Returns:
            Tuple of (response, response_time, token_counts)
        """
        # Format memories for prompt
        formatted_memories = json.dumps(memories, indent=2)
        
        # Create prompt
        template = Template(ANSWER_PROMPT)
        prompt = template.render(memories=formatted_memories, question=query)
        
        # Count input tokens
        system_message = "You are a helpful assistant that can answer questions based on conversation memories."
        input_tokens = self._count_tokens(system_message) + self._count_tokens(prompt)
        
        retries = 0
        while retries <= max_retries:
            try:
                start_time = time.time()
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=1000
                )
                response_time = time.time() - start_time
                
                # Get response and tokens
                answer = response.choices[0].message.content.strip()
                output_tokens = self._count_tokens(answer)
                
                # Get API usage if available
                api_usage = getattr(response, 'usage', None)
                if api_usage:
                    actual_input_tokens = api_usage.prompt_tokens
                    actual_output_tokens = api_usage.completion_tokens
                    total_tokens = api_usage.total_tokens
                else:
                    actual_input_tokens = input_tokens
                    actual_output_tokens = output_tokens
                    total_tokens = input_tokens + output_tokens
                
                token_counts = {
                    "input_tokens": actual_input_tokens,
                    "output_tokens": actual_output_tokens,
                    "total_tokens": total_tokens,
                    "estimated_input_tokens": input_tokens,
                    "estimated_output_tokens": output_tokens
                }
                
                return answer, response_time, token_counts
                
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    raise e
                time.sleep(1)
    
    def search_and_respond(
        self,
        user_id: str,
        query: str,
        top_k_values: List[int] = [1, 5, 10, 20, 30],
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Search and respond with multiple top_k values.
        
        Args:
            user_id: User identifier
            query: Search query
            top_k_values: List of top_k values to test
            max_retries: Maximum retry attempts
            
        Returns:
            List of results for each top_k value
        """
        results = []
        
        for top_k in top_k_values:
            try:
                # Search memories
                memories, search_time = self._search_memories(
                    user_id, query, top_k, max_retries
                )
                
                # Generate response
                answer, response_time, token_counts = self._generate_response(
                    query, memories, max_retries
                )
                
                result = {
                    "top_k": top_k,
                    "search_time_ms": round(search_time * 1000, 2),
                    "response_time_ms": round(response_time * 1000, 2),
                    "total_time_ms": round((search_time + response_time) * 1000, 2),
                    "retrieved_memories_count": len(memories),
                    "answer": answer,
                    "retrieved_memories": memories,
                    "token_counts": token_counts
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing top_k={top_k}: {e}")
                result = {
                    "top_k": top_k,
                    "error": str(e),
                    "search_time_ms": 0,
                    "response_time_ms": 0,
                    "total_time_ms": 0,
                    "retrieved_memories_count": 0,
                    "answer": "",
                    "retrieved_memories": [],
                    "token_counts": {}
                }
                results.append(result)
        
        return results