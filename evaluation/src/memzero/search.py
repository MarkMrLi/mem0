import json
import os
import requests
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm

# 假设这些是从你的 prompts.py 导入的
# from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH

load_dotenv()

class MemorySearch:
    def __init__(self, output_path="results.jsonl", top_k=10, filter_memories=False, is_graph=False):
        self.base_url = os.getenv("MEM0_BASE_URL", "http://127.0.0.1:7000")
        self.top_k = top_k
        self.openai_client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "no-key")
        )
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph
        
        # 使用锁来保证多线程写入文件时的安全
        self.file_lock = threading.Lock()
        
        # 结果缓存（如果仍需要在内存中保留一份）
        self.results = defaultdict(list)

        # 这里的提示词逻辑，实际运行时请确保 prompts 已正确导入
        try:
            from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH
            self.ANSWER_PROMPT_TEMPLATE = ANSWER_PROMPT_GRAPH if self.is_graph else ANSWER_PROMPT
        except ImportError:
            self.ANSWER_PROMPT_TEMPLATE = "Question: {{question}}" # 降级处理

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        while retries < max_retries:
            try:
                data = {
                    "query": query,
                    "user_id": user_id,
                    "top_k": self.top_k,
                    "filter_memories": self.filter_memories
                }
                if self.is_graph:
                    data.update({"enable_graph": True, "output_format": "v1.1"})

                response = requests.post(f"{self.base_url}/search", json=data, timeout=300)
                response.raise_for_status()
                memories = response.json()
                break
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    print(f"Search failed after {max_retries} retries for user {user_id}: {e}")
                    return [], [] if self.is_graph else None, time.time() - start_time
                time.sleep(retry_delay)

        end_time = time.time()
        
        # 解析 Semantic Memories
        results_list = memories.get("results", [])
        semantic_memories = [
            {
                "memory": m["memory"],
                "timestamp": m.get("metadata", {}).get("timestamp", "N/A"),
                "score": round(m.get("score", 0), 2),
            }
            for m in results_list
        ]

        # 解析 Graph Memories
        graph_memories = None
        if self.is_graph:
            graph_memories = [
                {"source": r["source"], "relationship": r["relationship"], "target": r["target"]}
                for r in memories.get("relations", [])
            ]
            
        return semantic_memories, graph_memories, end_time - start_time

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question):
        # 并发获取两个用户的记忆可以进一步优化性能，这里保持逻辑清晰
        s1_mem, s1_graph, s1_time = self.search_memory(speaker_1_user_id, question)
        s2_mem, s2_graph, s2_time = self.search_memory(speaker_2_user_id, question)

        search_1_text = [f"{item['timestamp']}: {item['memory']}" for item in s1_mem]
        search_2_text = [f"{item['timestamp']}: {item['memory']}" for item in s2_mem]

        template = Template(self.ANSWER_PROMPT_TEMPLATE)
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_text, indent=4, ensure_ascii=False),
            speaker_2_memories=json.dumps(search_2_text, indent=4, ensure_ascii=False),
            speaker_1_graph_memories=json.dumps(s1_graph, indent=4, ensure_ascii=False) if s1_graph else "[]",
            speaker_2_graph_memories=json.dumps(s2_graph, indent=4, ensure_ascii=False) if s2_graph else "[]",
            question=question,
        )

        t1 = time.time()
        response = self.openai_client.chat.completions.create(
            model=os.getenv("MODEL", "gpt-4"),
            messages=[{"role": "system", "content": answer_prompt}],
            temperature=0.0
        )
        response_time = time.time() - t1
        
        return (
            response.choices[0].message.content,
            s1_mem, s2_mem, s1_time, s2_time,
            s1_graph, s2_graph, response_time
        )

    def _save_result_to_file(self, result):
        """线程安全的单条结果写入 (JSONL 格式)"""
        with self.file_lock:
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id, conv_idx):
        question = val.get("question", "")
        
        (
            response, s1_mem, s2_mem, 
            s1_time, s2_time, s1_graph, s2_graph, 
            resp_time
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question)

        result = {
            "conv_idx": conv_idx,
            "question": question,
            "answer": val.get("answer", ""),
            "category": val.get("category", -1),
            "evidence": val.get("evidence", []),
            "response": response,
            "adversarial_answer": val.get("adversarial_answer", ""),
            "speaker_1_memories": s1_mem,
            "speaker_2_memories": s2_mem,
            "speaker_1_memory_time": s1_time,
            "speaker_2_memory_time": s2_time,
            "speaker_1_graph_memories": s1_graph,
            "speaker_2_graph_memories": s2_graph,
            "response_time": resp_time,
        }

        # 即时写入磁盘，不再全量重写 self.results
        self._save_result_to_file(result)
        return result

    def process_data_file(self, file_path, max_workers=5):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            qa_list = item["qa"]
            conv = item["conversation"]
            speaker_a_id = f"{conv['speaker_a']}_{idx}"
            speaker_b_id = f"{conv['speaker_b']}_{idx}"

            # 使用多线程处理当前对话下的所有问题
            self.process_questions_parallel(qa_list, speaker_a_id, speaker_b_id, idx, max_workers)

    def process_questions_parallel(self, qa_list, speaker_a_id, speaker_b_id, conv_idx, max_workers):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 使用 list 强行触发 map 的执行
            list(tqdm(
                executor.map(
                    lambda q: self.process_question(q, speaker_a_id, speaker_b_id, conv_idx), 
                    qa_list
                ),
                total=len(qa_list),
                desc=f"Conv {conv_idx}",
                leave=False
            ))

if __name__ == "__main__":
    # 使用示例
    searcher = MemorySearch(output_path="eval_results.jsonl", is_graph=True)
    # searcher.process_data_file("input_data.json", max_workers=10)