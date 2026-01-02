import json
import os
import requests
import time
import threading # 增加：为了多线程安全
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH
from tqdm import tqdm

load_dotenv()


class MemorySearch:
    def __init__(self, output_path="results.json", top_k=10, filter_memories=False, is_graph=False):
        self.base_url = os.getenv("MEM0_BASE_URL", "http://127.0.0.1:7000")
        self.top_k = top_k
        self.openai_client = OpenAI(base_url="http://localhost:8000/v1")
        self.results = defaultdict(list)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph
        self.lock = threading.Lock() # 增加：防止多线程同时写文件导致损坏

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT

    # 增加一个辅助方法，修复原代码中 json.(...) 的语法错误并集中管理写入
    def _save_to_disk(self):
        with self.lock:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=4, ensure_ascii=False)

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        while retries < max_retries:
            try:
                if self.is_graph:
                    data = {
                        "query": query,
                        "user_id": user_id,
                        "top_k": self.top_k,
                        "filter_memories": self.filter_memories,
                        "enable_graph": True,
                        "output_format": "v1.1"
                    }
                else:
                    data = {
                        "query": query,
                        "user_id": user_id,
                        "top_k": self.top_k,
                        "filter_memories": self.filter_memories
                    }

                response = requests.post(f"{self.base_url}/search", json=data, timeout=300)
                response.raise_for_status()
                memories = response.json()
                break
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()
        
        # 统一处理 results 字段，防止 Key 错误
        results_data = memories.get("results", [])
        semantic_memories = [
            {
                "memory": memory["memory"],
                "timestamp": memory.get("metadata", {}).get("timestamp", ""),
                "score": round(memory["score"], 2),
            }
            for memory in results_data
        ]
        
        graph_memories = None
        if self.is_graph:
            graph_memories = [
                {"source": relation["source"], "relationship": relation["relationship"], "target": relation["target"]}
                for relation in memories.get("relations", [])
            ]
            
        return semantic_memories, graph_memories, end_time - start_time

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category):
        speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = self.search_memory(
            speaker_1_user_id, question
        )
        speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = self.search_memory(
            speaker_2_user_id, question
        )

        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        template = Template(self.ANSWER_PROMPT)
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_memory, indent=4),
            speaker_2_memories=json.dumps(search_2_memory, indent=4),
            speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
            speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
            question=question,
        )

        t1 = time.time()
        response = self.openai_client.chat.completions.create(
            model=os.getenv("MODEL"), messages=[{"role": "system", "content": answer_prompt}], temperature=0.0
        )
        response_time = time.time() - t1
        return (
            response.choices[0].message.content,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "speaker_1_graph_memories": speaker_1_graph_memories,
            "speaker_2_graph_memories": speaker_2_graph_memories,
            "response_time": response_time,
        }
        
        # 删除了这里的磁盘写入，改为在 process_data_file 级别控制
        return result

    def process_data_file(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a_user_id = f"{conversation['speaker_a']}_{idx}"
            speaker_b_user_id = f"{conversation['speaker_b']}_{idx}"

            for question_item in qa:
                result = self.process_question(question_item, speaker_a_user_id, speaker_b_user_id)
                self.results[idx].append(result)

            # 修改点：处理完一整个对话后保存一次，而不是每个问题保存一次
            self._save_to_disk()

    def process_questions_parallel(self, qa_list, speaker_a_user_id, speaker_b_user_id, max_workers=1):
        def process_single_question(val):
            return self.process_question(val, speaker_a_user_id, speaker_b_user_id)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # map 后批量获取结果
            qa_results = list(tqdm(executor.map(process_single_question, qa_list), total=len(qa_list), desc="Answering Questions"))
            
        # 并行处理完毕后统一写入
        # 注意：这里需要外部逻辑调用 self.results[idx].extend(qa_results) 或自行处理
        self._save_to_disk()
        return qa_results