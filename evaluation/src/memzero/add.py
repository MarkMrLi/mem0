import json
import os
import asyncio
import httpx
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

load_dotenv()


# Update custom instructions
custom_instructions = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""


class MemoryADD:
    def __init__(self, data_path=None, batch_size=2, is_graph=False):
        self.base_url = os.getenv("MEM0_BASE_URL", "http://127.0.0.1:7000")
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.is_graph = is_graph
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    async def _async_add_single(self, client, user_id, message, metadata, semaphore, retries=3):
        """核心异步请求函数：使用信号量控制真正的并发数"""
        async with semaphore:
            data = {
                "messages": message if isinstance(message, list) else [message],
                "user_id": user_id,
                "metadata": metadata
            }
            for attempt in range(retries):
                try:
                    # 注意：由于新版服务端要做 Fact Extraction，timeout 必须给够
                    response = await client.post(
                        f"{self.base_url}/memories", 
                        json=data, 
                        timeout=300.0
                    )
                    response.raise_for_status()
                    return True
                except Exception as e:
                    if attempt < retries - 1:
                        await asyncio.sleep(2) # 指数退避
                        continue
                    else:
                        print(f"Failed to add memory for {user_id}: {e}")
                        return False

    async def _run_tasks(self, max_concurrent):
        """将原有逻辑拆解为纯异步任务流"""
        tasks_data = []
        # --- 保持你原有的数据拆解逻辑不变 ---
        for idx, item in enumerate(self.data):
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]
            uids = {"A": f"{speaker_a}_{idx}", "B": f"{speaker_b}_{idx}"}

            for key in conversation.keys():
                if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                    continue
                
                date_time_key = key + "_date_time"
                timestamp = conversation.get(date_time_key)
                chats = conversation[key]

                msg_a, msg_b = [], []
                for chat in chats:
                    content = f"{chat['speaker']}: {chat['text']}"
                    if chat["speaker"] == speaker_a:
                        msg_a.append({"role": "user", "content": content})
                        msg_b.append({"role": "assistant", "content": content})
                    else:
                        msg_a.append({"role": "assistant", "content": content})
                        msg_b.append({"role": "user", "content": content})

                # 按 batch_size 切分
                for i in range(0, len(msg_a), self.batch_size):
                    tasks_data.append((uids["A"], msg_a[i : i + self.batch_size], timestamp))
                for i in range(0, len(msg_b), self.batch_size):
                    tasks_data.append((uids["B"], msg_b[i : i + self.batch_size], timestamp))

        # --- 异步执行部分 ---
        semaphore = asyncio.Semaphore(max_concurrent)
        # 限制 HTTP 连接池，防止内核 Socket 溢出
        limits = httpx.Limits(max_keepalive_connections=max_concurrent, max_connections=max_concurrent + 2)
        
        async with httpx.AsyncClient(limits=limits, timeout=None) as client:
            tasks = [
                self._async_add_single(client, uid, msgs, {"timestamp": ts}, semaphore)
                for uid, msgs, ts in tasks_data
            ]
            # 使用 tqdm 监控进度
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Memories"):
                await f

    def process_all_conversations(self, max_workers=3):
        """对外接口保持一致，内部切换为异步事件循环"""
        if not self.data:
            raise ValueError("No data loaded.")
        
        # 核心改动：在同步方法内启动异步循环
        asyncio.run(self._run_tasks(max_workers))
        print("Messages added successfully")