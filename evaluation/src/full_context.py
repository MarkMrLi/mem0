import json
import os
import time
from collections import defaultdict

import tiktoken
from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

FULL_CONTEXT_PROMPT = """
# Complete Conversation Context:
{{CONTEXT}}

# Question: 
{{QUESTION}}

# Short answer:
"""


class FullContextProcessor:
    def __init__(self, data_path="dataset/locomo10.json"):
        self.model = os.getenv("MODEL")
        self.client = OpenAI(base_url=os.getenv("VLLM_BASE_URL"))
        self.data_path = data_path
        
        # 初始化tokenizer用于token计数
        try:
            # 尝试使用指定的模型
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # 如果模型不在tiktoken中，使用默认的cl100k_base编码
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def generate_response(self, question, context):
        template = Template(FULL_CONTEXT_PROMPT)
        prompt = template.render(CONTEXT=context, QUESTION=question)

        # 计算输入token数量
        system_message = "You are a helpful assistant that can answer questions based on the provided complete conversation context.If the question involves timing, use the conversation date for reference.Provide the shortest possible answer.Use words directly from the conversation when possible.Avoid using subjects in your answer."
        
        input_tokens = len(self.tokenizer.encode(system_message)) + len(self.tokenizer.encode(prompt))

        max_retries = 3
        retries = 0

        while retries <= max_retries:
            try:
                t1 = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that can answer "
                            "questions based on the provided complete conversation context."
                            "If the question involves timing, use the conversation date for reference."
                            "Provide the shortest possible answer."
                            "Use words directly from the conversation when possible."
                            "Avoid using subjects in your answer.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=1000
                )
                t2 = time.time()
                
                # 计算输出token数量
                output_content = response.choices[0].message.content.strip()
                output_tokens = len(self.tokenizer.encode(output_content))
                
                # 获取实际的API返回的token使用情况（如果可用）
                api_usage = getattr(response, 'usage', None)
                if api_usage:
                    actual_input_tokens = api_usage.prompt_tokens
                    actual_output_tokens = api_usage.completion_tokens
                    total_tokens = api_usage.total_tokens
                else:
                    actual_input_tokens = input_tokens
                    actual_output_tokens = output_tokens
                    total_tokens = input_tokens + output_tokens
                
                return (
                    output_content, 
                    t2 - t1,
                    {
                        "input_tokens": actual_input_tokens,
                        "output_tokens": actual_output_tokens,
                        "total_tokens": total_tokens,
                        "estimated_input_tokens": input_tokens,
                        "estimated_output_tokens": output_tokens
                    }
                )
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    raise e
                time.sleep(1)  # Wait before retrying

    def build_full_context(self, conversation):
        """
        Build complete context from all sessions in the conversation.
        
        Args:
            conversation: Dictionary containing all sessions and speaker info
            
        Returns:
            formatted_context: String containing all conversation history
        """
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]
        
        # Collect all messages from all sessions in chronological order
        all_messages = []
        
        # Find all session keys (excluding speaker info and timestamp keys)
        session_keys = [
            key for key in conversation.keys() 
            if key not in ["speaker_a", "speaker_b"] and 
               not key.endswith("_date_time") and
               not key.endswith("_timestamp")
        ]
        
        # Sort sessions chronologically if there's timestamp information
        for session_key in session_keys:
            date_time_key = f"{session_key}_date_time"
            timestamp = conversation.get(date_time_key, "")
            chats = conversation[session_key]
            
            for chat in chats:
                speaker = chat.get("speaker", "")
                text = chat.get("text", "")
                chat_timestamp = chat.get("timestamp", "")
                
                # Format: timestamp | speaker: text
                if chat_timestamp:
                    message_line = f"{chat_timestamp} | {speaker}: {text}"
                elif timestamp:
                    message_line = f"{timestamp} | {speaker}: {text}"
                else:
                    message_line = f"{speaker}: {text}"
                    
                all_messages.append(message_line)
        
        # Join all messages with newlines
        formatted_context = "\n".join(all_messages)
        
        return formatted_context

    def process_all_conversations(self, output_file_path):
        """
        Process all conversations using full context approach.
        
        Args:
            output_file_path: Path to save the results
        """
        with open(self.data_path, "r") as f:
            data = json.load(f)

        FINAL_RESULTS = defaultdict(list)
        
        # Handle both list and dict formats
        data_items = data if isinstance(data, list) else data.items()
        
        for idx, value in tqdm(enumerate(data_items), total=len(data_items), desc="Processing conversations"):
            # Handle both list items and dict key-value pairs
            if isinstance(data, list):
                conversation = value["conversation"]
                questions = value.get("qa", [])
                key = str(idx)
            else:
                key, value_data = value
                conversation = value_data["conversation"]
                questions = value_data.get("qa", [])
            
            # Build complete context from all sessions
            full_context = self.build_full_context(conversation)
            
            for item in tqdm(questions, desc="Answering questions", leave=False):
                question = item["question"]
                answer = item.get("answer", "")
                category = item.get("category", "")
                
                # Generate response using full context
                response, response_time, token_info = self.generate_response(question, full_context)

                FINAL_RESULTS[key].append(
                    {
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "context": full_context[:100],  # Include full context for reference
                        "response": response,
                        "response_time": response_time,
                        # Token使用信息
                        "input_tokens": token_info["input_tokens"],
                        "output_tokens": token_info["output_tokens"],
                        "total_tokens": token_info["total_tokens"],
                        "estimated_input_tokens": token_info["estimated_input_tokens"],
                        "estimated_output_tokens": token_info["estimated_output_tokens"],
                        # 上下文长度信息
                        "context_length": len(full_context),
                        "context_chars": len(full_context)
                    }
                )
                
                # Save incrementally
                with open(output_file_path, "w") as f:
                    json.dump(FINAL_RESULTS, f, indent=4)

        # Final save
        with open(output_file_path, "w") as f:
            json.dump(FINAL_RESULTS, f, indent=4)
        
        # 生成token使用统计
        self._print_token_statistics(FINAL_RESULTS)
        
        print(f"✅ Full context processing complete. Results saved to {output_file_path}")

    def _print_token_statistics(self, results):
        """打印token使用统计信息"""
        print("\n" + "=" * 50)
        print("📊 Token使用统计")
        print("=" * 50)
        
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        total_context_chars = 0
        question_count = 0
        
        for conv_id, questions in results.items():
            for qa in questions:
                total_input_tokens += qa.get("input_tokens", 0)
                total_output_tokens += qa.get("output_tokens", 0)
                total_tokens += qa.get("total_tokens", 0)
                total_context_chars += qa.get("context_chars", 0)
                question_count += 1
        
        if question_count == 0:
            print("❌ 没有找到任何问题")
            return
        
        avg_input_tokens = total_input_tokens / question_count
        avg_output_tokens = total_output_tokens / question_count
        avg_total_tokens = total_tokens / question_count
        avg_context_chars = total_context_chars / question_count
        
        print(f"📈 总问题数: {question_count}")
        print(f"🔤 总输入tokens: {total_input_tokens:,}")
        print(f"🔤 总输出tokens: {total_output_tokens:,}")
        print(f"🔤 总tokens: {total_tokens:,}")
        print(f"📝 平均输入tokens: {avg_input_tokens:.1f}")
        print(f"📝 平均输出tokens: {avg_output_tokens:.1f}")
        print(f"📝 平均总tokens: {avg_total_tokens:.1f}")
        print(f"📄 平均上下文字符数: {avg_context_chars:.1f}")
        
        # 计算字符到token的比例
        if avg_context_chars > 0 and avg_input_tokens > 0:
            chars_per_token = avg_context_chars / avg_input_tokens
            print(f"📊 字符/token比例: {chars_per_token:.2f}")
        
        print("=" * 50)