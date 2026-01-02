import json
import os
import time
from collections import defaultdict

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

    def generate_response(self, question, context):
        template = Template(FULL_CONTEXT_PROMPT)
        prompt = template.render(CONTEXT=context, QUESTION=question)

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
                return response.choices[0].message.content.strip(), t2 - t1
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
                response, response_time = self.generate_response(question, full_context)

                FINAL_RESULTS[key].append(
                    {
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "context": full_context[:100],  # Include full context for reference
                        "response": response,
                        "response_time": response_time,
                    }
                )
                
                # Save incrementally
                with open(output_file_path, "w") as f:
                    json.dump(FINAL_RESULTS, f, indent=4)

        # Final save
        with open(output_file_path, "w") as f:
            json.dump(FINAL_RESULTS, f, indent=4)
        
        print(f"✅ Full context processing complete. Results saved to {output_file_path}")