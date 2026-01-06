"""
Data converter for LongMemEval dataset format to mem0 compatible format.
"""
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime


class LongMemEvalDataConverter:
    """Convert LongMemEval dataset format to mem0 compatible format."""
    
    def __init__(self):
        self.question_type_mapping = {
            "single-session-user": "single_session_user",
            "single-session-assistant": "single_session_assistant", 
            "single-session-preference": "single_session_preference",
            "temporal-reasoning": "temporal_reasoning",
            "knowledge-update": "knowledge_update",
            "multi-session": "multi_session"
        }
    
    def convert_conversation_to_mem0_format(
        self, 
        conversation_data: Dict[str, Any],
        conversation_id: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Convert a single LongMemEval conversation to mem0 message format.
        
        Args:
            conversation_data: Single conversation from LongMemEval dataset
            conversation_id: Unique identifier for this conversation
            
        Returns:
            Tuple of (messages_list, metadata)
        """
        messages = []
        metadata = {
            "conversation_id": conversation_id,
            "question_id": conversation_data.get("question_id", ""),
            "question_type": conversation_data.get("question_type", ""),
            "question_date": conversation_data.get("question_date", ""),
            "original_question": conversation_data.get("question", ""),
            "ground_truth_answer": conversation_data.get("answer", ""),
            "haystack_session_ids": conversation_data.get("haystack_session_ids", []),
            "answer_session_ids": conversation_data.get("answer_session_ids", []),
            "is_abstention": conversation_data.get("question_id", "").endswith("_abs")
        }
        
        haystack_sessions = conversation_data.get("haystack_sessions", [])
        haystack_dates = conversation_data.get("haystack_dates", [])
        
        # Convert each session to mem0 message format
        for session_idx, session in enumerate(haystack_sessions):
            session_date = haystack_dates[session_idx] if session_idx < len(haystack_dates) else ""
            
            for turn in session:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                has_answer = turn.get("has_answer", False)
                
                # Create mem0-style message
                message = {
                    "role": role,
                    "content": content,
                    "metadata": {
                        "session_index": session_idx,
                        "session_date": session_date,
                        "has_answer": has_answer,
                        "conversation_id": conversation_id,
                        "turn_id": f"{conversation_id}_session{session_idx}_turn{len(messages)}"
                    }
                }
                messages.append(message)
        
        return messages, metadata
    
    def generate_user_id(self, conversation_id: str) -> str:
        """Generate unique user ID for a conversation."""
        return f"longmemeval_{conversation_id}"
    
    def normalize_question_type(self, question_type: str) -> str:
        """Normalize question type to standard format."""
        return self.question_type_mapping.get(question_type, question_type)
    
    def convert_dataset(
        self, 
        longmemeval_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert entire LongMemEval dataset to mem0 compatible format.
        
        Args:
            longmemeval_data: List of conversations from LongMemEval dataset
            
        Returns:
            List of converted conversations ready for mem0 processing
        """
        converted_data = []
        
        for idx, conversation in enumerate(longmemeval_data):
            conversation_id = conversation.get("question_id", f"conversation_{idx}")
            
            messages, metadata = self.convert_conversation_to_mem0_format(
                conversation, conversation_id
            )
            
            converted_conversation = {
                "conversation_id": conversation_id,
                "user_id": self.generate_user_id(conversation_id),
                "messages": messages,
                "metadata": metadata,
                "query": metadata["original_question"],
                "ground_truth": metadata["ground_truth_answer"],
                "question_type": self.normalize_question_type(metadata["question_type"]),
                "is_abstention": metadata["is_abstention"]
            }
            
            converted_data.append(converted_conversation)
        
        return converted_data
    
    def load_and_convert_json(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Load LongMemEval JSON file and convert to mem0 format.
        
        Args:
            json_path: Path to LongMemEval JSON file
            
        Returns:
            List of converted conversations
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            longmemeval_data = json.load(f)
        
        return self.convert_dataset(longmemeval_data)
    
    def get_statistics(
        self, 
        converted_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate statistics about converted dataset.
        
        Args:
            converted_data: List of converted conversations
            
        Returns:
            Dictionary with dataset statistics
        """
        total_conversations = len(converted_data)
        total_messages = sum(len(conv["messages"]) for conv in converted_data)
        
        question_types = {}
        abstention_count = 0
        
        for conv in converted_data:
            q_type = conv["question_type"]
            question_types[q_type] = question_types.get(q_type, 0) + 1
            
            if conv["is_abstention"]:
                abstention_count += 1
        
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "average_messages_per_conversation": total_messages / total_conversations if total_conversations > 0 else 0,
            "question_type_distribution": question_types,
            "abstention_questions": abstention_count,
            "regular_questions": total_conversations - abstention_count
        }