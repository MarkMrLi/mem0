"""
Performance evaluator for LongMemEval with comprehensive metrics collection.
"""
import json
from typing import List, Dict, Any
from collections import defaultdict
import pandas as pd


class PerformanceEvaluator:
    """Evaluate LongMemEval performance with comprehensive metrics."""
    
    def __init__(self):
        """Initialize performance evaluator."""
        self.results = []
        self.metrics_by_category = defaultdict(list)
        self.metrics_by_top_k = defaultdict(list)
    
    def evaluate_conversation(
        self,
        conversation_data: Dict[str, Any],
        add_metrics: Dict[str, Any],
        search_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate a single conversation's performance.
        
        Args:
            conversation_data: Original conversation data
            add_metrics: Metrics from memory addition phase
            search_results: Results from search phase for multiple top_k values
            
        Returns:
            Complete evaluation result for this conversation
        """
        result = {
            "conversation_id": conversation_data["conversation_id"],
            "question_type": conversation_data["question_type"],
            "question": conversation_data["query"],
            "ground_truth": conversation_data["ground_truth"],
            "is_abstention": conversation_data["is_abstention"],
            "add_phase": add_metrics["add_phase"],
            "search_response_phase": search_results
        }
        
        self.results.append(result)
        
        # Categorize metrics
        for search_result in search_results:
            top_k = search_result["top_k"]
            self.metrics_by_top_k[top_k].append({
                "conversation_id": conversation_data["conversation_id"],
                "question_type": conversation_data["question_type"],
                **search_result
            })
        
        return result
    
    def calculate_quality_metrics(
        self,
        predicted_answer: str,
        ground_truth: str
    ) -> Dict[str, float]:
        """
        Calculate quality metrics (placeholder for now).
        
        Args:
            predicted_answer: Model's answer
            ground_truth: Ground truth answer
            
        Returns:
            Dictionary with quality metrics
        """
        # This will be implemented later with BLEU, F1, LLM judge
        return {
            "bleu_score": 0.0,
            "f1_score": 0.0,
            "llm_judge": 0
        }
    
    def generate_performance_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance summary.
        
        Returns:
            Dictionary with performance summaries by different dimensions
        """
        summary = {
            "overall_summary": self._generate_overall_summary(),
            "by_top_k": self._generate_top_k_summary(),
            "by_question_type": self._generate_question_type_summary(),
            "performance_comparison": self._generate_performance_comparison()
        }
        
        return summary
    
    def _generate_overall_summary(self) -> Dict[str, Any]:
        """Generate overall performance summary."""
        total_conversations = len(self.results)
        
        add_times = [r["add_phase"]["total_time_ms"] for r in self.results]
        add_tokens = [r["add_phase"]["total_tokens"] for r in self.results]
        
        return {
            "total_conversations": total_conversations,
            "add_phase": {
                "average_time_ms": sum(add_times) / len(add_times) if add_times else 0,
                "total_tokens": sum(add_tokens),
                "average_tokens_per_conversation": sum(add_tokens) / len(add_tokens) if add_tokens else 0
            }
        }
    
    def _generate_top_k_summary(self) -> Dict[str, Any]:
        """Generate performance summary by top_k values."""
        summary = {}
        
        for top_k, results in self.metrics_by_top_k.items():
            search_times = [r["search_time_ms"] for r in results]
            response_times = [r["response_time_ms"] for r in results]
            total_times = [r["total_time_ms"] for r in results]
            input_tokens = [r.get("token_counts", {}).get("input_tokens", 0) for r in results]
            output_tokens = [r.get("token_counts", {}).get("output_tokens", 0) for r in results]
            total_tokens = [r.get("token_counts", {}).get("total_tokens", 0) for r in results]
            
            summary[f"top_k_{top_k}"] = {
                "search": {
                    "average_time_ms": sum(search_times) / len(search_times) if search_times else 0,
                    "min_time_ms": min(search_times) if search_times else 0,
                    "max_time_ms": max(search_times) if search_times else 0
                },
                "response": {
                    "average_time_ms": sum(response_times) / len(response_times) if response_times else 0,
                    "min_time_ms": min(response_times) if response_times else 0,
                    "max_time_ms": max(response_times) if response_times else 0
                },
                "total": {
                    "average_time_ms": sum(total_times) / len(total_times) if total_times else 0,
                    "min_time_ms": min(total_times) if total_times else 0,
                    "max_time_ms": max(total_times) if total_times else 0
                },
                "tokens": {
                    "average_input_tokens": sum(input_tokens) / len(input_tokens) if input_tokens else 0,
                    "average_output_tokens": sum(output_tokens) / len(output_tokens) if output_tokens else 0,
                    "average_total_tokens": sum(total_tokens) / len(total_tokens) if total_tokens else 0,
                    "total_input_tokens": sum(input_tokens),
                    "total_output_tokens": sum(output_tokens),
                    "total_tokens": sum(total_tokens)
                },
                "count": len(results)
            }
        
        return summary
    
    def _generate_question_type_summary(self) -> Dict[str, Any]:
        """Generate performance summary by question type."""
        question_types = defaultdict(list)
        
        for result in self.results:
            q_type = result["question_type"]
            question_types[q_type].append(result)
        
        summary = {}
        for q_type, results in question_types.items():
            # Aggregate search results across all top_k values
            all_search_results = []
            for r in results:
                all_search_results.extend(r["search_response_phase"])
            
            total_times = [sr["total_time_ms"] for sr in all_search_results]
            total_tokens = [sr.get("token_counts", {}).get("total_tokens", 0) for sr in all_search_results]
            
            summary[q_type] = {
                "conversation_count": len(results),
                "average_total_time_ms": sum(total_times) / len(total_times) if total_times else 0,
                "average_total_tokens": sum(total_tokens) / len(total_tokens) if total_tokens else 0,
                "total_conversations": len(results)
            }
        
        return summary
    
    def _generate_performance_comparison(self) -> Dict[str, Any]:
        """Generate performance comparison between top_k values."""
        comparison = {}
        
        for top_k, results in self.metrics_by_top_k.items():
            avg_search_time = sum(r["search_time_ms"] for r in results) / len(results) if results else 0
            avg_response_time = sum(r["response_time_ms"] for r in results) / len(results) if results else 0
            avg_total_time = sum(r["total_time_ms"] for r in results) / len(results) if results else 0
            avg_tokens = sum(r.get("token_counts", {}).get("total_tokens", 0) for r in results) / len(results) if results else 0
            
            comparison[f"top_k_{top_k}"] = {
                "avg_search_time_ms": round(avg_search_time, 2),
                "avg_response_time_ms": round(avg_response_time, 2),
                "avg_total_time_ms": round(avg_total_time, 2),
                "avg_total_tokens": round(avg_tokens, 2),
                "time_efficiency": round(1 / avg_total_time if avg_total_time > 0 else 0, 4),
                "token_efficiency": round(1 / avg_tokens if avg_tokens > 0 else 0, 4)
            }
        
        return comparison
    
    def save_results(self, output_path: str):
        """
        Save evaluation results to file.
        
        Args:
            output_path: Path to save results
        """
        output_data = {
            "individual_results": self.results,
            "performance_summary": self.generate_performance_summary()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    def generate_report(self) -> str:
        """
        Generate human-readable performance report.
        
        Returns:
            Formatted report string
        """
        summary = self.generate_performance_summary()
        
        report = []
        report.append("=" * 80)
        report.append("LongMemEval Performance Report")
        report.append("=" * 80)
        
        # Overall summary
        overall = summary["overall_summary"]
        report.append(f"\nTotal Conversations: {overall['total_conversations']}")
        report.append(f"Add Phase - Avg Time: {overall['add_phase']['average_time_ms']:.2f}ms")
        report.append(f"Add Phase - Avg Tokens: {overall['add_phase']['average_tokens_per_conversation']:.0f}")
        
        # Top-k comparison
        report.append("\n" + "=" * 80)
        report.append("Performance by Top-K Values")
        report.append("=" * 80)
        
        for top_k_key, metrics in summary["by_top_k"].items():
            report.append(f"\n{top_k_key}:")
            report.append(f"  Search Time: {metrics['search']['average_time_ms']:.2f}ms")
            report.append(f"  Response Time: {metrics['response']['average_time_ms']:.2f}ms")
            report.append(f"  Total Time: {metrics['total']['average_time_ms']:.2f}ms")
            report.append(f"  Total Tokens: {metrics['tokens']['average_total_tokens']:.0f}")
        
        # Question type summary
        report.append("\n" + "=" * 80)
        report.append("Performance by Question Type")
        report.append("=" * 80)
        
        for q_type, metrics in summary["by_question_type"].items():
            report.append(f"\n{q_type}:")
            report.append(f"  Conversations: {metrics['conversation_count']}")
            report.append(f"  Avg Total Time: {metrics['average_total_time_ms']:.2f}ms")
            report.append(f"  Avg Total Tokens: {metrics['average_total_tokens']:.0f}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)