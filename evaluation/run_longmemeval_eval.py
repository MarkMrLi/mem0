#!/usr/bin/env python3
"""
Main evaluation script for LongMemEval dataset with mem0.
This script evaluates mem0's performance on the LongMemEval benchmark.
"""
import argparse
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.longmemeval.data_converter import LongMemEvalDataConverter
from src.longmemeval.add import LongMemEvalMemoryAdder
from src.longmemeval.search import LongMemEvalMemorySearch
from src.longmemeval.performance_evaluator import PerformanceEvaluator

load_dotenv()


def run_longmemeval_evaluation(
    dataset_path: str,
    output_dir: str,
    top_k_values: list = None,
    max_concurrent: int = 3,
    limit_conversations: int = None
):
    """
    Run complete LongMemEval evaluation pipeline.
    
    Args:
        dataset_path: Path to LongMemEval JSON file
        output_dir: Directory to save results
        top_k_values: List of top_k values to test
        max_concurrent: Maximum concurrent requests for memory addition
        limit_conversations: Limit number of conversations to evaluate (for testing)
    """
    if top_k_values is None:
        top_k_values = [1, 5, 10, 20, 30]
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(output_dir) / f"longmemeval_eval_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting LongMemEval Evaluation")
    print(f"📁 Results will be saved to: {experiment_dir}")
    print(f"📊 Dataset: {dataset_path}")
    print(f"🔍 Top-K values to test: {top_k_values}")
    
    # Initialize components
    converter = LongMemEvalDataConverter()
    adder = LongMemEvalMemoryAdder()
    searcher = LongMemEvalMemorySearch()
    evaluator = PerformanceEvaluator()
    
    # Load and convert dataset
    print(f"\n📖 Loading and converting dataset...")
    converted_data = converter.load_and_convert_json(dataset_path)
    
    # Get dataset statistics
    stats = converter.get_statistics(converted_data)
    print(f"   Total conversations: {stats['total_conversations']}")
    print(f"   Total messages: {stats['total_messages']}")
    print(f"   Question types: {stats['question_type_distribution']}")
    
    # Limit conversations for testing
    if limit_conversations:
        converted_data = converted_data[:limit_conversations]
        print(f"   ⚠️ Limited to {limit_conversations} conversations for testing")
    
    # Process each conversation
    print(f"\n🔄 Processing conversations...")
    
    results = []
    
    for idx, conversation in enumerate(tqdm(converted_data, desc="Evaluating")):
        conversation_id = conversation["conversation_id"]
        
        try:
            # Step 1: Add conversation memories to mem0
            add_metrics = adder.add_conversation(
                conversation_data=conversation,
                max_concurrent=max_concurrent
            )
            
            # Step 2: Search and respond with different top_k values
            search_results = searcher.search_and_respond(
                user_id=conversation["user_id"],
                query=conversation["query"],
                top_k_values=top_k_values
            )
            
            # Step 3: Evaluate performance
            conversation_result = evaluator.evaluate_conversation(
                conversation_data=conversation,
                add_metrics=add_metrics,
                search_results=search_results
            )
            
            results.append(conversation_result)
            
        except Exception as e:
            print(f"   ❌ Error processing {conversation_id}: {e}")
            # Add error result
            error_result = {
                "conversation_id": conversation_id,
                "error": str(e),
                "question_type": conversation["question_type"],
                "question": conversation["query"]
            }
            results.append(error_result)
        
        # Save intermediate results every 10 conversations
        if (idx + 1) % 10 == 0:
            intermediate_file = experiment_dir / f"intermediate_results_{idx + 1}.json"
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Generate final evaluation report
    print(f"\n📊 Generating evaluation report...")
    
    # Save complete results
    results_file = experiment_dir / "complete_results.json"
    evaluator.save_results(str(results_file))
    
    # Generate and save report
    report = evaluator.generate_report()
    report_file = experiment_dir / "performance_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save summary statistics
    summary = evaluator.generate_performance_summary()
    summary_file = experiment_dir / "performance_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n✅ Evaluation completed!")
    print(f"\n{report}")
    
    print(f"\n📁 Results saved to: {experiment_dir}")
    print(f"   - Complete results: {results_file}")
    print(f"   - Performance report: {report_file}")
    print(f"   - Performance summary: {summary_file}")
    
    return experiment_dir


def main():
    """Main entry point for LongMemEval evaluation."""
    parser = argparse.ArgumentParser(
        description="Run LongMemEval evaluation with mem0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with default settings
  python run_longmemeval_eval.py --dataset dataset/longmemeval_s.json
  
  # Run with custom top_k values
  python run_longmemeval_eval.py --dataset dataset/longmemeval_s.json --top_k 5 10 15
  
  # Run with limited conversations for testing
  python run_longmemeval_eval.py --dataset dataset/longmemeval_s.json --limit 5
  
  # Run with custom output directory
  python run_longmemeval_eval.py --dataset dataset/longmemeval_s.json --output results/my_eval
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to LongMemEval dataset JSON file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/longmemeval",
        help="Output directory for results (default: results/longmemeval)"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 30],
        help="Top-K values to test (default: 1 5 10 20 30)"
    )
    
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=3,
        help="Maximum concurrent requests for memory addition (default: 3)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of conversations to evaluate (for testing)"
    )
    
    args = parser.parse_args()
    
    # Check if dataset file exists
    if not os.path.exists(args.dataset):
        print(f"❌ Error: Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    # Run evaluation
    try:
        run_longmemeval_evaluation(
            dataset_path=args.dataset,
            output_dir=args.output,
            top_k_values=args.top_k,
            max_concurrent=args.max_concurrent,
            limit_conversations=args.limit
        )
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()