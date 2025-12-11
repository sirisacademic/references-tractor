#!/usr/bin/env python3
"""
Citation Linking Evaluation Script
==================================

This script evaluates the performance of the citation linking pipeline
against a human-validated gold standard dataset using classification metrics.

Features:
- TP/FP/TN/FN classification metrics
- Strict vs Loose evaluation modes
- Comparison tables with gold standard IDs
- Detailed individual API analysis

Usage:
    python evaluate_citations.py [--limit N] [--output-dir DIR] [--gold-standard PATH]

Examples:
    # Test with first 5 citations
    python evaluate_citations.py --limit 5
    
    # Full evaluation
    python evaluate_citations.py
    
    # Custom paths
    python evaluate_citations.py --gold-standard my_test.json --output-dir my_results/
"""

EVALUATE_APIS = ["openalex", "openaire", "pubmed", "crossref", "hal"]
#EVALUATE_APIS = ["openalex"]

import argparse
import sys
import os
from pathlib import Path
import traceback
from datetime import datetime

# Determine project root
current_file = Path(__file__).absolute()
if current_file.parent.name == "evaluation":
    # Running from evaluation/ directory or as evaluation/script.py
    project_root = current_file.parent.parent
else:
    # Running from project root
    project_root = current_file.parent

# Add project root to Python path if needed
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from references_tractor import ReferencesTractor
    print("ReferencesTractor import successful")
except ImportError as e:
    print(f"Error importing ReferencesTractor: {e}")
    print(f"Python path: {sys.path[:3]}")
    sys.exit(1)

try:
    from references_tractor.utils import CitationEvaluator
    print("CitationEvaluator import successful")
except ImportError as e:
    print(f"Error importing CitationEvaluator: {e}")
    print(f"Available in utils: {list((project_root / 'references_tractor' / 'utils').iterdir())}")
    sys.exit(1)

def main():
    """Main evaluation script"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate citation linking pipeline with classification metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--gold-standard", 
        type=str, 
        default="evaluation/api_linking_test.json",
        help="Path to gold standard JSON file (default: evaluation/api_linking_test.json)"
    )
    
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit evaluation to first N citations (for testing). If not specified, evaluates all citations."
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="evaluation_results",
        help="Output directory for results (default: evaluation_results)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for model inference. 'auto' detects best available device (default: auto)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output during evaluation"
    )
    
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show classification metrics preview after evaluation"
    )

    # Add new argument for evaluation mode (strict or loose)
    parser.add_argument(
        "--evaluation-mode",
        type=str,
        choices=["strict", "loose"],
        default="strict",
        help="Choose evaluation mode: 'strict' or 'loose' (default: strict)"
    )
    
    parser.add_argument(
        "--resume-from", 
        type=str, 
        default=None,
        help="Resume evaluation from previous results JSON file"
    )
    
    parser.add_argument(
        "--skip", 
        type=int, 
        default=0,
        help="Number of citations to skip from the beginning"
    )

    args = parser.parse_args()
    
    # Print startup information
    print("="*80)
    print("CITATION LINKING PIPELINE EVALUATION")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Evaluating: {EVALUATE_APIS}")
    print(f"Gold standard: {args.gold_standard}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device selection: {args.device}")
    print(f"Evaluation Mode: {args.evaluation_mode}")
    print("Features:")
    print("  • Strict vs Loose Evaluation")
    if args.limit:
        print(f"Evaluation limit: {args.limit} citations")
    else:
        print("Evaluation limit: All citations")
    print()
    
    # Check if gold standard file exists
    gold_standard_path = project_root / args.gold_standard
    if not gold_standard_path.exists():
        print(f"Error: Gold standard file not found: {gold_standard_path}")
        print("Please check the file path or use --gold-standard to specify the correct path.")
        sys.exit(1)
    
    try:
        # Initialize the citation pipeline
        print("Initializing citation pipeline...")
        if args.device == "auto":
            print("Auto-detecting best available device...")
        
        pipeline = ReferencesTractor(device=args.device)
        print("Pipeline initialized successfully")
        
        # Initialize the evaluator
        print("Loading gold standard and initializing evaluator...")
        evaluator = CitationEvaluator(
            gold_standard_path=str(gold_standard_path),
            pipeline=pipeline,
            apis=EVALUATE_APIS
        )
        print(f"Loaded {len(evaluator.gold_standard)} citations from gold standard")
        
        # Run the evaluation with the selected evaluation mode (strict or loose)
        print("\n" + "="*50)
        print("STARTING EVALUATION")
        print("="*50)
        
        if args.limit:
            print(f"Running evaluation on first {args.limit} citations...")
        else:
            print(f"Running full evaluation on {len(evaluator.gold_standard)} citations...")
            print("This may take several minutes...")
        
        print("Processing each citation through all APIs...")
        print("Computing classification metrics...")
        
        # Run the evaluation with resume capability
        evaluator.run_evaluation(
            limit=args.limit, 
            evaluation_mode=args.evaluation_mode,
            resume_from=args.resume_from,
            skip_count=args.skip
        )
       
        print("\nEvaluation completed successfully!")
        
        # Show classification metrics preview if requested
        if args.show_preview or args.verbose:
            print("\n" + "="*50)
            print("CLASSIFICATION METRICS PREVIEW") 
            print("="*50)
            try:
                classification_metrics = evaluator.calculate_classification_metrics(args.evaluation_mode)
                print(f"{'API':<12} {'Accuracy':<10}")
                print("-" * 50)
                
                for api in ['openalex', 'openaire', 'pubmed', 'crossref', 'ensemble']:
                    if api in classification_metrics:
                        metrics = classification_metrics[api]
                        accuracy = metrics['accuracy']
                        print(f"{api.title():<12} {accuracy:<10.3f}")
                        
            except Exception as e:
                print(f"Error calculating preview metrics: {e}")
        
        # Save all results
        print(f"\nSaving results to {args.output_dir}/...")
        output_dir = evaluator.save_results(args.output_dir)
        
        print("\n" + "="*50)
        print("EVALUATION COMPLETE")
        print("="*50)
        print(f"Summary displayed above")
        print(f"Detailed results saved to: {output_dir}/")
        print(f"Files generated:")
        
        # List generated files with descriptions
        result_files = []
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.endswith(('.txt', '.json', '.tsv')):
                    result_files.append(file)
        
        for file in sorted(result_files):
            print(f"   {file}")
        
        print(f"\nEvaluation finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
               
        if args.limit:
            print(f"\nThis was a limited evaluation ({args.limit} citations).")
            print("   Run without --limit for full evaluation on all citations.")
            print("   Use --show-preview for quick metrics preview.")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nEvaluation failed with error: {str(e)}")
        print("\nFull error traceback:")
        traceback.print_exc()
        
        print(f"\nTroubleshooting tips:")
        print("1. Check that all required packages are installed")
        print("2. Verify the gold standard file path and format")
        print("3. Ensure you have internet connectivity for API calls")
        print("4. Try running with --limit 1 to test with a single citation")
        print("5. Use --device cpu if you're having GPU-related issues")
        print("6. Use --device auto to auto-detect the best available device")
        
        sys.exit(1)


if __name__ == "__main__":
    main()

