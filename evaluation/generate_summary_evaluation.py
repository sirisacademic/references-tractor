#!/usr/bin/env python3
"""
Regenerate accuracy metrics and summary dashboard from existing detailed results
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import sys

# Determine project root
current_file = Path(__file__).absolute()
if current_file.parent.name == "evaluation":
    # Running from evaluation/ directory or as evaluation/script.py
    project_root = current_file.parent.parent
else:
    # Running from project root
    project_root = current_file.parent

def calculate_metrics(results_data, evaluation_mode="strict"):
    """
    Calculate metrics from evaluation results data
    """
    metrics = {}
    approaches = ['openalex', 'openaire', 'pubmed', 'crossref', 'hal', 'ensemble']
    
    for approach in approaches:
        total_count = 0
        total_correct_count = 0
        correct_matches_count = 0
        correct_no_result_count = 0
        total_incorrect_count = 0
        incorrect_matches_count = 0
        incorrect_missing_count = 0
        incorrect_spurious_count = 0

        
        for result in results_data:
            gold_standard = result['gold_standard']
            
            # Get expected value for this approach
            if approach == 'ensemble' or approach == 'crossref':
                expected_value = gold_standard.get('doi')  # Ensemble and CrossRef expect DOI
                if approach == 'ensemble':
                    api_result = result.get('ensemble', {})
                else:
                    api_result = result.get('api_results', {}).get(approach, {})
            else:
                expected_value = gold_standard.get(approach)  # Other APIs expect their specific ID
                api_result = result.get('api_results', {}).get(approach, {})
            
            status = api_result.get('status', 'ERROR')
            final_evaluation = api_result.get('final_evaluation', 'ERROR')
            
            # Skip actual errors from counting
            if final_evaluation == 'ERROR':
                continue
            
            total_count += 1
            
            # Categorize result types
            has_result = (status == 'RESULT')
            has_expected = bool(expected_value)

            # Count correct results
            if final_evaluation == "CORRECT" or (evaluation_mode == "loose" and final_evaluation == "LIKELY"):
                total_correct_count += 1
                if not has_expected:
                    correct_no_result_count += 1
                else:
                    correct_matches_count += 1
          # Count incorrect results  
            else: 
                total_incorrect_count += 1
                if has_result and has_expected:
                    # There is a match but it is incorrect
                    incorrect_matches_count += 1
                if not has_result and has_expected:
                    # No result but expected something - Missing
                    incorrect_missing_count += 1
                elif has_result and not has_expected:
                    # Has result but expected nothing - Spurious
                    incorrect_spurious_count += 1
        
        accuracy = total_correct_count / total_count if total_count > 0 else 0.0
        
        metrics[approach] = {
            'accuracy': accuracy,
            'total_count': total_count,
            'total_correct_count': total_correct_count,
            'correct_matches_count': correct_matches_count,
            'correct_no_result_count': correct_no_result_count,
            'total_incorrect_count': total_incorrect_count,
            'incorrect_matches_count': incorrect_matches_count,
            'incorrect_missing_count': incorrect_missing_count,
            'incorrect_spurious_count': incorrect_spurious_count
        }
    
    return metrics

def regenerate_accuracy_metrics_tsv(metrics, output_file):
    """
    Generate the accuracy metrics TSV file
    """
    metrics_data = []
    for api, metric_values in metrics.items():
        metrics_data.append({
            'API': api.title(),
            'Accuracy': f"{metric_values['accuracy']:.3f}",
            'Total': metric_values['total_count'],
            'Total_Correct': metric_values['total_correct_count'],
            'Correct_Matches': metric_values['correct_matches_count'],
            'Correct_No_Result': metric_values['correct_no_result_count'],
            'Total_Incorrect': metric_values['total_incorrect_count'],
            'Incorrect_Matches': metric_values['incorrect_matches_count'],
            'Incorrect_Missing': metric_values['incorrect_missing_count'],
            'Incorrect_Spurious': metric_values['incorrect_spurious_count']
        })
        
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
    print(f"Generated: {output_file}")

def regenerate_summary_dashboard(metrics, output_file, total_citations, evaluation_mode):
    """
    Generate the summary dashboard with improved table formatting
    """
    output = []
    output.append("="*120)
    output.append("CITATION LINKING EVALUATION - SUMMARY DASHBOARD")
    output.append("="*120)
    output.append(f"Regenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"Total Citations Evaluated: {total_citations}")
    output.append(f"APIs Tested: openalex, openaire, pubmed, crossref, hal")
    output.append(f"Evaluation Mode: {evaluation_mode.title()}")
    output.append("Features: Metrics (Match, No_Result, Missing, Spurious)")
    output.append("")
    
    # Improved metrics table with better column spacing
    output.append("ACCURACY METRICS:")
    output.append("="*120)
    
    # Create properly formatted header
    header = f"{'API':<12} {'Accuracy':<9} {'Total':<6} {'T_Corr':<7} {'C_Match':<8} {'C_NoRes':<8} {'T_Incor':<8} {'I_Match':<8} {'I_Miss':<7} {'I_Spur':<7}"
    output.append(header)
    output.append("-"*120)
    
    # Add data rows with consistent formatting
    for api in ['openalex', 'openaire', 'pubmed', 'crossref', 'hal', 'ensemble']:
        if api in metrics:
            m = metrics[api]
            row = f"{api.title():<12} {m['accuracy']:<9.3f} {m['total_count']:<6} {m['total_correct_count']:<7} {m['correct_matches_count']:<8} {m['correct_no_result_count']:<8} {m['total_incorrect_count']:<8} {m['incorrect_matches_count']:<8} {m['incorrect_missing_count']:<7} {m['incorrect_spurious_count']:<7}"
            output.append(row)
    
    output.append("="*120)
    output.append("")
    
    # Enhanced column definitions with better formatting
    output.append("COLUMN DEFINITIONS:")
    output.append("-"*50)
    output.append("BASIC METRICS:")
    output.append("  • API        : API name (OpenAlex, OpenAIRE, PubMed, CrossRef, HAL, Ensemble)")
    output.append("  • Accuracy   : Percentage of correct predictions")
    output.append("  • Total      : Total citations processed (excluding errors)")
    output.append("")
    output.append("CORRECT RESULTS:")
    output.append("  • T_Corr     : Total Correct - All correctly classified citations")
    output.append("  • C_Match    : Correct Matches - Citations correctly linked to expected results")
    output.append("  • C_NoRes    : Correct No Result - Citations that correctly returned no match")
    output.append("")
    output.append("INCORRECT RESULTS:")
    output.append("  • T_Incor    : Total Incorrect - All incorrectly classified citations") 
    output.append("  • I_Match    : Incorrect Matches - Wrong links when correct link expected")
    output.append("  • I_Miss     : Incorrect Missing - No result when correct link expected")
    output.append("  • I_Spur     : Incorrect Spurious - Unexpected result when no link expected")
    output.append("")
    output.append("INTERPRETATION:")
    output.append("  • High I_Miss indicates the API is missing papers that should be found")
    output.append("  • High I_Spur indicates the API is returning false positives")
    output.append("  • High I_Match indicates the API finds papers but links to wrong ones")
    output.append("  • Perfect score: T_Corr = Total, T_Incor = 0")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))
    
    print(f"Generated: {output_file}")

def main():
    """
    Main function to regenerate metrics from existing results
    """
    if len(sys.argv) < 2:
        print("Usage: python evaluation/generate_summary_evaluation.py <results_json_file> [evaluation_mode] [output_dir]")
        print("Example: python evaluation/generate_summary_evaluation.py evaluation_results/05_results_20250623_170050.json strict evaluation_results")
        sys.exit(1)
    
    results_file = sys.argv[1]
    evaluation_mode = sys.argv[2] if len(sys.argv) > 2 else "strict"
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(f"{project_root}/evaluation_results")
    
    # Load results
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        print(f"Loaded {len(results_data)} results from {results_file}")
    except FileNotFoundError:
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in results file: {results_file}")
        sys.exit(1)
    
    # Calculate metrics
    metrics = calculate_metrics(results_data, evaluation_mode)
    
    # Generate timestamp for new files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate new files
    accuracy_file = output_dir / f"02_accuracy_metrics_{timestamp}.tsv"
    dashboard_file = output_dir / f"01_summary_dashboard_{timestamp}.txt"
    
    regenerate_accuracy_metrics_tsv(metrics, accuracy_file)
    regenerate_summary_dashboard(metrics, dashboard_file, len(results_data), evaluation_mode)
    
    print(f"Summary metrics regenerated successfully!")
    print(f"Files created in: {output_dir}")

if __name__ == "__main__":
    main()
