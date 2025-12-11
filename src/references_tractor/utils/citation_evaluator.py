import json
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime
import traceback

class CitationEvaluator:
    """
    Evaluation framework with classification metrics (TP, FP, TN, FN)
    and improved output formats with strict/loose metrics
    """
    
    def __init__(self, gold_standard_path: str, pipeline, apis=["openalex", "openaire", "pubmed", "crossref", "hal"]):
        """
        Initialize evaluator with gold standard and pipeline
        
        Args:
            gold_standard_path: Path to JSON file with gold standard data
            pipeline: Citation parser/linker instance (ReferencesTractor)
        """
        self.gold_standard_path = gold_standard_path
        self.pipeline = pipeline
        self.gold_standard = self.load_gold_standard()
        self.apis = apis
        self.results = []
        self.gpu_cleanup_frequency = 10  # Clear GPU cache every N citations

    def _cleanup_gpu_memory(self):
        """Clean up GPU memory cache"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("GPU cache cleared")
        except ImportError:
            pass  # PyTorch not available
        except Exception as e:
            print(f"Error clearing GPU cache: {e}")

    def load_gold_standard(self) -> Dict[str, Dict[str, str]]:
        """Load gold standard from JSON file"""
        with open(self.gold_standard_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def normalize_doi(self, doi: Optional[str]) -> Optional[str]:
        """Normalize DOI for comparison by removing prefixes and standardizing format"""
        if not doi:
            return None
        
        doi = str(doi).strip()
        if not doi:
            return None
        
        # Remove common prefixes
        prefixes_to_remove = [
            "https://doi.org/", "http://doi.org/", "https://dx.doi.org/",
            "http://dx.doi.org/", "doi:", "DOI:"
        ]
        
        doi_lower = doi.lower()
        for prefix in prefixes_to_remove:
            if doi_lower.startswith(prefix.lower()):
                doi = doi[len(prefix):]
                break
        
        doi = doi.strip().rstrip('/')
        return doi.lower() if doi else None

    def normalize_api_id(self, api_id: Optional[str], api: str) -> Optional[str]:
        """Normalize API ID by extracting the core identifier from URLs"""
        if not api_id:
            return None
        
        api_id = str(api_id).strip()
        if not api_id:
            return None
        
        try:
            if api == "openalex":
                if "openalex.org/" in api_id:
                    return api_id.split("openalex.org/")[-1]
                return api_id
                    
            elif api == "openaire":
                return api_id
                    
            elif api == "pubmed":
                if "pubmed.ncbi.nlm.nih.gov/" in api_id:
                    return api_id.split("pubmed.ncbi.nlm.nih.gov/")[-1]
                return api_id
                    
            elif api == "crossref":
                return self.normalize_doi(api_id)
                
            elif api == "hal":
                if "hal.science/" in api_id:
                    return api_id.split("hal.science/")[-1]
                return api_id
            
            return api_id
            
        except Exception:
            return api_id

    def load_previous_results(self, results_file: str) -> List[Dict]:
        """Load previous evaluation results from JSON file"""
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Previous results file not found: {results_file}")
            return []
        except json.JSONDecodeError:
            print(f"Invalid JSON in results file: {results_file}")
            return []

    def determine_final_evaluation(self, result: Dict, expected_gold: Dict, api: str, evaluation_mode: str) -> str:
        """
        Determine final status using strict or loose interpretation
        Returns: CORRECT, INCORRECT, LIKELY, ERROR
        """
        # Handle explicit error status first
        if isinstance(result, dict) and result.get('status') == 'ERROR':
            return "ERROR"
        
        # Handle empty/no results
        if not result or result == {}:
            # For empty results, check if we expected something
            expected_doi = expected_gold.get('doi')
            if api == "ensemble":
                # For ensemble, we only expect DOI
                expected_value = expected_doi
            elif api == "crossref":
                # For crossref, the expected value is the DOI
                expected_value = expected_doi
            else:
                # For other APIs, check their specific expected ID
                expected_value = expected_gold.get(api)
            
            # If we expected nothing, empty result is correct
            if not expected_value:
                return "CORRECT"
            else:
                # If we expected something but got nothing, it's incorrect
                return "INCORRECT"
        
        # Get expected values from gold standard
        expected_doi = expected_gold.get('doi')
        
        # Handle ensemble case specifically
        if api == "ensemble":
            consensus_doi = result.get('consensus_doi') or result.get('main_doi') or result.get('doi')
            
            if expected_doi and consensus_doi:
                normalized_expected = self.normalize_doi(expected_doi)
                normalized_consensus = self.normalize_doi(consensus_doi)
                if normalized_expected and normalized_consensus:
                    if normalized_expected == normalized_consensus:
                        return "CORRECT"
                    else:
                        return "INCORRECT"
            elif not expected_doi and not consensus_doi:
                # Both empty - correct
                return "CORRECT"
            else:
                # One has value, other doesn't - incorrect
                return "INCORRECT"
        
        # Handle individual API cases
        if api == "crossref":
            expected_id = expected_doi
        else:
            expected_id = expected_gold.get(api)
        
        # Extract retrieved values
        main_doi = result.get('main_doi') or result.get('doi')
        alternative_dois = result.get('alternative_dois', [])
        all_dois = [main_doi] + alternative_dois if main_doi else alternative_dois
        retrieved_id = result.get('id') or result.get(f'{api}_id')
        
        # Check for exact DOI matches
        doi_match = False
        if expected_doi:
            normalized_expected_doi = self.normalize_doi(expected_doi)
            if normalized_expected_doi:
                for doi in all_dois:
                    normalized_doi = self.normalize_doi(doi)
                    if normalized_doi and normalized_doi == normalized_expected_doi:
                        doi_match = True
                        break
        
        # Check for exact ID matches (for non-crossref APIs)
        id_match = False
        if expected_id and retrieved_id and api != "crossref":
            normalized_expected_id = self.normalize_api_id(expected_id, api)
            normalized_retrieved_id = self.normalize_api_id(retrieved_id, api)
            if normalized_expected_id and normalized_retrieved_id:
                id_match = (normalized_expected_id == normalized_retrieved_id)
        
        # If we found an exact match
        if doi_match or id_match:
            return "CORRECT"
        
        # Check for likely matches (metadata-based)
        metadata_match = result.get('metadata_match', 'N/A')
        if metadata_match == "LIKELY_SAME":
            if evaluation_mode == "strict":
                return "INCORRECT"
            elif evaluation_mode == "loose":
                return "CORRECT"
        
        # If we retrieved something but it doesn't match
        if main_doi or alternative_dois or retrieved_id:
            return "INCORRECT"
        
        # If we expected something but retrieved nothing
        if expected_doi or expected_id:
            return "INCORRECT"
        
        # Fallback - should rarely reach here
        return "ERROR"

    def calculate_classification_metrics(self, evaluation_mode: str = "strict") -> Dict[str, Dict[str, Any]]:
        """
        Calculate metrics from evaluation results data - updated to match generate_summary_evaluation.py
        """
        metrics = {}
        approaches = self.apis + ['ensemble']
        
        for approach in approaches:
            total_count = 0
            total_correct_count = 0
            correct_matches_count = 0
            correct_no_result_count = 0
            total_incorrect_count = 0
            incorrect_matches_count = 0
            incorrect_missing_count = 0
            incorrect_spurious_count = 0

            
            for result in self.results:
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
    
    def evaluate_single_citation(self, citation: str, expected_results: Dict[str, str], evaluation_mode: str) -> Dict[str, Any]:
        """
        Evaluation with final status determination
        """
        citation_result = {
            'citation_id': f"Cit_{len(self.results)+1:03d}",
            'original_citation': citation,
            'gold_standard': expected_results,
            'api_results': {},
            'ensemble': {}
        }
        
        # Test each API individually
        for api in self.apis:
            print(f"Calling {api} for citation: {citation[:50]}...")
            try:
                result = self.pipeline.link_citation(citation, api_target=api, output='advanced')
                
                if result and ('result' in result or isinstance(result, str)):
                    if not isinstance(result, dict):
                        citation_result['api_results'][api] = {
                            'status': 'ERROR',
                            'final_evaluation': 'ERROR'
                        }
                        continue
                    
                    # Store result
                    api_data = {
                        'status': 'RESULT',
                        'score': result.get('score', 'N/A'),
                        'id': result.get(f'{api}_id') or result.get('id'),
                        'main_doi': result.get('main_doi') or result.get('doi'),
                        'alternative_dois': result.get('alternative_dois', []),
                        'total_dois': result.get('total_dois', 0),
                        'all_dois': result.get('all_dois', []),
                        'formatted_citation': result.get('result', ''),
                        'metadata_match': result.get('metadata_match', 'N/A')
                    }
                    
                    # Determine final status
                    api_data['final_evaluation'] = self.determine_final_evaluation(api_data, expected_results, api, evaluation_mode)
                    citation_result['api_results'][api] = api_data
                    
                else:
                    citation_result['api_results'][api] = {
                        'status': 'NO_RESULT',
                        'final_evaluation': self.determine_final_evaluation({}, expected_results, api, evaluation_mode)
                    }
                    
            except Exception as e:
                citation_result['api_results'][api] = {
                    'status': 'ERROR',
                    'final_evaluation': 'ERROR',
                    'error': str(e)
                }
                print(str(e))
        
        # Test ensemble
        try:
            ensemble_result = self.pipeline.link_citation_ensemble(citation, api_targets=self.apis, output='advanced')
            
            if ensemble_result and ensemble_result.get('doi'):
                ensemble_data = {
                    'status': 'RESULT',
                    'consensus_doi': ensemble_result['doi'],
                    'external_ids': ensemble_result.get('external_ids', {}),
                    'metadata': ensemble_result.get('ensemble_metadata', {}),
                    # Add fields that determine_final_evaluation expects
                    'main_doi': ensemble_result['doi'],  # For DOI matching
                    'doi': ensemble_result['doi']        # Fallback
                }
                ensemble_data['final_evaluation'] = self.determine_final_evaluation(
                    ensemble_data, expected_results, 'ensemble', evaluation_mode  # Pass evaluation_mode here
                )
                citation_result['ensemble'] = ensemble_data
            else:
                citation_result['ensemble'] = {
                    'status': 'NO_RESULT',
                    'final_evaluation': self.determine_final_evaluation({}, expected_results, 'ensemble', evaluation_mode)  # Pass evaluation_mode
                }
                
        except Exception as e:
            citation_result['ensemble'] = {
                'status': 'ERROR',
                'final_evaluation': 'ERROR',
                'error': str(e)
            }
        
        return citation_result
       
    def run_evaluation(self, limit: Optional[int] = None, evaluation_mode: str = "strict", 
                      resume_from: Optional[str] = None, skip_count: int = 0) -> None:
        """Run evaluation with resume capability and GPU error handling"""
        
        # Load previous results if resuming
        if resume_from:
            print(f"Loading previous results from: {resume_from}")
            previous_results = self.load_previous_results(resume_from)
            
            if skip_count > 0:
                # Keep only the first skip_count results (the good ones)
                self.results = previous_results[:skip_count]
                print(f"Kept first {len(self.results)} previous results")
                print(f"Will reprocess from citation {skip_count + 1} onwards")
                total_to_skip = skip_count
            else:
                # Keep all previous results and continue from where we left off
                self.results = previous_results
                total_to_skip = len(previous_results)
                print(f"Loaded {len(self.results)} previous results")
                print(f"Resuming from citation {total_to_skip + 1}")
        else:
            self.results = []
            total_to_skip = skip_count
            if skip_count > 0:
                print(f"Skipping first {skip_count} citations (fresh start)")
        
        print(f"Starting evaluation with classification metrics...")
        
        citations_to_process = list(self.gold_standard.items())
        
        # Skip citations based on resume or skip parameter
        if total_to_skip > 0:
            if total_to_skip >= len(citations_to_process):
                print(f"Error: Trying to skip {total_to_skip} citations, but only {len(citations_to_process)} available")
                return
            citations_to_process = citations_to_process[total_to_skip:]
            print(f"Processing {len(citations_to_process)} remaining citations")
        
        # Apply limit after skipping
        if limit and len(citations_to_process) > limit:
            citations_to_process = citations_to_process[:limit]
            print(f"Limited to {limit} citations after skipping")
        
        for i, (citation, expected_results) in enumerate(citations_to_process):
            current_index = total_to_skip + i + 1
            print(f"\n\nProcessing citation {current_index}/{len(self.gold_standard)}: {citation[:80]}...")
            
            try:
                result = self.evaluate_single_citation(citation, expected_results, evaluation_mode)
                self.results.append(result)
                
                # GPU cache cleanup every N citations
                if (len(self.results) % self.gpu_cleanup_frequency == 0):
                    self._cleanup_gpu_memory()
                    
            except Exception as e:
                print(f"Failed to process citation {current_index}: {str(e)}")
                traceback.print_exc()
        
        print(f"Evaluation completed! Total results: {len(self.results)} citations.")

    def generate_summary_dashboard(self, evaluation_mode: str = "strict") -> str:
        """Generate summary dashboard with improved table formatting - updated to match generate_summary_evaluation.py"""
        classification_metrics = self.calculate_classification_metrics(evaluation_mode)
        
        output = []
        output.append("="*120)
        output.append("CITATION LINKING EVALUATION - SUMMARY DASHBOARD")
        output.append("="*120)
        output.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"Total Citations Evaluated: {len(self.results)}")
        output.append(f"APIs Tested: {', '.join(self.apis)}")
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
        for api in self.apis + ['ensemble']:
            if api in classification_metrics:
                m = classification_metrics[api]
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
        
        return "\n".join(output)
    
    def save_comparison_table_tsv(self, filename: str):
        """Save concise comparison table with gold standard IDs and final status"""
        if not self.results:
            return
        
        table_data = []
        for result in self.results:
            gold_standard = result['gold_standard']
            
            row = {
                'Citation_ID': result['citation_id'],
                'Citation': result['original_citation'][:80] + "..." if len(result['original_citation']) > 80 else result['original_citation'],
                'Gold_DOI': gold_standard.get('doi', ''),
                'Gold_OpenAlex_ID': gold_standard.get('openalex', ''),
                'Gold_OpenAIRE_ID': gold_standard.get('openaire', ''),
                'Gold_PubMed_ID': gold_standard.get('pubmed', ''),
                'Gold_CrossRef_ID': gold_standard.get('crossref', ''),
                'Gold_HAL_ID': gold_standard.get('hal', '')
            }
            
            # Add status columns (using loose interpretation)
            for api in self.apis:
                api_data = result.get('api_results', {}).get(api, {})
                row[f'{api.title()}_Status'] = api_data.get('final_evaluation', 'ERROR')
                row[f'{api.title()}_DOI'] = api_data.get('main_doi', '')
            
            # Add ensemble
            ensemble_data = result.get('ensemble', {})
            row['Ensemble_Status'] = ensemble_data.get('final_evaluation', 'ERROR')
            row['Ensemble_DOI'] = ensemble_data.get('consensus_doi', '')
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        df.to_csv(filename, sep='\t', index=False, encoding='utf-8')
        
    def save_individual_table_tsv(self, approach: str, filename: str):
        """Save individual approach table with gold standard information"""
        if approach == 'ensemble':
            ensemble_data = []
            for result in self.results:
                gold_standard = result['gold_standard']
                ensemble_info = result.get('ensemble', {})
                
                ensemble_data.append({
                    'Citation_ID': result['citation_id'],
                    'Citation': result['original_citation'],
                    'Gold_DOI': gold_standard.get('doi', ''),
                    'Consensus_DOI': ensemble_info.get('consensus_doi', ''),
                    'External_IDs': str(ensemble_info.get('external_ids', {})),
                    'Status': ensemble_info.get('status', 'ERROR'),
                    'Final_Evaluation': ensemble_info.get('final_evaluation', 'ERROR')
                })
            
            df = pd.DataFrame(ensemble_data)
        else:
            api_data = []
            for result in self.results:
                gold_standard = result['gold_standard']
                api_info = result.get('api_results', {}).get(approach, {})
                
                api_data.append({
                    'Citation_ID': result['citation_id'],
                    'Citation': result['original_citation'],
                    'Gold_DOI': gold_standard.get('doi', ''),
                    f'Gold_{approach.title()}_ID': gold_standard.get(approach, ''),
                    'Retrieved_ID': api_info.get('id', ''),
                    'Main_DOI': api_info.get('main_doi', ''),
                    'Alternative_DOIs': str(api_info.get('alternative_dois', [])),
                    'Total_DOIs': api_info.get('total_dois', 0),
                    'Score': api_info.get('score', 'N/A'),
                    'Status': api_info.get('status', 'ERROR'),
                    'Final_Evaluation': api_info.get('final_evaluation', 'ERROR'),
                    'Formatted_Citation': api_info.get('formatted_citation', '')
                })
            
            df = pd.DataFrame(api_data)
        
        df.to_csv(filename, sep='\t', index=False, encoding='utf-8')
    
    def save_results(self, output_dir: str = "evaluation_results", evaluation_mode: str = "strict"):
        """Save all evaluation results with updated metrics structure"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary dashboard
        summary = self.generate_summary_dashboard(evaluation_mode)
        with open(f"{output_dir}/01_summary_dashboard_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(summary)
        
        # Save accuracy metrics as separate TSV - updated to match generate_summary_evaluation.py structure
        classification_metrics = self.calculate_classification_metrics(evaluation_mode)
        metrics_data = []
        for api, metric_values in classification_metrics.items():
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
        df_metrics.to_csv(f"{output_dir}/02_accuracy_metrics_{timestamp}.tsv", sep='\t', index=False, encoding='utf-8')
        
        # Save concise comparison table
        self.save_comparison_table_tsv(f"{output_dir}/03_comparison_table_{timestamp}.tsv")
        
        # Save individual API detailed tables
        for approach in self.apis + ['ensemble']:
            filename = f"{output_dir}/04_{approach}_detailed_{timestamp}.tsv"
            self.save_individual_table_tsv(approach, filename)
        
        # Save raw results
        with open(f"{output_dir}/05_results_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"Evaluation results saved to {output_dir}/ directory")
        return output_dir
