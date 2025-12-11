# core.py

# Import necessary modules
import re
import time
import json
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib

import requests
import pandas as pd
from tqdm.notebook import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline,
)
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from .utils.span import extract_references_and_mentions, match_mentions_to_reference
from .utils.prescreening import prescreen_references

THRESHOLD_PARWISE_MODEL = 0.90 # Higher threshold for SELECT model
THRESHOLD_NER_SIMILARITY = 0.60 # Lower threshold for NER similarity

#THRESHOLD_PARWISE_MODEL = 0.80 # Higher threshold for SELECT model
#THRESHOLD_NER_SIMILARITY = 0.70 # Lower threshold for NER similarity

def _safe_import():
    """Safely import internal modules with fallback handling"""
    # Try absolute imports first (works with pip install)
    try:
        from references_tractor.search import search_api
        from references_tractor.search import citation_formatter
        from references_tractor.utils.entity_validation import EntityValidator
        return search_api, citation_formatter, EntityValidator
    except ImportError:
        # Fallback to relative imports (works with -e install and development)
        try:
            from .search import search_api
            from .search import citation_formatter
            from .utils.entity_validation import EntityValidator
            return search_api, citation_formatter, EntityValidator
        except ImportError:
            # Final fallback for development/testing
            import sys
            import os
            from pathlib import Path
            
            # Add project root to path
            current_dir = Path(__file__).parent
            project_root = current_dir.parent if current_dir.name == 'references_tractor' else current_dir
            sys.path.insert(0, str(project_root))
            
            try:
                from references_tractor.search import search_api
                from references_tractor.search import citation_formatter
                from references_tractor.utils.entity_validation import EntityValidator
                return search_api, citation_formatter, EntityValidator
            except ImportError as e:
                raise ImportError(f"Could not import required modules. Please ensure references_tractor is properly installed. Error: {e}")

# Import internal modules
search_api, citation_formatter, EntityValidator = _safe_import()

class ReferencesTractor:

    def __init__(
        self,
        ner_model_path: str = "SIRIS-Lab/citation-parser-ENTITY",
        select_model_path: str = "SIRIS-Lab/citation-parser-SELECT",
        prescreening_model_path: str = "SIRIS-Lab/citation-parser-TYPE",
        span_model_path: str = "SIRIS-Lab/citation-parser-SPAN",
        device: Union[int, str] = "auto",
        enable_caching: bool = True,
        cache_size_limit: int = 1000,
        select_threshold: float = THRESHOLD_PARWISE_MODEL,
        ner_threshold: float = THRESHOLD_NER_SIMILARITY,
        debug: bool = False,
    ):
        self._switched_to_cpu = False  # Track if we've switched to CPU due to GPU errors

        # Auto-detect device if not specified
        if device == "auto":
            device = self._detect_best_device()
            print(f"Auto-detected device: {device}")
        elif device == "cpu":
            print("Using CPU for model inference")
        else:
            print(f"Using specified device: {device}")
        
        # Initialize three different transformer pipelines:
        # 1. NER for citation entity extraction
        # 2. Selection model to rank possible citation matches
        # 3. Prescreening model to filter non-citation inputs
        self.ner_pipeline = self._init_pipeline("ner", ner_model_path, device, agg_strategy="simple")
        self.select_pipeline = self._init_pipeline("text-classification", select_model_path, device)
        self.prescreening_pipeline = self._init_pipeline("text-classification", prescreening_model_path, device)
        self.span_pipeline = self._init_pipeline("ner", span_model_path, device, agg_strategy="simple")
        self.searcher = search_api.SearchAPI(debug=debug)
        
        # Initialize caching system
        self.enable_caching = enable_caching
        self.cache_size_limit = cache_size_limit
        self._citation_cache = {}
        self._cache_stats = {'hits': 0, 'misses': 0, 'size': 0}

        self.debug = debug
        if self.debug:
            print("DEBUGGING set to True")

        # Initialize thresholds
        self.select_threshold = select_threshold
        self.ner_threshold = ner_threshold

    def _detect_best_device(self) -> str:
        """
        Auto-detect the best available device for model inference
        Returns: device string ("cuda", "mps", or "cpu")
        """
        try:
            import torch
            
            # Check for NVIDIA CUDA GPU
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                print(f"CUDA GPU detected: {gpu_name} (GPU count: {gpu_count})")
                return "cuda"
            
            # Check for Apple Silicon MPS (Metal Performance Shaders)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("Apple Silicon GPU (MPS) detected")
                return "mps"
            
            # Fallback to CPU
            else:
                import multiprocessing
                cpu_count = multiprocessing.cpu_count()
                print(f"No GPU detected, using CPU ({cpu_count} cores)")
                return "cpu"
                
        except ImportError:
            print("PyTorch not found, defaulting to CPU")
            return "cpu"
        except Exception as e:
            print(f"Error during device detection: {e}, defaulting to CPU")
            return "cpu"

    def _is_cuda_error(self, error: Exception) -> bool:
        """Check if error is CUDA-related"""
        error_str = str(error).lower()
        cuda_indicators = [
            "cuda error", "cuda out of memory", "cuda kernel errors",
            "unspecified launch failure", "device-side assertions"
        ]
        return any(indicator in error_str for indicator in cuda_indicators)
    
    def _switch_to_cpu(self):
        """Switch all pipelines to CPU after GPU error"""
        if self._switched_to_cpu:
            return  # Already switched
        
        print("GPU error detected. Switching to CPU for remaining processing...")
        
        try:
            # Reinitialize all pipelines on CPU
            self.ner_pipeline = self._init_pipeline("ner", "SIRIS-Lab/citation-parser-ENTITY", "cpu", agg_strategy="simple")
            self.select_pipeline = self._init_pipeline("text-classification", "SIRIS-Lab/citation-parser-SELECT", "cpu")
            self.prescreening_pipeline = self._init_pipeline("text-classification", "SIRIS-Lab/citation-parser-TYPE", "cpu")
            self.span_pipeline = self._init_pipeline("ner", "SIRIS-Lab/citation-parser-SPAN", "cpu", agg_strategy="simple")
            
            self._switched_to_cpu = True
            print("Successfully switched to CPU")
            
        except Exception as e:
            print(f"Error switching to CPU: {e}")
            raise

    def _init_pipeline(
        self, task: str, model_path: str, device: Union[int, str], agg_strategy: Optional[str] = None
    ):
        # Helper to initialize the appropriate transformer pipeline with enhanced device handling
        try:
            kwargs = {
                "model": AutoModelForTokenClassification.from_pretrained(model_path)
                if task == "ner"
                else AutoModelForSequenceClassification.from_pretrained(model_path),
                "tokenizer": AutoTokenizer.from_pretrained(model_path),
                "device": device,
            }
            if agg_strategy:
                kwargs["aggregation_strategy"] = agg_strategy
            
            pipeline_obj = pipeline(task, **kwargs)
            
            # Verify device placement
            actual_device = next(pipeline_obj.model.parameters()).device
            
            # More specific logging based on model path
            model_name = model_path.split('/')[-1].replace('citation-parser-', '').upper()
            print(f"{model_name} model loaded on device: {actual_device}")
            
            return pipeline_obj
            
        except Exception as e:
            model_name = model_path.split('/')[-1].replace('citation-parser-', '').upper()
            print(f"Error loading {model_name} model on {device}: {e}")
            print("Falling back to CPU...")
            
            # Fallback to CPU
            kwargs["device"] = "cpu"
            pipeline_obj = pipeline(task, **kwargs)
            print(f"{model_name} model loaded on CPU (fallback)")
            return pipeline_obj

    def _generate_cache_key(self, citation: str, api_target: str, output: str) -> str:
        """Generate a consistent cache key for the given parameters"""
        # Create a hash of the parameters to handle long citations
        key_string = f"{citation}|{api_target}|{output}"
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()
    
    def _manage_cache_size(self):
        """Remove oldest entries if cache exceeds size limit"""
        if len(self._citation_cache) > self.cache_size_limit:
            # Remove oldest 20% of entries (simple FIFO)
            items_to_remove = len(self._citation_cache) - int(self.cache_size_limit * 0.8)
            keys_to_remove = list(self._citation_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self._citation_cache[key]
            self._cache_stats['size'] = len(self._citation_cache)
    
    def clear_cache(self):
        """Clear the citation cache and reset stats"""
        self._citation_cache.clear()
        self._cache_stats = {'hits': 0, 'misses': 0, 'size': 0}
        print("Citation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics"""
        total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_rate = (self._cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_enabled': self.enable_caching,
            'cache_size': len(self._citation_cache),
            'cache_limit': self.cache_size_limit,
            'cache_hits': self._cache_stats['hits'],
            'cache_misses': self._cache_stats['misses'],
            'hit_rate': f"{hit_rate:.1f}%",
            'total_requests': total_requests
        }

    def process_ner_entities(self, citation: str) -> Dict[str, List[str]]:
        """
        Run NER and merge adjacent fragments belonging to the same entity group.
        Example: "20" + "23" → "2023"
        """

        raw = self.ner_pipeline(citation)

        # STEP 1 — sort entities by start index
        raw = sorted(raw, key=lambda x: x["start"])

        merged = []
        current = None

        def flush():
            nonlocal current, merged
            if current:
                merged.append(current)
                current = None

        for ent in raw:
            group = ent["entity_group"]
            word = ent["word"]
            start = ent["start"]
            end = ent["end"]

            if current is None:
                current = {
                    "entity_group": group,
                    "word": word,
                    "start": start,
                    "end": end,
                    "score": ent["score"]
                }
                continue

            # Check if mergeable:
            same_group = (group == current["entity_group"])
            touching = (start <= current["end"] + 1)

            if same_group and touching:
                # merge text
                current["word"] += word
                current["end"] = end
                current["score"] = max(current["score"], ent["score"])
            else:
                flush()
                current = {
                    "entity_group": group,
                    "word": word,
                    "start": start,
                    "end": end,
                    "score": ent["score"]
                }

        flush()

        # STEP 2 — convert into dict like before
        entity_dict = {}
        for ent in merged:
            key = ent["entity_group"]
            entity_dict.setdefault(key, []).append(ent["word"])

        # STEP 3 — clean & validate using existing utility
        cleaned = EntityValidator.validate_and_clean_entities(entity_dict)

        return cleaned

    def generate_apa_citation(self, data: dict, api: str = "openalex") -> str:
        # Format a citation from retrieved metadata in APA style
        # PubMed now uses parsed dict structure like other APIs - no special handling needed
        formatter = citation_formatter.CitationFormatterFactory.get_formatter(api)
        return formatter.generate_apa_citation(data)
    
    def extract_fields_from_formatted_citation(self, formatted_citation: str, api: str) -> Dict[str, str]:
        """
        Extract fields from APA-formatted citation using pattern matching
        This complements NER extraction with more reliable pattern-based extraction
        """
        extracted = {}
        
        # Common APA citation patterns
        patterns = {
            'volume_issue_pages': r'(\d+)\s*\((\d+)\)\s*(\d+)-(\d+)',  # 188 (1) 33-42
            'volume_pages': r'(\d+)\s+(\d+)-(\d+)',  # 188 33-42
            'volume_issue': r'(\d+)\s*\((\d+)\)',  # 188 (1)
            'volume_only': r',\s*(\d+)\s*(?:\(|$)',  # Journal, 188
            'year': r'\((\d{4})\)',  # (2002)
            'doi': r'DOI:\s*(10\.\d+/[^\s]+)',  # DOI: 10.1034/...
            'pages_only': r'(\d+)-(\d+)\.',  # 33-42.
            'electronic_pages': r'(e\d+)',  # e123456
        }
        
        # Extract volume, issue, pages (try most specific first)
        volume_match = re.search(patterns['volume_issue_pages'], formatted_citation)
        if volume_match:
            extracted['volume'] = volume_match.group(1)
            extracted['issue'] = volume_match.group(2) 
            extracted['page_first'] = volume_match.group(3)
            extracted['page_last'] = volume_match.group(4)
        else:
            # Try volume with issue but no pages
            volume_issue_match = re.search(patterns['volume_issue'], formatted_citation)
            if volume_issue_match:
                extracted['volume'] = volume_issue_match.group(1)
                extracted['issue'] = volume_issue_match.group(2)
            else:
                # Try volume only
                volume_only_match = re.search(patterns['volume_only'], formatted_citation)
                if volume_only_match:
                    extracted['volume'] = volume_only_match.group(1)
            
            # Try to find pages separately
            pages_match = re.search(patterns['pages_only'], formatted_citation)
            if pages_match:
                extracted['page_first'] = pages_match.group(1)
                extracted['page_last'] = pages_match.group(2)
            else:
                # Check for electronic article IDs
                electronic_match = re.search(patterns['electronic_pages'], formatted_citation)
                if electronic_match:
                    extracted['page_first'] = electronic_match.group(1)
        
        # Extract year
        year_match = re.search(patterns['year'], formatted_citation)
        if year_match:
            extracted['publication_year'] = year_match.group(1)
        
        # Extract DOI
        doi_match = re.search(patterns['doi'], formatted_citation)
        if doi_match:
            extracted['doi'] = doi_match.group(1)
        
        return extracted

    def compute_ner_similarity(self, original_citation: str, candidate_citation: str) -> float:
        """
        Enhanced similarity computation using both NER and pattern extraction.
        Returns a score between 0.0 and 1.0, where 1.0 is perfect match.
        
        Improvements:
        - Uses pattern extraction to complement NER for formatted citations
        - Enhanced journal abbreviation matching
        - Better author surname comparison
        - More robust field extraction
        """
        import re
        from difflib import SequenceMatcher
        
        # Extract NER entities from both citations
        original_entities = self.process_ner_entities(original_citation)
        candidate_entities = self.process_ner_entities(candidate_citation)
        
        # Enhance candidate entities with pattern extraction from formatted citation
        pattern_fields = self.extract_fields_from_formatted_citation(candidate_citation, "openalex")
        for field, value in pattern_fields.items():
            field_upper = field.upper()
            if field_upper not in candidate_entities or not candidate_entities[field_upper]:
                candidate_entities[field_upper] = [value]

        if self.debug:
            print("\n--------------------------------------------")
            print(f"Citation: {original_citation}")
            print(f"    -> NER entities: {original_entities}")
            print()
            print(f"Candidate: {candidate_citation}")
            print(f"    -> NER entities: {candidate_entities}")
            print(f"    -> Pattern extracted: {pattern_fields}")
            print("--------------------------------------------")

        total_score = 0.0
        field_weights = {
            'DOI': 1,
            'TITLE': 0.7,
            'AUTHORS': 0.5,
            'JOURNAL': 0.2,
            'VOLUME': 0.2,
            'ISSUE': 0.2,
            'PAGE_FIRST': 0.2,
            'PUBLICATION_YEAR': 0.2
        }
        
        def normalize_text(text):
            """Normalize text for comparison"""
            if not text:
                return ""
            # Convert to lowercase, remove extra spaces, punctuation
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        def fuzzy_similarity(text1, text2):
            """Compute fuzzy string similarity using SequenceMatcher"""
            if not text1 or not text2:
                return 0.0
            norm1 = normalize_text(text1)
            norm2 = normalize_text(text2)
            if not norm1 or not norm2:
                return 0.0
            return SequenceMatcher(None, norm1, norm2).ratio()
        
        def author_similarity(authors1, authors2):
            """Enhanced author similarity using surname extraction with word overlap fallback"""
            if not authors1 or not authors2:
                return 0.0
            
            # Try surname extraction approach first
            try:
                # Extract surnames using the improved logic
                if hasattr(self, 'field_mapper'):
                    surname1 = self.field_mapper.extract_author_surname(authors1)
                    surname2 = self.field_mapper.extract_author_surname(authors2)
                    
                    if surname1 and surname2:
                        # Normalize surnames for comparison
                        norm_surname1 = normalize_text(surname1)
                        norm_surname2 = normalize_text(surname2)
                        
                        if norm_surname1 and norm_surname2:
                            # Direct surname comparison
                            surname_similarity = SequenceMatcher(None, norm_surname1, norm_surname2).ratio()
                            
                            # If surnames are very similar, return high score
                            if surname_similarity > 0.8:
                                return surname_similarity
                            
                            # Check if one surname is contained in the other (for different formats)
                            if norm_surname1 in norm_surname2 or norm_surname2 in norm_surname1:
                                return 0.9
            except:
                pass  # Fall back to word overlap method
            
            # Fallback to original word overlap method
            fuzzy_score = fuzzy_similarity(authors1, authors2)
            
            # Extract meaningful words (length > 2, not common words)
            def extract_author_words(text):
                # Clean text and extract words
                clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
                words = clean_text.split()
                # Filter out common non-name words and very short words
                stop_words = {'et', 'al', 'and', 'the', 'of', 'in', 'at', 'to', 'for', 'with'}
                return [w for w in words if len(w) > 2 and w not in stop_words and not w.isdigit()]
            
            words1 = set(extract_author_words(authors1))
            words2 = set(extract_author_words(authors2))
            
            if not words1 or not words2:
                return fuzzy_score
            
            # Calculate word overlap (Jaccard similarity)
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            word_overlap = intersection / union if union > 0 else 0.0
            
            # Combine fuzzy similarity with word overlap
            return max(fuzzy_score, word_overlap * 0.8)
        
        def journal_similarity(journal1, journal2):
            """Enhanced journal name matching with standard abbreviation patterns"""
            if not journal1 or not journal2:
                return 0.0
            
            def normalize_journal(journal):
                if not journal:
                    return ""
                normalized = journal.lower().strip()
                # Remove common prefixes/suffixes but keep review/reviews for abbreviation matching
                normalized = re.sub(r'\b(journal|proceedings|international|transactions)\b', '', normalized)
                normalized = re.sub(r'\b(of|the|and|for|in)\b', '', normalized)
                # Remove punctuation and normalize whitespace
                normalized = re.sub(r'[^\w\s]', ' ', normalized)
                normalized = re.sub(r'\s+', ' ', normalized).strip()
                return normalized
            
            norm1 = normalize_journal(journal1)
            norm2 = normalize_journal(journal2)
            
            if not norm1 or not norm2:
                return 0.0
            
            # Exact match after normalization
            if norm1 == norm2:
                return 1.0
            
            def check_abbreviation_match(abbrev, full):
                """Check if abbrev could be a standard abbreviation of full"""
                abbrev_words = abbrev.split()
                full_words = full.split()
                
                if len(abbrev_words) != len(full_words):
                    return False
                
                # Check if each abbreviated word matches the start of the full word
                for abbrev_word, full_word in zip(abbrev_words, full_words):
                    # Remove trailing dots from abbreviation
                    clean_abbrev = abbrev_word.rstrip('.')
                    # Standard abbreviation: first few letters match
                    if not full_word.startswith(clean_abbrev.lower()):
                        return False
                
                return True
            
            # Test both directions (one could be abbreviation of the other)
            if check_abbreviation_match(norm1, norm2) or check_abbreviation_match(norm2, norm1):
                return 0.95
            
            # Check for partial word matches (common in journal abbreviations)
            words1 = set(norm1.split())
            words2 = set(norm2.split())
            
            if words1 and words2:
                # Calculate word overlap with abbreviation consideration
                matches = 0
                for w1 in words1:
                    for w2 in words2:
                        # Exact match
                        if w1 == w2:
                            matches += 1
                            break
                        # Abbreviation match (one starts with the other)
                        elif w1.startswith(w2) or w2.startswith(w1):
                            matches += 0.8
                            break
                
                overlap_score = matches / max(len(words1), len(words2))
                if overlap_score > 0.6:
                    return overlap_score
            
            # Fuzzy matching fallback
            return SequenceMatcher(None, norm1, norm2).ratio() * 0.7
        
        def extract_any_year(text):
            """Extract any 4-digit year from text"""
            if not text:
                return None
            year_match = re.search(r'\b(19|20)\d{2}\b', str(text))
            return int(year_match.group()) if year_match else None
        
        def year_similarity(year1, year2):
            """Year matching with fallback extraction from any field"""
            # Try direct extraction first
            y1 = extract_any_year(year1)
            y2 = extract_any_year(year2)
            
            if y1 is None or y2 is None:
                return 0.0
            
            # Year comparison with tolerance
            diff = abs(y1 - y2)
            if diff == 0:
                return 1.0
            elif diff == 1:
                return 0.7  # 1-year difference (common for online/print dates)
            elif diff == 2:
                return 0.3  # 2-year difference
            else:
                return 0.0
        
        def doi_similarity(doi1, doi2):
            """DOI exact matching"""
            if not doi1 or not doi2:
                return 0.0
            
            # Normalize DOIs (remove prefixes)
            def normalize_doi(doi):
                doi = doi.lower().strip()
                prefixes = ['https://doi.org/', 'http://doi.org/', 'doi:', 'doi ']
                for prefix in prefixes:
                    if doi.startswith(prefix):
                        doi = doi[len(prefix):]
                return doi.strip()
            
            norm_doi1 = normalize_doi(doi1)
            norm_doi2 = normalize_doi(doi2)
            
            return 1.0 if norm_doi1 == norm_doi2 else 0.0
        
        def volume_similarity(vol1, vol2):
            """Volume exact matching with numeric extraction"""
            if not vol1 or not vol2:
                return 0.0
            
            # Extract numeric volume
            def extract_volume_number(vol_text):
                if not vol_text:
                    return None
                # Extract first number sequence
                vol_match = re.search(r'\d+', str(vol_text))
                return vol_match.group() if vol_match else None
            
            num1 = extract_volume_number(vol1)
            num2 = extract_volume_number(vol2)
            
            if num1 and num2:
                return 1.0 if num1 == num2 else 0.0
            
            return 0.0
        
        def issue_similarity(issue1, issue2):
            """Issue exact matching with numeric extraction"""
            if not issue1 or not issue2:
                return 0.0
            
            # Extract numeric issue
            def extract_issue_number(issue_text):
                if not issue_text:
                    return None
                # Extract first number sequence
                issue_match = re.search(r'\d+', str(issue_text))
                return issue_match.group() if issue_match else None
            
            num1 = extract_issue_number(issue1)
            num2 = extract_issue_number(issue2)
            
            if num1 and num2:
                return 1.0 if num1 == num2 else 0.0
            
            return 0.0
        
        def page_similarity(page1, page2):
            """Page number matching with electronic article ID support"""
            if not page1 or not page2:
                return 0.0
            
            def extract_page_info(page_text):
                if not page_text:
                    return None
                
                page_text = str(page_text).strip()
                
                # Check for electronic article identifiers (e.g., e123456, e1234)
                e_article_match = re.search(r'\be\d+\b', page_text, re.IGNORECASE)
                if e_article_match:
                    return ('electronic', e_article_match.group().lower())
                
                # Extract regular page numbers
                page_match = re.search(r'\d+', page_text)
                if page_match:
                    return ('numeric', page_match.group())
                
                return None
            
            page_info1 = extract_page_info(page1)
            page_info2 = extract_page_info(page2)
            
            if page_info1 and page_info2:
                type1, value1 = page_info1
                type2, value2 = page_info2
                
                # Both must be same type and same value
                if type1 == type2 and value1 == value2:
                    return 1.0
            
            return 0.0
        
        # Calculate field-specific similarities with fallback extraction
        field_scores = {}
        
        # Create combined text for fallback searches
        orig_combined = f"{original_citation} " + " ".join([" ".join(v) for v in original_entities.values()])
        cand_combined = f"{candidate_citation} " + " ".join([" ".join(v) for v in candidate_entities.values()])
        
        for field, weight in field_weights.items():
            orig_value = original_entities.get(field, [])
            cand_value = candidate_entities.get(field, [])
            
            # Get first value from lists (cleaned by EntityValidator)
            orig_text = orig_value[0] if orig_value else ""
            cand_text = cand_value[0] if cand_value else ""
            
            # Generic fallback: if field not found in NER, try extracting from full text
            if not orig_text and field == 'PUBLICATION_YEAR':
                orig_text = str(extract_any_year(orig_combined) or "")
            if not cand_text and field == 'PUBLICATION_YEAR':
                cand_text = str(extract_any_year(cand_combined) or "")
            
            # Apply field-specific similarity functions
            if field == 'TITLE':
                field_scores[field] = fuzzy_similarity(orig_text, cand_text)
            elif field == 'AUTHORS':
                field_scores[field] = author_similarity(orig_text, cand_text)
            elif field == 'PUBLICATION_YEAR':
                field_scores[field] = year_similarity(orig_text, cand_text)
            elif field == 'DOI':
                field_scores[field] = doi_similarity(orig_text, cand_text)
            elif field == 'JOURNAL':
                field_scores[field] = journal_similarity(orig_text, cand_text)
            elif field == 'VOLUME':
                field_scores[field] = volume_similarity(orig_text, cand_text)
            elif field == 'ISSUE':
                field_scores[field] = issue_similarity(orig_text, cand_text)
            elif field == 'PAGE_FIRST':
                field_scores[field] = page_similarity(orig_text, cand_text)
            else:
                field_scores[field] = 0.0
            
            # Add weighted score
            total_score += field_scores[field] * weight
            
        if self.debug:
            print(f"=> Field scores: {field_scores}")
        
        return min(total_score, 1.0)  # Ensure score doesn't exceed 1.0
    
    def get_highest_true_position(
        self, outputs: List[List[Dict[str, Any]]], inputs: List[Any], 
        original_citation: str = None, api_target: str = None
    ) -> Tuple[Optional[Any], Optional[float]]:

        if self.debug:
            print(f"\n=== DEBUG: get_highest_true_position ===")
            print(f"SELECT threshold={self.select_threshold}, NER threshold={self.ner_threshold}")
            print(f"Number of candidates: {len(inputs)}")

        confident_true_scores = []
        uncertain_true_scores = []
        any_true = False     # ← NEW FLAG

        # ---- STEP 1: Analyze SELECT model outputs ----
        for i, result in enumerate(outputs):
            label = result[0]["label"]
            score = result[0]["score"]

            if self.debug:
                print(f"Candidate {i} → label={label}, score={score:.3f}")

            if label is True:
                any_true = True
                if score >= self.select_threshold:
                    confident_true_scores.append((i, score))
                else:
                    uncertain_true_scores.append((i, score))

        # ---- NEW RULE: If ALL labels are False → STOP ----
        if not any_true:
            if self.debug:
                print("\n✗ SELECT model says ALL candidates are FALSE → No match.")
            return None, None
        # ---------------------------------------------------

        if self.debug:
            print(f"\nSUMMARY: {len(confident_true_scores)} confident True, "
                f"{len(uncertain_true_scores)} uncertain True")

        # ---- STEP 2: Use confident True labels ----
        if confident_true_scores:
            if len(confident_true_scores) > 1:
                best_index, best_score = self._resolve_confident_candidates_tie(
                    confident_true_scores, inputs, original_citation, api_target
                )
            else:
                best_index, best_score = confident_true_scores[0]

            self._last_scoring_method = "select_model"

            if self.debug:
                print(f"✓ Using SELECT confident match: {best_index}, score={best_score:.3f}")

            return inputs[best_index], best_score

        # ---- STEP 3: NER validation of uncertain True ----
        if uncertain_true_scores and original_citation and api_target:

            validated = []
            for i, sel_score in uncertain_true_scores:
                formatted = self.generate_apa_citation(inputs[i], api=api_target)
                ner_score = self.compute_ner_similarity(original_citation, formatted)

                if ner_score >= self.ner_threshold:
                    validated.append((i, ner_score))

            if validated:
                best_index, best_score = max(validated, key=lambda x: x[1])
                self._last_scoring_method = "ner_similarity"
                return inputs[best_index], best_score

            if self.debug:
                print("✗ No uncertain True labels passed NER validation.")
        

        
        # If only uncertain True labels, validate with NER
        if uncertain_true_scores and original_citation and api_target:
            if self.debug:
                print(f"\n--- Validating uncertain True labels with NER similarity ---")
            
            validated_candidates = []
            for i, select_score in uncertain_true_scores:
                try:
                    formatted_citation = self.generate_apa_citation(inputs[i], api=api_target)
                    ner_score = self.compute_ner_similarity(original_citation, formatted_citation)
                    
                    if ner_score >= self.ner_threshold:
                        validated_candidates.append((i, ner_score))
                        if self.debug:
                            print(f"  ✓ PASSED NER validation (>= {self.ner_threshold})")
                    else:
                        if self.debug:
                            print(f"  ✗ FAILED NER validation (< {self.ner_threshold})")
                            
                except Exception as e:
                    if self.debug:
                        print(f"\nCandidate {i}: ERROR computing NER - {e}")
            
            if validated_candidates:
                best_index, best_score = max(validated_candidates, key=lambda x: x[1])
                self._last_scoring_method = 'ner_similarity'
                
                if self.debug:
                    print(f"\n✓ DECISION: Using NER-validated candidate")
                    print(f"  Selected candidate {best_index} with NER score {best_score:.3f}")
                
                return inputs[best_index], best_score
            elif self.debug:
                print(f"\n✗ No candidates passed NER validation")
        
        # Fallback to NER-only similarity (no True labels or validation failed)
        if self.debug:
            print(f"\n--- Fallback: Computing NER similarity for all candidates ---")
        
        if not original_citation or not api_target:
            if self.debug:
                print(f"✗ Cannot compute NER similarity - missing parameters")
            return None, None
        
        ner_scores = []
        for i, pub in enumerate(inputs):
            try:
                formatted_citation = self.generate_apa_citation(pub, api=api_target)
                ner_score = self.compute_ner_similarity(original_citation, formatted_citation)
                ner_scores.append((i, ner_score))
                    
            except Exception as e:
                if self.debug:
                    print(f"\nCandidate {i}: ERROR computing NER - {e}")
                ner_scores.append((i, 0.0))
        
        if ner_scores:
            best_index, best_score = max(ner_scores, key=lambda x: x[1])
            
            if self.debug:
                print(f"\nBest NER candidate: {best_index} with score {best_score:.3f}")
            
            # Apply NER threshold even in fallback
            if best_score < self.ner_threshold:
                if self.debug:
                    print(f"✗ DECISION: Best NER score {best_score:.3f} below threshold {self.ner_threshold}, rejecting all")
                return None, None
            
            self._last_scoring_method = 'ner_similarity'
            
            if self.debug:
                print(f"✓ DECISION: Using best NER candidate {best_index} with score {best_score:.3f}")
            
            return inputs[best_index], best_score
        
        if self.debug:
            print(f"✗ DECISION: No valid candidates found")
        
        return None, None

    def _resolve_confident_candidates_tie(
        self, confident_candidates: List[Tuple[int, float]], inputs: List[Any], 
        original_citation: str, api_target: str
    ) -> Tuple[int, float]:
        """
        Resolve ties between confident True candidates using NER similarity as tiebreaker.
        
        Args:
            confident_candidates: List of (index, score) tuples for confident True candidates
            inputs: List of publication candidates
            original_citation: Original citation text
            api_target: Target API name
            
        Returns:
            Tuple of (best_index, best_score) from the original SELECT scores
        """
        if self.debug:
            print(f"\n--- Resolving tie between {len(confident_candidates)} confident candidates ---")
        
        # Set tie threshold (1% difference)
        tie_threshold = 0.01
        
        # Sort candidates by SELECT score (highest first)
        sorted_candidates = sorted(confident_candidates, key=lambda x: x[1], reverse=True)
        top_score = sorted_candidates[0][1]
        
        # Find all candidates within tie threshold
        tied_candidates = []
        for idx, score in sorted_candidates:
            score_diff = abs(score - top_score)
            if score_diff <= tie_threshold:
                tied_candidates.append((idx, score))
                if self.debug:
                    print(f"  Candidate {idx}: SELECT score {score:.3f} (diff: {score_diff:.3f}) - IN TIE")
            else:
                if self.debug:
                    print(f"  Candidate {idx}: SELECT score {score:.3f} (diff: {score_diff:.3f}) - CLEAR LOSER")
        
        # If only one candidate in the tie group, return it
        if len(tied_candidates) == 1:
            if self.debug:
                print(f"  No actual tie found, using highest scorer")
            return tied_candidates[0]
        
        if self.debug:
            print(f"  Actual tie detected between {len(tied_candidates)} candidates")
            print(f"  Using NER similarity as tiebreaker...")
        
        # Compute NER similarity for tied candidates only
        ner_results = []
        for idx, select_score in tied_candidates:
            try:
                formatted_citation = self.generate_apa_citation(inputs[idx], api=api_target)
                ner_score = self.compute_ner_similarity(original_citation, formatted_citation)
                ner_results.append((idx, select_score, ner_score))
                
                if self.debug:
                    print(f"    Candidate {idx}: SELECT={select_score:.3f}, NER={ner_score:.3f}")
                    print(f"      Citation: {formatted_citation[:100]}...")
                    
            except Exception as e:
                if self.debug:
                    print(f"    Candidate {idx}: ERROR computing NER similarity - {e}")
                ner_results.append((idx, select_score, 0.0))
        
        # Select candidate with highest NER similarity (but return original SELECT score)
        if ner_results:
            best_result = max(ner_results, key=lambda x: x[2])  # Sort by NER score
            best_index, best_select_score, best_ner_score = best_result
            
            if self.debug:
                print(f"  TIE RESOLVED: Candidate {best_index} wins with NER score {best_ner_score:.3f}")
                print(f"  Returning original SELECT score: {best_select_score:.3f}")
            
            # Return the original SELECT score, not the NER score
            return best_index, best_select_score
        
        # Fallback: return highest SELECT score if NER computation failed
        if self.debug:
            print(f"  TIE RESOLUTION FAILED: Falling back to highest SELECT score")
        return sorted_candidates[0]

    def search_api(self, ner_entities: Dict[str, List[str]], api: str = "openalex", 
                      target_count: int = 10) -> List[dict]:
            # Search a bibliographic API using extracted NER entities with progressive strategy
            return self.searcher.search_api(ner_entities, api=api, target_count=target_count)
        
    def get_uri(self, pid: Optional[str], doi: Optional[str], api: str) -> Optional[str]:
        # Construct the canonical URL to the publication based on available identifiers
        uri_templates = {
            "openalex": lambda: f"https://openalex.org/{pid}" if pid else None,
            "openaire": lambda: f"https://explore.openaire.eu/search/publication?pid={doi or pid}" if pid else None,
            "pubmed": lambda: f"https://pubmed.ncbi.nlm.nih.gov/{pid}" if pid else None,
            "crossref": lambda: f"https://doi.org/{doi}" if doi else None,
            "hal": lambda: f"https://hal.science/{pid}" if pid else None,
        }
        return uri_templates.get(api, lambda: None)()

    def extract_id(self, publication: dict, api: str) -> Optional[str]:
        # Extract the publication ID depending on the API source
        if publication and isinstance(publication, dict):
            if api == "openalex":
                return publication.get("id", "").replace("https://openalex.org/", "")
            elif api == "openaire":
                return publication.get('id')
            elif api == "pubmed":
                return publication.get("pmid") or publication.get("id")
            elif api == "crossref":
                return publication.get("DOI")
            elif api == "hal":
                return publication.get("halId_s")
        return None

    def extract_doi(self, publication: dict, api: str) -> Optional[str]:
        """Extract the DOI depending on the API source - Updated for PubMed dict structure"""
        
        # First try to get from enhanced structure
        if isinstance(publication, dict) and 'main_doi' in publication:
            return publication.get('main_doi')
        
        # Handle PubMed - now expects parsed dict structure
        if api == "pubmed":
            if isinstance(publication, dict):
                # Try direct DOI field from parsed structure
                doi = publication.get('doi')
                if doi:
                    return doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
                
                # Fallback: if still has xml_content, try parsing
                if 'xml_content' in publication:
                    try:
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(publication['xml_content'])
                        doi_elem = root.find(".//ELocationID[@EIdType='doi']")
                        if doi_elem is not None:
                            return doi_elem.text.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
                    except Exception:
                        pass
            return None
        
        # Fallback to original extraction logic for other APIs
        if api == "openalex":
            doi = publication.get("doi")
            if isinstance(doi, str):
                return doi.replace("https://doi.org/", "")
            return None
        elif api == "openaire":
            # Check enhanced structure first
            if 'pids' in publication:
                identifiers = publication.get('pids', [])
                if isinstance(identifiers, dict):
                    identifiers = [identifiers]
                for pid in identifiers:
                    if isinstance(pid, dict) and pid.get("scheme") == "doi":
                        return pid.get("value", "").replace("https://doi.org/", "")
            return None
        elif api == "crossref":
            return publication.get("DOI")
        elif api == "hal":
            return publication.get("doiId_s")
        return None

    def link_citation(self, citation: str, output: str = 'simple', api_target: str = 'openalex') -> Dict[str, Any]:
        """
        Main function to process a citation string with caching support:
        - Check if it's a valid citation
        - Extract entities
        - Search target API
        - Format results and rank them
        """
        
        try:
        
        
            # Check cache first if enabled
            if self.enable_caching:
                cache_key = self._generate_cache_key(citation, api_target, output)
                if cache_key in self._citation_cache:
                    self._cache_stats['hits'] += 1
                    return self._citation_cache[cache_key].copy()  # Return copy to avoid mutation
                else:
                    self._cache_stats['misses'] += 1

            try:

                # Prescreen input to ensure it's likely a citation
                if not self.prescreening_pipeline(citation)[0]["label"]:
                    result = {"error": "This text is not a citation. Please introduce a valid citation."}
                    if self.enable_caching:
                        self._citation_cache[cache_key] = result.copy()
                        self._manage_cache_size()
                    return result

                ner_entities = self.process_ner_entities(citation)
                
            except Exception as e:
                if self._is_cuda_error(e) and not self._switched_to_cpu:
                    self._switch_to_cpu()
                    # Retry with CPU
                    if not self.prescreening_pipeline(citation)[0]["label"]:
                        result = {"error": "This text is not a citation. Please introduce a valid citation."}
                        if self.enable_caching:
                            self._citation_cache[cache_key] = result.copy()
                            self._manage_cache_size()
                        return result
                    ner_entities = self.process_ner_entities(citation)
                else:
                    raise
                
            pubs = self.search_api(ner_entities, api=api_target, target_count=10)
            if self.debug:
                print(f"\n=== DEBUG: Citation Selection Phase ===")
                print(f"Found {len(pubs)} candidates from search")
                for i, pub in enumerate(pubs):
                    title = pub.get('title', 'No title')[:50]
                    doi = pub.get('doi', 'No DOI')
                    print(f"  Candidate {i}: {title}... | DOI: {doi}")

            if not pubs:
                result = {}
                if self.enable_caching:
                    self._citation_cache[cache_key] = result.copy()
                    self._manage_cache_size()
                return result

            # Format candidate citations and classify best match
            cits = [self.generate_apa_citation(pub, api=api_target) for pub in pubs]
            if self.debug:
                print(f"\nFormatted citations:\n")
                for i, cit in enumerate(cits):
                    print(f"-  Citation {i}: {cit}\n")
            
            try:
                pairwise_scores = [self.select_pipeline(f"{citation} [SEP] {cit}") for cit in cits]
                if self.debug:
                    print(f"\nPairwise scores:")
                    for i, score in enumerate(pairwise_scores):
                        print(f"  Score {i}: {score}")
                
            except Exception as e:
                if self._is_cuda_error(e) and not self._switched_to_cpu:
                    self._switch_to_cpu()
                    # Retry with CPU
                    pairwise_scores = [self.select_pipeline(f"{citation} [SEP] {cit}") for cit in cits]
                else:
                    raise

            # If only one candidate, check if SELECT model is confident
            if len(cits) == 1:
                selected_score = pairwise_scores[0][0]
                pub = pubs[0]
                
                # Check if SELECT model returned False OR low confidence - use NER similarity
                if selected_score['label'] is True and selected_score['score'] < self.select_threshold:
                    ner_score = self.compute_ner_similarity(citation, cits[0])
                    if self.debug:
                        print(f"\nNER score: {ner_score}")
                    
                    # Apply NER threshold
                    if ner_score < self.ner_threshold:
                        result = {}
                        if self.enable_caching:
                            self._citation_cache[cache_key] = result.copy()
                            self._manage_cache_size()
                        return result
                    
                    # Use NER score
                    final_score = ner_score
                    score_data = {"score": final_score}
                else:
                    # SELECT model confident and True
                    final_score = selected_score['score']
                    score_data = selected_score
                
                pub_id = self.extract_id(pub, api_target)
                pub_doi = self.extract_doi(pub, api_target)
                url = self.get_uri(pub_id, pub_doi, api_target)
                result = self._format_result(cits[0], score_data, pub_id, pub_doi, url, pub, output, api_target)
                
                if self.enable_caching:
                    self._citation_cache[cache_key] = result.copy()
                    self._manage_cache_size()
                return result

            # Choose the most likely correct match using classification scores
            reranked_pub, best_score = self.get_highest_true_position(
                pairwise_scores, pubs, citation, api_target
            )
            if reranked_pub:
                pub_id = self.extract_id(reranked_pub, api_target)
                pub_doi = self.extract_doi(reranked_pub, api_target)
                url = self.get_uri(pub_id, pub_doi, api_target)
                formatted_cit = self.generate_apa_citation(reranked_pub, api=api_target)
                result = self._format_result(formatted_cit, {"score": best_score}, pub_id, pub_doi, url, reranked_pub, output, api_target)
                
                if self.enable_caching:
                    self._citation_cache[cache_key] = result.copy()
                    self._manage_cache_size()
                return result

            result = {}
            if self.enable_caching:
                self._citation_cache[cache_key] = result.copy()
                self._manage_cache_size()
            
            return result
            
        except Exception as e:
            if self._is_cuda_error(e):
                print(f"CUDA error in link_citation: {e}")
                if not self._switched_to_cpu:
                    try:
                        self._switch_to_cpu()
                        # Retry the entire operation with CPU
                        return self.link_citation(citation, output, api_target)
                    except Exception as retry_error:
                        result = {"error": f"Failed even after switching to CPU: {retry_error}"}
                        if self.enable_caching:
                            cache_key = self._generate_cache_key(citation, api_target, output)
                            self._citation_cache[cache_key] = result.copy()
                            self._manage_cache_size()
                        return result
            raise

    def _format_result(
        self, citation: str, score_data: dict, pub_id: Optional[str], doi: Optional[str],
        url: Optional[str], pub: dict, output: str, api_target: str
    ) -> Dict[str, Any]:
        """Helper to format the output result with enhanced DOI support"""
        
        # Extract enhanced DOI information from publication
        main_doi = pub.get('main_doi')
        alternative_dois = pub.get('alternative_dois', [])
        total_dois = pub.get('total_dois', 0)
        all_dois = pub.get('all_dois', [])
        
        # If enhanced structure not available, use legacy DOI
        if not main_doi:
            main_doi = doi
            alternative_dois = []
            total_dois = 1 if doi else 0
            all_dois = [doi] if doi else []
        
        result = {
            "result": citation,
            "score": score_data.get("score", False),
            f"{api_target}_id": pub_id,
            "doi": main_doi,  # Backward compatibility - use main DOI
            "url": url,
            # Enhanced DOI information
            "main_doi": main_doi,
            "alternative_dois": alternative_dois,
            "total_dois": total_dois,
            "all_dois": all_dois
        }
        
        if output == "advanced":
            result["full-publication"] = pub
        
        return result

    def _should_include_in_ensemble_voting(self, result: Dict[str, Any], original_citation: str, api: str) -> bool:
        """
        Determine if a result should be included in ensemble DOI voting.
        
        Applies the same quality thresholds that individual APIs use to ensure
        ensemble only considers results that would pass individual API validation.
        
        Args:
            result: The API result to evaluate
            original_citation: The original citation text for similarity comparison
            api: The source API name
            
        Returns:
            bool: True if result should contribute to ensemble voting, False otherwise
        """
        # Must have a DOI to contribute to ensemble voting
        main_doi = result.get('main_doi') or result.get('doi')
        if not main_doi:
            return False
        
        # Apply same quality thresholds as individual APIs
        formatted_citation = result.get('result')
        if formatted_citation:
            try:
                ner_score = self.compute_ner_similarity(original_citation, formatted_citation)
                if ner_score < self.ner_threshold:
                    return False
            except Exception:
                return False
        
        return True

    def link_citation_ensemble(
            self, citation: str,
            output: str = 'simple',
            api_targets: List[str] = ['openalex', 'openaire', 'pubmed', 'crossref', 'hal']
        ) -> Dict[str, Any]:
            """
            Attempts to link a citation using multiple APIs in an ensemble fashion.
            Enhanced to consider alternative DOIs in voting for improved accuracy.
            Selects the most agreed-upon DOI among sources.
            Now benefits from caching - individual API calls may be cached.
            """
            doi_counter = Counter()
            extract_ids = {}
            missing_sources = []
            api_results = {}  # Store full results for enhanced processing

            # Try to link using each API (these calls will use cache if available)
            for api in api_targets:
                try:
                    res = self.link_citation(citation, output=output, api_target=api)

                    if not self._should_include_in_ensemble_voting(res, citation, api):
                        missing_sources.append(api)
                        continue

                    # Store full result for enhanced processing
                    api_results[api] = res
                    
                    # Extract main DOI and alternative DOIs if available
                    main_doi = res.get("doi")
                    alternative_dois = res.get("alternative_dois", [])
                    
                    if main_doi:
                        # Enhanced voting: Count main DOI
                        doi_counter[main_doi] += 1.0
                        
                        # Store the API ID for this DOI
                        api_id_key = f"{api}_id"
                        extract_ids[api_id_key] = res.get(api_id_key, None)
                    
                    # Enhanced voting: Count alternative DOIs with equal weight
                    if alternative_dois:
                        for alt_doi in alternative_dois:
                            if alt_doi and alt_doi != main_doi:  # Avoid double-counting
                                doi_counter[alt_doi] += 1.0
                    
                    # If no DOIs found, mark as missing
                    if not main_doi and not alternative_dois:
                        missing_sources.append(api)
                        
                except Exception as e:
                    missing_sources.append(api)

            # Enhanced ensemble decision
            if not doi_counter:
                return {"doi": None, "external_ids": {}}

            # Choose DOI with most votes (main + alternative DOIs counted equally)
            best_doi, vote_count = doi_counter.most_common(1)[0]

            # Enhanced backfill: Try to get IDs for the selected DOI from APIs that missed it
            for api in missing_sources:
                try:
                    # Search specifically for the selected DOI (this will also use cache)
                    pubs = self.search_api({'DOI': [best_doi]}, api=api, target_count=1)
                    if pubs:
                        pub_id = self.extract_id(pubs[0], api) 
                        if pub_id:
                            extract_ids[f"{api}_id"] = pub_id
                except Exception as e:
                    pass  # Fail silently for backfill

            # Enhanced result: Include voting information and alternative DOI info
            result = {
                "doi": best_doi,
                "external_ids": extract_ids
            }
            
            # Add enhanced ensemble metadata
            if output == 'advanced':
                # Find all DOIs that point to the same paper
                related_dois = [doi for doi, votes in doi_counter.items() if votes > 0]
                
                # Count contributing APIs
                contributing_apis = []
                for api, api_result in api_results.items():
                    main_doi = api_result.get("doi")
                    alternative_dois = api_result.get("alternative_dois", [])
                    
                    # Check if this API contributed to the selected DOI
                    if (main_doi == best_doi or 
                        best_doi in alternative_dois or
                        main_doi in related_dois):
                        contributing_apis.append(api)
                
                result["ensemble_metadata"] = {
                    "selected_doi_votes": vote_count,
                    "total_dois_found": len(related_dois),
                    "all_related_dois": related_dois,
                    "contributing_apis": contributing_apis,
                    "doi_vote_breakdown": dict(doi_counter.most_common())
                }

            return result
    
    def _debug_print_references(self, valid_refs, invalid_refs):
        print("\n==================== VALID REFERENCES ====================")
        for i, ref in enumerate(valid_refs, 1):
            print(f"{i:2d}. {ref.get('id','')} → {ref.get('text','')}")

        print("\n==================== INVALID REFERENCES ====================")
        for i, ref in enumerate(invalid_refs, 1):
            print(f"{i:2d}. {ref.get('id','')} → {ref.get('text','')}")

    def extract_and_link_from_text(self, text: str, api_target: str = "openalex", plot=False):

        extracted = extract_references_and_mentions(text, self.span_pipeline, plot=plot)

        references = extracted["references"]
        mentions = extracted["mentions"]

        print("🔦 Running prescreening classsifier on references...")
        screened_refs = prescreen_references(references, self.prescreening_pipeline)
        invalid_refs = [r for r in references if r not in screened_refs]

        print(f"📘 Total references extracted: {len(references)} ( ✔️ {len(screened_refs)} valid | ❌ {len(invalid_refs)} invalid )")


        # Debug printing
        if self.debug:
            self._debug_print_references(screened_refs, invalid_refs)

        print("🔗 Linking references\n")

        linked_references = []
        for ref in tqdm(screened_refs, desc="Linking References", unit=" reference"):
            ref_text = ref["text"]

            ref_ner = self.process_ner_entities(ref_text)
            ref['ner'] = ref_ner

            try:
                linked = self.link_citation(ref_text, api_target=api_target)
            except Exception as e:
                linked = {"error": str(e)}

            matched_mentions = match_mentions_to_reference(ref, mentions)

            ref.pop("ner", None)

            linked_references.append({
                **ref,          # keep id, text, start, end
                "linked": linked,
                "claims_in_context": matched_mentions
            })

            if self.debug:
                print(f"• {ref_text[:60]}… → {linked.get('id', 'NO MATCH')}")

        print(f"✅ Linked {len(linked_references)} references.\n")

        # Return everything nicely packaged
        return {
            "references": linked_references,  # ← now includes linked metadata
            "invalid_references": invalid_refs,
            "mentions": mentions
        }