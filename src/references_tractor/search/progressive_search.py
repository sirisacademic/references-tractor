# search/progressive_search.py
"""
Implements progressive search strategy with target-based candidate collection.
Enhanced with multiple DOI deduplication support and API-specific retry configurations.
Clean output with reduced verbosity.
"""

import time
import requests
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

from .api_capabilities import APICapabilities
from .field_mapper import FieldMapper, DOIResult

PER_PAGE_SPECIFIC = 5
PER_PAGE_BROAD = 10

class ResultDeduplicator:
    """Handles deduplication of search results across different strategies with multiple DOI support"""
    
    def __init__(self, field_mapper: FieldMapper):
        self.field_mapper = field_mapper

    def deduplicate_candidates(self, candidates: List[Dict], api: str) -> List[Dict]:
        """Remove duplicate candidates based on DOI or publication ID with multiple DOI awareness"""
        seen_dois = set()
        seen_ids = set()
        unique_candidates = []
        
        response_fields = APICapabilities.get_response_fields(api)
        
        for candidate in candidates:
            should_keep = True
            candidate_dois = set()
            candidate_id = None
            
            # Extract all DOIs from this candidate (if enhanced with multiple DOI support)
            if 'all_dois' in candidate and candidate['all_dois']:
                candidate_dois.update(candidate['all_dois'])
            else:
                # Fallback: extract DOI using field mapper
                try:
                    doi_result = self.field_mapper.extract_dois_from_result(candidate, api)
                    candidate_dois.update(self.field_mapper.get_all_dois_from_result(doi_result))
                except Exception:
                    # Final fallback: try to get DOI from standard fields
                    doi_config = response_fields.get('doi')
                    if doi_config:
                        doi = self.field_mapper.extract_response_field(candidate, doi_config, api)
                        if doi:
                            cleaned_doi = self.field_mapper.clean_doi(doi)
                            if cleaned_doi:
                                candidate_dois.add(cleaned_doi)
            
            # Check for DOI overlap with previously seen candidates
            if candidate_dois:
                doi_overlap = candidate_dois.intersection(seen_dois)
                if doi_overlap:
                    should_keep = False
                else:
                    seen_dois.update(candidate_dois)
            
            # If no DOI available or no overlap, check by publication ID
            if should_keep and not candidate_dois:
                id_config = response_fields.get('id')
                if id_config:
                    candidate_id = self.field_mapper.extract_response_field(candidate, id_config, api)
                    if candidate_id and candidate_id in seen_ids:
                        should_keep = False
                    elif candidate_id:
                        seen_ids.add(candidate_id)
            
            if should_keep:
                candidate['dedup_info'] = {
                    'dois_found': list(candidate_dois),
                    'id_found': candidate_id,
                    'dedup_method': 'doi' if candidate_dois else ('id' if candidate_id else 'title_year')
                }
                unique_candidates.append(candidate)
        
        return unique_candidates

class ProgressiveSearchStrategy:
    """Implements target-based progressive search with API-specific retry configurations and clean output"""
    
    def __init__(self, field_mapper: FieldMapper, deduplicator: 'ResultDeduplicator', 
                 default_target_count: int = 10, debug: bool = False):
        self.field_mapper = field_mapper
        self.deduplicator = deduplicator
        self.default_target_count = default_target_count
        self.debug = debug
        
        # Track API call timing to respect rate limits
        self.last_api_call_time = {}
    
    def _get_api_retry_config(self, api: str):
        """Get retry configuration for specific API"""
        return APICapabilities.get_retry_config(api)
    
    def _wait_for_rate_limit(self, api: str, rate_limit_delay: float):
        """Wait for rate limit before making API call with API-specific delay"""
        current_time = time.time()
        last_call_time = self.last_api_call_time.get(api, 0)
        
        time_since_last_call = current_time - last_call_time
        if time_since_last_call < rate_limit_delay:
            sleep_time = rate_limit_delay - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_api_call_time[api] = time.time()
    
    def _calculate_delay(self, attempt: int, api: str, config) -> float:
        """Calculate delay for retry attempt with API-specific configuration"""
        delay = config.base_delay * (config.backoff_multiplier ** attempt)
        return min(delay, config.max_delay)
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is worth retrying"""
        if isinstance(error, requests.exceptions.Timeout):
            return True
        elif isinstance(error, requests.exceptions.ConnectionError):
            return True
        elif isinstance(error, requests.exceptions.HTTPError):
            if hasattr(error, 'response') and error.response:
                status_code = error.response.status_code
                return 500 <= status_code < 600
            return True
        elif isinstance(error, requests.exceptions.RequestException):
            return True
        else:
            return False
        
    def _execute_search_with_retry(self, query_params: Dict[str, Any], api: str, 
                                 combination: List[str], strategy_instance: Any, per_page: int = 1) -> List[Dict]:
        """Execute search with API-specific retry configuration and debugging"""
        
        if self.debug:
            print(f"\n--- Executing {api} search ---")
            print(f"Field combination: {combination}")
            print(f"Query params: {query_params}")
        
        # Get API-specific configuration
        config = self._get_api_retry_config(api)
        
        for attempt in range(config.max_retries):
            try:
                # Rate limiting with API-specific delay
                self._wait_for_rate_limit(api, config.rate_limit_delay)
                
                # Build API URL
                if hasattr(strategy_instance, '_build_api_url'):
                    api_url = strategy_instance._build_api_url(query_params, per_page)
                    
                    if self.debug:
                        print(f"API URL (attempt {attempt + 1}): {api_url}")
                else:
                    if self.debug:
                        print(f"ERROR: Strategy has no _build_api_url method")
                    return []
                
                # Make the request with API-specific timeout
                if self.debug:
                    print(f"Making HTTP request with timeout {config.timeout}s...")
                
                response = requests.get(api_url, timeout=config.timeout)
                
                if self.debug:
                    print(f"Response status: {response.status_code}")
                    print(f"Response size: {len(response.content)} bytes")
                
                if response.status_code == 200:
                    results = strategy_instance.parse_response(response)
                    
                    if self.debug:
                        print(f"Parsed {len(results)} results from response")
                        if results and api != "pubmed":  # Don't print XML
                            for i, result in enumerate(results[:3]):  # Show first 3
                                if isinstance(result, dict):
                                    title = str(result.get('title', result.get('name', 'No title')))[:60]
                                    print(f"  Result {i}: {title}...")
                    
                    return results
                else:
                    if self.debug:
                        print(f"HTTP error {response.status_code}: {response.text[:200]}...")
                    
                    # Don't retry on client errors (4xx)
                    if 400 <= response.status_code < 500:
                        if self.debug:
                            print(f"Client error - not retrying")
                        return []
                    
                    # Retry on server errors (5xx)
                    if attempt < config.max_retries - 1:
                        delay = self._calculate_delay(attempt, api, config)
                        if self.debug:
                            print(f"Server error - retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        if self.debug:
                            print(f"Max retries reached")
                        return []
                    
            except requests.exceptions.Timeout as e:
                if self.debug:
                    print(f"Timeout error: {e}")
                if attempt < config.max_retries - 1:
                    delay = self._calculate_delay(attempt, api, config)
                    if self.debug:
                        print(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                else:
                    if self.debug:
                        print(f"Max retries reached after timeout")
                    return []
                    
            except requests.exceptions.ConnectionError as e:
                if self.debug:
                    print(f"Connection error: {e}")
                if attempt < config.max_retries - 1:
                    delay = self._calculate_delay(attempt, api, config)
                    if self.debug:
                        print(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                else:
                    if self.debug:
                        print(f"Max retries reached after connection error")
                    return []
                    
            except Exception as e:
                if self.debug:
                    print(f"Unexpected error: {e}")
                return []
        
        return []
    
    def search_with_target(self, ner_entities: Dict[str, List[str]], api: str, 
                          target_count: Optional[int] = None, 
                          strategy_instance: Any = None) -> List[Dict]:
        """Perform progressive search with dynamic target adjustment"""
        if target_count is None:
            target_count = self.default_target_count
        
        # DYNAMIC TARGET ADJUSTMENT based on available fields
        has_title = bool(ner_entities.get('TITLE', []))
        has_bibliographic = any(ner_entities.get(field, []) for field in ['VOLUME', 'ISSUE', 'PAGE_FIRST', 'PAGE_LAST'])
        has_doi = bool(ner_entities.get('DOI', []))
        
        original_target = target_count
        
        if not has_title and has_bibliographic:
            # No title but we have volume/issue/page data - need more combinations
            target_count = min(target_count * 3, 30)
            adjustment_reason = "no title but bibliographic data available"
        elif not has_title and not has_doi:
            # No title and no DOI - need even more combinations
            target_count = min(target_count * 2, 25)
            adjustment_reason = "no title and no DOI"
        else:
            adjustment_reason = None
        
        if self.debug:
            print(f"\n=== Progressive Search for {api} ===")
            print(f"Original target: {original_target}")
            if adjustment_reason:
                print(f"Adjusted target to {target_count} ({adjustment_reason})")
            else:
                print(f"Using original target: {target_count}")
        
        # Keep track of already performed queries to avoid repeating them.
        seen_query_params = set()
        
        # Get API-specific configurations
        search_capabilities = APICapabilities.get_search_fields(api)
        field_combinations = APICapabilities.get_field_combinations(api)
        
        if not search_capabilities or not field_combinations:
            if self.debug:
                print(f"ERROR: No search capabilities or field combinations for {api}")
            return []
        
        if self.debug:
            print(f"Available field combinations: {len(field_combinations)}")
        
        all_candidates = []
        tried_combinations = []
        
        for combo_index, combination in enumerate(field_combinations):
            if len(all_candidates) >= target_count:
                # SMART STOPPING: Check if we should continue for better precision
                if not has_title and has_bibliographic:
                    # Check if we've tried any bibliographic combinations yet
                    bibliographic_tried = any(
                        any(field in combo for field in ['VOLUME', 'ISSUE', 'PAGE_FIRST']) 
                        for combo in tried_combinations
                    )
                    
                    if not bibliographic_tried and combo_index < len(field_combinations) - 3:
                        if self.debug:
                            print(f"Target reached but no bibliographic combinations tried yet, continuing...")
                    else:
                        if self.debug:
                            print(f"Target count {target_count} reached, stopping search")
                        break
                else:
                    if self.debug:
                        print(f"Target count {target_count} reached, stopping search")
                    break
            
            if self.debug:
                print(f"\n--- Trying combination {combo_index + 1}: {combination} ---")
            
            available_combination = []
            for field in combination:
                if APICapabilities.supports_field(api, field):
                    # Check if field is available in NER entities
                    if field == "TITLE_SEGMENTED":
                        if "TITLE" in ner_entities and ner_entities["TITLE"] and ner_entities["TITLE"][0]:
                            available_combination.append(field)
                            if self.debug:
                                print(f"  ✓ {field} (using TITLE data)")
                    elif field in ner_entities and ner_entities[field] and ner_entities[field][0]:
                        available_combination.append(field)
                        if self.debug:
                            print(f"  ✓ {field}: {ner_entities[field][0]}")
                    else:
                        if self.debug:
                            print(f"  ✗ {field}: not available in NER")
                else:
                    if self.debug:
                        print(f"  ✗ {field}: not supported by {api}")
            
            if not available_combination:
                if self.debug:
                    print(f"  No available fields for this combination, skipping")
                continue

            # Skip overly broad single-field searches (unless it's a high-value field)
            if len(available_combination) == 1 and available_combination[0] not in ['DOI', 'TITLE', 'TITLE_SEGMENTED']:
                if self.debug:
                    print(f"  Skipping overly broad single-field search: {available_combination}")
                continue

            tried_combinations.append(available_combination)
            
            if self.debug:
                print(f"  Using fields: {available_combination}")
            
            query_params = self.field_mapper.build_query_params(
                api, available_combination, ner_entities, search_capabilities
            )
            
            if not query_params:
                if self.debug:
                    print(f"  No query parameters generated, skipping")
                continue

            # Create a hashable representation of query_params for deduplication
            params_key = tuple(sorted(query_params.items()))
            if params_key in seen_query_params:
                if self.debug:
                    print(f"  Skipping duplicate query parameters")
                continue
            seen_query_params.add(params_key)

            # Determine if this is a broad search
            is_broad_search = (
                len(available_combination) <= 2 and 
                not any(field in available_combination for field in ['TITLE', 'DOI'])
            )

            per_page = PER_PAGE_BROAD if is_broad_search else PER_PAGE_SPECIFIC

            # Execute search with retry configuration
            candidates = self._execute_search_with_retry(
                query_params, api, available_combination, strategy_instance, per_page
            )
            
            if candidates:
                all_candidates.extend(candidates)
                if self.debug:
                    print(f"  Found {len(candidates)} candidates (total: {len(all_candidates)})")
                
                # EARLY SUCCESS: Stop if we find a precise result with many fields
                if (len(candidates) == 1 and len(available_combination) >= 4 and 
                    any(field in available_combination for field in ['TITLE', 'DOI', 'VOLUME', 'ISSUE', 'PAGE_FIRST'])):
                    if self.debug:
                        print(f"  Precise result found with {len(available_combination)} fields, stopping search")
                    break
            else:
                if self.debug:
                    print(f"  No candidates found")
        
        # Deduplicate all candidates
        unique_candidates = self.deduplicator.deduplicate_candidates(all_candidates, api)
        
        if self.debug:
            print(f"\nDeduplication: {len(all_candidates)} → {len(unique_candidates)} unique")
            print(f"Bibliographic combinations tried: {len([c for c in tried_combinations if any(f in c for f in ['VOLUME', 'ISSUE', 'PAGE_FIRST'])])}")
        
        final_results = unique_candidates[:original_target]  # Return original target amount
        
        if self.debug:
            print(f"Final results: {len(final_results)} candidates")
        
        return final_results

class SearchOrchestrator:
    """Orchestrates the complete search process across APIs with enhanced DOI support and clean output"""
    
    def __init__(self, target_count: int = 10, debug: bool = False):
        self.field_mapper = FieldMapper(debug=debug)
        self.deduplicator = ResultDeduplicator(self.field_mapper)
        self.progressive_search = ProgressiveSearchStrategy(
            self.field_mapper, self.deduplicator, target_count
        )
        self.debug = debug
    
    def search_single_api(self, ner_entities: Dict[str, List[str]], api: str, 
                         target_count: Optional[int] = 1, 
                         strategy_instance: Any = None) -> List[Dict]:
        """Search a single API with progressive strategy and debugging"""
        self.progressive_search.debug = self.debug
        return self.progressive_search.search_with_target(
            ner_entities, api, target_count, strategy_instance
        )
    
    def search_multiple_apis(self, ner_entities: Dict[str, List[str]], 
                           apis: List[str], target_count_per_api: int = 5,
                           strategy_instances: Dict[str, Any] = None) -> Dict[str, List[Dict]]:
        """Search multiple APIs with API-specific configurations"""
        results = {}
        
        for api in apis:
            if api not in APICapabilities.get_supported_apis():
                continue
            
            strategy_instance = strategy_instances.get(api) if strategy_instances else None
            
            candidates = self.search_single_api(
                ner_entities, api, target_count_per_api, strategy_instance
            )
            
            if candidates:
                results[api] = candidates
        
        return results
