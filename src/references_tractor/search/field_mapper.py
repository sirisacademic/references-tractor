# search/field_mapper.py
"""
Handles API-specific field mapping and preprocessing.
Enhanced with multiple DOI extraction support.
"""

import re
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from googlesearch import search
import requests

class DOIResult(NamedTuple):
    """Structure for DOI extraction results"""
    main_doi: Optional[str]
    alternative_dois: List[str]
    total_count: int

class FieldMapper:
    """Handles field preprocessing and API-specific transformations"""
    
    def __init__(self, debug: bool = False):
        self._journal_id_cache = {}  # Cache for OpenAlex journal ID lookups
        self.debug = debug
           
    def clean_doi(self, doi: str) -> Optional[str]:
        """Clean and normalize a single DOI"""
        if not doi:
            return None
        
        # Handle case where DOI might be a list
        if isinstance(doi, list):
            doi = doi[0] if doi else ""
        
        if not isinstance(doi, str):
            return None
        
        # Remove common prefixes and whitespace
        cleaned = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "").strip()
        
        # Validate DOI format
        if re.match(r'^10\.\d{4,}/[^\s]+$', cleaned):
            return cleaned
        
        return None
    
    def extract_dois_from_result(self, api_result: Dict, api: str) -> DOIResult:
        """
        Extract main DOI and alternative DOIs from a single API result
        Returns: DOIResult(main_doi, alternative_dois, total_count)
        """
        main_doi = None
        alternative_dois = []
        
        try:
            if api == "openalex":
                # OpenAlex typically has one DOI per result
                doi = api_result.get("doi")
                if doi:
                    main_doi = self.clean_doi(doi)
                        
            elif api == "openaire":
                # OpenAIRE can have multiple DOIs in pids array
                if 'results' in api_result and api_result['results']:
                    result = api_result['results'][0]
                else:
                    result = api_result
                
                # Extract DOIs from main pids array - NULL SAFE
                doi_values = []
                pids = result.get('pids') or []  # Handle both missing and null
                if isinstance(pids, list):  # Safety check
                    for pid in pids:
                        if isinstance(pid, dict) and pid.get('scheme') == 'doi':
                            cleaned = self.clean_doi(pid.get('value', ''))
                            if cleaned and cleaned not in doi_values:
                                doi_values.append(cleaned)
                
                # Also check instances for additional DOIs - NULL SAFE
                instances = result.get('instances') or []  # Handle both missing and null
                if isinstance(instances, list):  # Safety check
                    for instance in instances:
                        if not isinstance(instance, dict):
                            continue
                        instance_pids = instance.get('pids') or []  # Handle null
                        if isinstance(instance_pids, list):  # Safety check
                            for pid in instance_pids:
                                if isinstance(pid, dict) and pid.get('scheme') == 'doi':
                                    cleaned = self.clean_doi(pid.get('value', ''))
                                    if cleaned and cleaned not in doi_values:
                                        doi_values.append(cleaned)
                
                # Set main DOI as first one, rest as alternatives
                if doi_values:
                    main_doi = doi_values[0]
                    alternative_dois = doi_values[1:] if len(doi_values) > 1 else []
                            
            elif api == "crossref":
                # CrossRef has one main DOI per result
                doi = api_result.get("DOI")
                if doi:
                    main_doi = self.clean_doi(doi)
                        
            elif api == "pubmed":
                # PubMed now uses parsed dict structure - NULL SAFE
                if isinstance(api_result, dict):
                    # Extract from parsed structure
                    doi = api_result.get('doi')
                    if doi:
                        main_doi = self.clean_doi(doi)
                    else:
                        # Fallback to XML parsing if still has xml_content
                        xml_content = api_result.get('xml_content')
                        if xml_content:
                            try:
                                import xml.etree.ElementTree as ET
                                root = ET.fromstring(xml_content)
                                doi_elem = root.find(".//ELocationID[@EIdType='doi']")
                                if doi_elem is not None:
                                    main_doi = self.clean_doi(doi_elem.text)
                            except Exception:
                                pass
                
            elif api == "hal":
                # HAL typically has one DOI per result
                doi = api_result.get("doiId_s")
                if doi:
                    main_doi = self.clean_doi(doi)
                        
        except Exception as e:
            if self.debug:
                print(f"Error extracting DOIs for {api}: {e}")
        
        # Calculate total count
        total_count = len([d for d in [main_doi] + alternative_dois if d])
        
        return DOIResult(main_doi, alternative_dois, total_count)
    
    def check_doi_match_with_alternatives(self, doi_result: DOIResult, expected_doi: str) -> str:
        """
        Check if main DOI or any alternative DOI matches the expected DOI
        Returns: EXACT, INCORRECT, or N/A
        """
        if not expected_doi:
            return "N/A"
        
        normalized_expected = self.clean_doi(expected_doi)
        if not normalized_expected:
            return "N/A"
        
        # Check main DOI first
        if doi_result.main_doi and self.clean_doi(doi_result.main_doi) == normalized_expected:
            return "EXACT"
        
        # Check alternative DOIs
        for alt_doi in doi_result.alternative_dois:
            if self.clean_doi(alt_doi) == normalized_expected:
                return "EXACT"
        
        # If we have any DOIs but none match
        if doi_result.main_doi or doi_result.alternative_dois:
            return "INCORRECT"
        
        return "N/A"
    
    def get_all_dois_from_result(self, doi_result: DOIResult) -> List[str]:
        """
        Get all DOIs (main + alternatives) as a flat list
        Useful for deduplication and ensemble logic
        """
        all_dois = []
        if doi_result.main_doi:
            all_dois.append(doi_result.main_doi)
        all_dois.extend(doi_result.alternative_dois)
        return all_dois
        
    def clean_title(self, title: str) -> Optional[str]:
        """Clean title for search"""
        if not title:
            return None
        
        # Remove quotes and normalize whitespace
        cleaned = re.sub(r'[\'\"]+', '', title)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove trailing punctuation for search
        cleaned = re.sub(r'[.,;:\-\s]+$', '', cleaned)
        
        return cleaned if len(cleaned) >= 3 else None

    def segment_title_for_or_search(self, title: str) -> Optional[str]:
        """Segment title into 2 parts for OR search"""
        if not title:
            return None
        
        # Clean title first
        cleaned = self.clean_title(title)
        if not cleaned or len(cleaned.split()) < 4:  # Too short to segment
            return cleaned
        
        words = cleaned.split()
        mid_point = len(words) // 2
        
        # Split into two halves
        first_half = " ".join(words[:mid_point]).strip()
        second_half = " ".join(words[mid_point:]).strip()
        
        # Return as OR query
        return f"{first_half}|{second_half}"
   
    def extract_author_surname_boolean(self, authors: str) -> Optional[str]:
        """
        Extract surname(s) from author string for boolean search (OpenAlex/OpenAIRE).
        Returns boolean query format when multiple authors detected.
        Format: ((AUTHOR1 AND AUTHOR2 AND AUTHOR3) OR AUTHOR1 OR AUTHOR2 OR AUTHOR3)
        """
        if not authors:
            return None
        
        # Step 1: Remove all "et al" variants and common noise words
        cleaned = re.sub(r'\b(et\.?al\.?|et\s+al\.?|&\s*al\.?|and\s+others|等|etc\.?)\b', '', authors, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        
        # Step 2: Remove content in parentheses (affiliations, etc.)
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        cleaned = cleaned.strip()
        
        # Step 3: Remove trailing/leading punctuation and whitespace
        cleaned = re.sub(r'^[,;\s\-\.]+|[,;\s\-\.]+$', '', cleaned)
        
        if not cleaned:
            return None
        
        # Step 4: Check for multiple authors separated by " ; "
        if ' ; ' in cleaned:
            return self._extract_multi_author_surnames_boolean(cleaned)
        
        # Single author - use existing logic
        return self._extract_single_author_surname(cleaned)

    def _extract_multi_author_surnames_boolean(self, authors_string: str) -> Optional[str]:
        """
        Extract surnames from multiple authors and build boolean query.
        Format: ((AUTHOR1 AND AUTHOR2 AND AUTHOR3) OR AUTHOR1 OR AUTHOR2 OR AUTHOR3)
        """
        # Split by " ; " to get individual authors
        author_parts = authors_string.split(' ; ')
        
        # Extract surname from each author (limit to 3)
        surnames = []
        for author_part in author_parts[:3]:  # Max 3 authors
            author_part = author_part.strip()
            if not author_part:
                continue
                
            # Use existing single author extraction logic for each part
            surname = self._extract_single_author_surname(author_part)
            if surname and len(surname) > 1:  # Valid surname
                surnames.append(surname)
        
        if not surnames:
            return None
        
        # If only one valid surname found, return as single author
        if len(surnames) == 1:
            return surnames[0]
        
        # Build boolean query: ((A AND B AND C) OR A OR B OR C)
        and_clause = ' AND '.join(surnames)
        or_clause = ' OR '.join(surnames)
        boolean_query = f"(({and_clause}) OR {or_clause})"
        
        return boolean_query

    def extract_author_surname(self, authors: str) -> Optional[str]:
        """
        Extract surname from author string for single-author search (PubMed/CrossRef/HAL).
        Enhanced to handle various formats.
        """
        if not authors:
            return None
        
        # Step 1: Remove all "et al" variants and common noise words
        cleaned = re.sub(r'\b(et\.?al\.?|et\s+al\.?|&\s*al\.?|and\s+others|等|etc\.?)\b', '', authors, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        
        # Step 2: Remove content in parentheses (affiliations, etc.)
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        cleaned = cleaned.strip()
        
        # Step 3: Remove trailing/leading punctuation and whitespace
        cleaned = re.sub(r'^[,;\s\-\.]+|[,;\s\-\.]+$', '', cleaned)
        
        if not cleaned:
            return None
        
        # For single-author APIs, just extract the first author (existing behavior)
        return self._extract_single_author_surname(cleaned)

    def _extract_single_author_surname(self, author: str) -> Optional[str]:
        """
        Extract surname from single author string - existing logic extracted to shared method.
        """
        if not author:
            return None
        
        # Handle comma-separated format "LastName, FirstName" or "LastName, Initials"
        if "," in author:
            parts = author.split(",")
            first_part = parts[0].strip()
            
            # Remove initials from the first part if it contains them
            first_part_clean = re.sub(r'\b[A-Z]\.?\s*$', '', first_part).strip()
            
            # Also remove leading initials pattern from the first part
            if "." in first_part:
                first_part_no_initials = re.sub(r'^([A-Z]\.)+', '', first_part)
                if first_part_no_initials and len(first_part_no_initials) > 2:
                    surname = first_part_no_initials
                else:
                    surname = first_part_clean if first_part_clean else first_part
            else:
                surname = first_part_clean if first_part_clean else first_part
                
            if len(surname) > 1:
                return surname
        
        # Handle semicolon-separated multiple authors - take the first one
        if ";" in author:
            first_author = author.split(";")[0].strip()
            # Recursively process the first author
            return self._extract_single_author_surname(first_author)
        
        # Handle space-separated names
        parts = author.split()
        
        if len(parts) == 0:
            return None
        
        elif len(parts) == 1:
            # Single word - could be surname or needs cleaning
            word = parts[0]
            
            # Remove initials pattern from single words
            if "." in word and len(word) > 3:
                cleaned_word = re.sub(r'^([A-Z]\.)+', '', word)
                if cleaned_word and len(cleaned_word) > 2:
                    return cleaned_word
            
            # Accept single words if they look like surnames
            if len(word) > 2 and word.replace(".", "").replace("-", "").isalpha():
                return word
            else:
                return None
        
        elif len(parts) == 2:
            # Two parts - could be "FirstName LastName" or "Initial LastName" or "LastName Initial"
            first, second = parts
            
            # If first part looks like initials, use second
            first_is_initial = len(first.replace(".", "")) <= 3 and ("." in first or first.isupper())
            if first_is_initial and len(second) > 1:
                return second
            
            # If second part looks like initials, use first  
            second_is_initial = len(second.replace(".", "")) <= 3 and ("." in second or second.isupper())
            if second_is_initial and len(first) > 1:
                return first
            
            # Default: use first part (common in citations)
            if len(first) > 1:
                return first
            elif len(second) > 1:
                return second
        
        else:
            # Multiple parts - take the last substantial word
            substantial_parts = [p for p in parts if len(p.replace(".", "")) > 1]
            
            if substantial_parts:
                return substantial_parts[-1]
            else:
                # Fallback to last part even if short
                surname = parts[-1]
                if len(surname) > 1:
                    return surname
        
        return None
    
    def year_to_date_range(self, year: str) -> Tuple[str, str]:
        """Convert year to date range for OpenAIRE"""
        if not year:
            return None, None
        
        try:
            year_int = int(year)
            if 1800 <= year_int <= 2050:
                return f"{year}-01-01", f"{year}-12-31"
        except ValueError:
            pass
        
        return None, None
    
    def year_to_filter(self, year: str) -> Optional[str]:
        """Convert year to filter format for CrossRef"""
        if not year:
            return None
        
        try:
            year_int = int(year)
            if 1800 <= year_int <= 2050:
                return f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31"
        except ValueError:
            pass
        
        return None
    
    def resolve_journal_id(self, journal_name: str) -> Optional[str]:
        """Resolve journal name to OpenAlex ID with NLM catalog integration"""
        # TODO: Consider also:
        # https://woodward.library.ubc.ca/woodward/research-help/journal-abbreviations/ for science and engineering
        # https://cassi.cas.org for general abbreviations (in part. chemical, biochemical, etc)
        
        if not journal_name:
            return None
        
        # Check cache first
        if journal_name in self._journal_id_cache:
            if self.debug:
                cached_result = self._journal_id_cache[journal_name]
                print(f"\n JOURNAL CACHE HIT: '{journal_name}' → {cached_result}")
            return self._journal_id_cache[journal_name]
        
        if self.debug:
            print(f"\n RESOLVING JOURNAL: '{journal_name}'")
            print(f"   Cache miss - performing lookup...")
        
        # Step 1: Try direct OpenAlex search first
        if self.debug:
            print(f"\n   STEP 1: Direct OpenAlex search")
        journal_id = self._search_openalex_direct(journal_name)
        if journal_id:
            if self.debug:
                print(f"    SUCCESS: Found journal ID '{journal_id}' via direct OpenAlex search")
            self._journal_id_cache[journal_name] = journal_id
            return journal_id
        elif self.debug:
            print(f"    FAILED: No matches in direct OpenAlex search")
        
        # Step 2: Try NLM catalog lookup
        if self.debug:
            print(f"\n   STEP 2: NLM catalog lookup")
        full_title = self._get_full_title_from_nlm(journal_name)
        if full_title and full_title.lower() != journal_name.lower():
            if self.debug:
                print(f"    NLM expansion: '{journal_name}' → '{full_title}'")
            full_title = full_title.lower()
            # Sometimes in OpenAlex "The" at the beginning is omitted.
            search_title = f"{full_title}|{full_title.replace('the ', '', 1)}" if full_title.startswith('the ') else full_title
            journal_id = self._search_openalex_direct(search_title)
            if journal_id:
                if self.debug:
                    print(f"    SUCCESS: Found journal ID '{journal_id}' via NLM-expanded title")
                self._journal_id_cache[journal_name] = journal_id
                return journal_id
            elif self.debug:
                print(f"    FAILED: NLM-expanded title not found in OpenAlex")
        elif self.debug:
            if not full_title:
                print(f"    FAILED: No NLM expansion found")
            else:
                print(f"     SKIPPED: NLM title same as input")
        
        # Step 3: Try Google search as last resort (with better validation)
        if self.debug:
            print(f"\n   STEP 3: Google search fallback")
        try:
            results = search(f"{journal_name} journal", num_results=2, advanced=True)
            
            if self.debug:
                print(f"    Google search query: '{journal_name} journal'")
                print(f"    Google returned {len(list(results))} results")
            
            # Reset results iterator
            results = search(f"{journal_name} journal", num_results=2, advanced=True)
            
            for i, result in enumerate(results):
                if not result or not result.title:
                    if self.debug:
                        print(f"    Result {i+1}: Empty or no title, skipping")
                    continue
                    
                # Better title extraction
                title = result.title.lower().strip()
                if not title or len(title) < 3:
                    if self.debug:
                        print(f"    Result {i+1}: Title too short ('{title}'), skipping")
                    continue
                    
                if self.debug:
                    print(f"    Result {i+1}: Raw title = '{title}'")
                    
                # Remove common prefixes and clean up
                cleaned_title = re.sub(r'^(journal|proceedings|international|transactions)\s+', '', title, flags=re.IGNORECASE)
                cleaned_title = re.split(r'[-:]', cleaned_title)[0].strip()
                cleaned_title = re.sub(r'\s*\([^)]*\)\s*', '', cleaned_title)  # Remove parentheses
                
                if self.debug:
                    print(f"    Result {i+1}: Cleaned title = '{cleaned_title}'")
                
                if cleaned_title and len(cleaned_title) > 3:
                    search_title = f"{cleaned_title}|{cleaned_title.replace('the ', '', 1)}" if cleaned_title.startswith('the ') else cleaned_title
                    journal_id = self._search_openalex_direct(search_title)
                    if journal_id:
                        if self.debug:
                            print(f"    SUCCESS: Found journal ID '{journal_id}' via Google result")
                        self._journal_id_cache[journal_name] = journal_id
                        return journal_id
                    elif self.debug:
                        print(f"    Result {i+1}: OpenAlex search failed for cleaned title")
                elif self.debug:
                    print(f"    Result {i+1}: Cleaned title too short or empty")
                        
        except Exception as e:
            if self.debug:
                print(f"    FAILED: Google search error - {e}")
        
        # Cache negative result to avoid repeated lookups
        if self.debug:
            print(f"\n    FINAL RESULT: No journal ID found for '{journal_name}'")
            print(f"    Caching negative result to avoid future lookups")
        self._journal_id_cache[journal_name] = None
        return None

    def _search_openalex_direct(self, journal_name: str) -> Optional[str]:
        """Direct search in OpenAlex sources"""
        if not journal_name or len(journal_name.strip()) < 3:
            if self.debug:
                print(f"       OpenAlex search: Journal name too short or empty")
            return None
            
        source_url = "https://api.openalex.org/sources"
        
        try:
            # URL encode the journal name properly
            encoded_name = journal_name.replace(' ', '%20')
            url = f"{source_url}?filter=display_name.search:{encoded_name}"
            if self.debug:
                print(f"       OpenAlex search URL: {url}")
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                journals = data.get("results", [])
                if self.debug:
                    print(f"       OpenAlex returned {len(journals)} journal matches")
                    
                    # Show top 3 matches for debugging
                    for i, journal in enumerate(journals[:3]):
                        display_name = journal.get('display_name', 'No name')
                        journal_id = journal.get('id', 'No ID').split('/')[-1]
                        print(f"         Match {i+1}: '{display_name}' (ID: {journal_id})")
                
                if journals:
                    # Return the first match
                    journal_id = journals[0]["id"].split("/")[-1]
                    best_match = journals[0].get('display_name', 'Unknown')
                    if self.debug:
                        print(f"       Selected best match: '{best_match}' → ID: {journal_id}")
                    return journal_id
                elif self.debug:
                    print(f"       No journals found in OpenAlex response")
                    
            else:
                if self.debug:
                    print(f"       OpenAlex API error: {response.status_code}")
                    print(f"       Response: {response.text[:200]}...")
                        
        except Exception as e:
            if self.debug:
                print(f"       OpenAlex search exception: {e}")
        
        return None

    def _get_full_title_from_nlm(self, journal_abbrev: str) -> Optional[str]:
        """Get full journal title from NLM catalog with improved matching"""
        if not journal_abbrev:
            return None
        
        if self.debug:
            print(f"       NLM lookup for: '{journal_abbrev}'")
        
        # URL encode the journal abbreviation
        encoded_abbrev = journal_abbrev.replace(' ', '%20').replace('.', '%2E')
        nlm_url = f"https://www.ncbi.nlm.nih.gov/nlmcatalog/?term={encoded_abbrev}&report=docsum&format=text"
        
        if self.debug:
            print(f"       NLM URL: {nlm_url}")
        
        try:
            response = requests.get(nlm_url, timeout=15)
            
            if response.status_code == 200:
                response_text = response.text
                if self.debug:
                    print(f"       NLM response length: {len(response_text)} chars")
                    # Show first 200 chars of response for debugging
                    print(f"       NLM response preview: {response_text[:200]}...")
                
                best_match = self._find_best_nlm_match(response_text, journal_abbrev)
                
                if best_match:
                    if self.debug:
                        print(f"       NLM best match: '{best_match}'")
                    return best_match
                elif self.debug:
                    print(f"       No suitable matches found in NLM response")
            else:
                if self.debug:
                    print(f"       NLM API error: {response.status_code}")
                        
        except Exception as e:
            if self.debug:
                print(f"       NLM lookup error: {e}")
        
        return None

    def _find_best_nlm_match(self, response_text: str, search_term: str) -> Optional[str]:
        """Find the best matching journal from NLM results with 0.6 threshold"""
        if not response_text or not search_term:
            return None
        
        if self.debug:
            print(f"       Finding best NLM match for: '{search_term}'")
        
        # Parse all journal entries from the response
        journal_entries = self._parse_nlm_entries(response_text)
        
        if not journal_entries:
            if self.debug:
                print(f"       No journal entries parsed from NLM response")
            return None
        
        if self.debug:
            print(f"       Parsed {len(journal_entries)} journal entries")
            # Show parsed entries
            for i, entry in enumerate(journal_entries[:3]):  # Show first 3
                print(f"         Entry {i+1}: Title='{entry['title']}', Abbrev='{entry['abbreviation']}'")
                       
        # Normalize search term for comparison
        normalized_search = self._normalize_journal_name(search_term)
        if self.debug:
            print(f"       Normalized search term: '{normalized_search}'")
        
        best_match = None
        best_score = 0
        
        for i, entry in enumerate(journal_entries):
            # Check against title abbreviation first (most likely match)
            if entry['abbreviation']:
                abbrev_score = self._calculate_journal_similarity(normalized_search, entry['abbreviation'])
                if self.debug:
                    print(f"       Entry {i+1} abbrev similarity: {abbrev_score:.3f} ('{entry['abbreviation']}')")
                if abbrev_score > best_score:
                    best_score = abbrev_score
                    best_match = entry['title']
                    if self.debug:
                        print(f"         New best match via abbreviation!")
            
            # Check against full title as backup
            if entry['title']:
                title_score = self._calculate_journal_similarity(normalized_search, entry['title']) * 0.8  # Lower weight
                if self.debug:
                    print(f"       Entry {i+1} title similarity: {title_score:.3f} ('{entry['title']}')")
                if title_score > best_score:
                    best_score = title_score
                    best_match = entry['title']
                    if self.debug:
                        print(f"         New best match via title!")
        
        if self.debug:
            print(f"       Best overall score: {best_score:.3f}")
            print(f"       Threshold: 0.6")  # Using 0.6 as requested (favoring recall)
        
        # Use 0.6 threshold as requested (prioritizing recall over precision)
        if best_score > 0.6:
            if self.debug:
                print(f"       Match accepted: '{best_match}'")
            return best_match
        else:
            if self.debug:
                print(f"       No match above threshold")
            return None

    def _parse_nlm_entries(self, response_text: str) -> List[Dict[str, str]]:
        """Parse all journal entries from NLM docsum response"""
        entries = []
        
        # Split by numbered entries
        entry_pattern = r'(\d+\.\s+.+?)(?=\n\d+\.\s+|\n\n|\Z)'
        matches = re.findall(entry_pattern, response_text, re.DOTALL)
        
        for match in matches:
            entry = self._parse_single_nlm_entry(match)
            if entry and entry['type'] == 'serial':  # Only journals/serials, not books/videos
                entries.append(entry)
        
        return entries

    def _parse_single_nlm_entry(self, entry_text: str) -> Optional[Dict[str, str]]:
        """Parse a single NLM entry"""
        if not entry_text:
            return None
        
        try:
            # Extract title (first line after number)
            title_match = re.search(r'^\d+\.\s+(.+?)(?:\n|$)', entry_text)
            title = title_match.group(1).strip() if title_match else ""
            
            # Extract abbreviation
            abbrev_match = re.search(r'NLM Title Abbreviation:\s*([^.\n]+)', entry_text)
            abbreviation = abbrev_match.group(1).strip() if abbrev_match else ""
            
            # Determine type (Serial vs other)
            entry_type = "serial" if "[Serial]" in entry_text else "other"
            
            # Skip non-serial entries (books, videos, etc.)
            if entry_type != "serial":
                return None
            
            return {
                'title': title,
                'abbreviation': abbreviation,
                'type': entry_type,
                'raw': entry_text
            }
            
        except Exception as e:
            return None

    def _get_journal_reverse_abbreviation_map(self) -> Dict[str, str]:
        """
        Get reverse mapping: full_word -> abbreviation.
        Multiple full words can map to the same abbreviation.
        
        Returns:
            Dictionary mapping full_word -> abbreviation
        """
        return {
            # Publication types - can have multiple variants
            'journal': 'j',
            'journals': 'j',
            'proceedings': 'proc',
            'proceeding': 'proc',
            'transactions': 'trans',
            'transaction': 'trans',
            'annals': 'ann',
            'bulletin': 'bull',
            'bulletins': 'bull',
            'review': 'rev',
            'reviews': 'rev',
            'quarterly': 'q',
            'communications': 'commun',
            'communication': 'commun',
            'letters': 'lett',
            'letter': 'lett',
            'reports': 'rep',
            'report': 'rep',
            'abstracts': 'abstr',
            'abstract': 'abstr',
            'supplement': 'suppl',
            'supplements': 'suppl',
            'series': 'ser',
            'newsletter': 'newsl',
            'newsletters': 'newsl',
            'digest': 'dig',
            
            # Academic/Institutional terms
            'university': 'univ',
            'universities': 'univ',
            'college': 'coll',
            'colleges': 'coll',
            'institute': 'inst',
            'institutes': 'inst',
            'academy': 'acad',
            'academies': 'acad',
            'association': 'assoc',
            'associations': 'assoc',
            'foundation': 'found',
            'foundations': 'found',
            'laboratory': 'lab',
            'laboratories': 'lab',
            'department': 'dept',
            'departments': 'dept',
            'center': 'cent',
            'centre': 'cent',  # British spelling
            'centers': 'cent',
            'centres': 'cent',
            'society': 'soc',
            'societies': 'soc',
            
            # Geographic/Institutional terms with variants
            'international': 'int',
            'internal': 'int',
            'intensive': 'int',
            'american': 'am',
            'america': 'am',
            'european': 'eur',
            'europe': 'eur',
            'british': 'br',
            'britain': 'br',
            'canadian': 'can',
            'canada': 'can',
            'australian': 'aust',
            'australia': 'aust',
            'national': 'natl',
            'nationwide': 'natl',
            
            # Scientific fields with multiple variants
            'science': 'sci',
            'sciences': 'sci',
            'scientific': 'sci',
            'research': 'res',
            'studies': 'stud',
            'study': 'stud',
            'advances': 'adv',
            'advanced': 'adv',
            'advance': 'adv',
            'biology': 'biol',
            'biological': 'biol',
            'biologic': 'biol',
            'biologically': 'biol',
            'chemistry': 'chem',
            'chemical': 'chem',
            'chemically': 'chem',
            'engineering': 'eng',
            'engineer': 'eng',
            'medicine': 'med',
            'medical': 'med',
            'medicinal': 'med',
            'physics': 'phys',
            'physical': 'phys',
            'analysis': 'anal',
            'analytical': 'anal',
            'analyze': 'anal',
            'technology': 'technol',
            'technological': 'technol',
            'technologies': 'technol',
            'mathematics': 'math',
            'mathematical': 'math',
            'ecology': 'ecol',
            'ecological': 'ecol',
            'computer': 'comp',
            'computational': 'comp',
            'computing': 'comp',
            'comparative': 'comp',
            'comparison': 'comp',
            'environmental': 'environ',
            'environment': 'environ',
            'molecular': 'mol',
            'molecule': 'mol',
            'clinical': 'clin',
            'clinic': 'clin',
            'applied': 'appl',
            'application': 'appl',
            'applications': 'appl',
            'management': 'manag',
            'manager': 'manag',
            'managing': 'manag',
            
            # Previously ambiguous terms - now handled properly
            'systematic': 'syst',
            'systematically': 'syst',
            'systems': 'syst',
            'system': 'syst',
            'general': 'gen',
            'generally': 'gen',
            'genetics': 'gen',
            'genetic': 'gen',
            'genomics': 'gen',
            'genomic': 'gen',
            'genome': 'gen',
            'experimental': 'exp',
            'experiment': 'exp',
            'experiments': 'exp',
            'expert': 'exp',
            'expertise': 'exp',
            'theoretical': 'theor',
            'theory': 'theor',
            'theories': 'theor',
            'modern': 'mod',
            'modeling': 'mod',
            'modelling': 'mod',  # British spelling
            'model': 'mod',
            'models': 'mod',
            
            # Specialized compound fields
            'biochemistry': 'biochem',
            'biochemical': 'biochem',
            'biophysics': 'biophys',
            'biophysical': 'biophys',
            'biotechnology': 'biotechnol',
            'biotechnological': 'biotechnol',
            'neuroscience': 'neurosci',
            'neurosciences': 'neurosci',
            'neuroscientific': 'neurosci',
            'pharmacology': 'pharmacol',
            'pharmacological': 'pharmacol',
            'psychology': 'psychol',
            'psychological': 'psychol',
            'sociology': 'sociol',
            'sociological': 'sociol',
            'anthropology': 'anthropol',
            'anthropological': 'anthropol',
            'geology': 'geol',
            'geological': 'geol',
            'geography': 'geogr',
            'geographical': 'geogr',
            'geographic': 'geogr',
            'statistics': 'stat',
            'statistical': 'stat',
            'economics': 'econ',
            'economic': 'econ',
            'philosophy': 'philos',
            'philosophical': 'philos',
            'history': 'hist',
            'historical': 'hist',
            'literature': 'lit',
            'literary': 'lit',
            
            # Frequency terms
            'monthly': 'mon',
            'weekly': 'wkly',
            'bimonthly': 'bimon',
            'biannual': 'biann',
            'biannually': 'biann',
            'annual': 'ann',
            'annually': 'ann',
            
            # Medical specialties with variants
            'bacteriology': 'bacteriol',
            'bacteriological': 'bacteriol',
            'immunology': 'immunol',
            'immunological': 'immunol',
            'microbiology': 'microbiol',
            'microbiological': 'microbiol',
            'cardiology': 'cardiol',
            'cardiological': 'cardiol',
            'neurology': 'neurol',
            'neurological': 'neurol',
            'pathology': 'pathol',
            'pathological': 'pathol',
            'physiology': 'physiol',
            'physiological': 'physiol',
            'oncology': 'oncol',
            'oncological': 'oncol',
            'gynecology': 'gynecol',
            'gynecological': 'gynecol',
            'gynaecology': 'gynecol',  # British spelling
            'radiology': 'radiol',
            'radiological': 'radiol',
            'endocrinology': 'endocrinol',
            'endocrinological': 'endocrinol',
            'epidemiology': 'epidemiol',
            'epidemiological': 'epidemiol',
            'pharmacokinetics': 'pharmacokinet',
            'biostatistics': 'biostatist',
            'biostatistical': 'biostatist',
        }

    def _handle_special_acronyms(self, word1: str, word2: str) -> float:
        """
        Handle special journal acronyms that don't follow standard abbreviation patterns.
        Returns score if match found, 0.0 otherwise.
        """
        # Special acronym mappings
        special_acronyms = {
            'jama': ['journal', 'american', 'medical', 'association'],
            'pnas': ['proceedings', 'national', 'academy', 'sciences'],
            'nejm': ['new', 'england', 'journal', 'medicine'],
            'bmj': ['british', 'medical', 'journal'],
            'jacs': ['journal', 'american', 'chemical', 'society'],
        }
        
        word1_lower = word1.lower()
        word2_lower = word2.lower()
        
        # Check if word1 is a special acronym and word2 contains its expansion
        if word1_lower in special_acronyms:
            expansion_words = special_acronyms[word1_lower]
            word2_normalized = self._normalize_journal_name(word2_lower)
            
            # Check if all acronym words appear in the full name
            matches = 0
            for exp_word in expansion_words:
                if exp_word in word2_normalized:
                    matches += 1
            
            if matches >= len(expansion_words) * 0.75:  # At least 75% of words match
                return 0.95
        
        # Check reverse direction
        if word2_lower in special_acronyms:
            expansion_words = special_acronyms[word2_lower]
            word1_normalized = self._normalize_journal_name(word1_lower)
            
            matches = 0
            for exp_word in expansion_words:
                if exp_word in word1_normalized:
                    matches += 1
            
            if matches >= len(expansion_words) * 0.75:
                return 0.95
        
        return 0.0

    def _calculate_journal_similarity(self, term1: str, term2: str) -> float:
        """
        Calculate similarity between two journal names with intelligent abbreviation matching.
        Uses hybrid position-aware alignment with hierarchical word scoring.
        """
        if not term1 or not term2:
            return 0.0
        
        # Step 1: Apply general normalization
        norm1 = self._normalize_journal_name(term1)
        norm2 = self._normalize_journal_name(term2)
        
        if not norm1 or not norm2:
            return 0.0
        
        # Exact match after normalization
        if norm1 == norm2:
            return 1.0
        
        # Step 2: Tokenize into words
        words1 = norm1.split()
        words2 = norm2.split()
        
        if not words1 or not words2:
            return 0.0
        
        # Step 3: Position-aware alignment with hierarchical scoring
        alignment_score = self._compute_positional_alignment_score(words1, words2)
        
        # Step 4: Apply length penalty for very different word counts
        length_ratio = min(len(words1), len(words2)) / max(len(words1), len(words2))
        if length_ratio < 0.4:  # Very different lengths
            alignment_score *= 0.7
        elif length_ratio < 0.6:  # Moderately different lengths
            alignment_score *= 0.9
        
        return min(alignment_score, 1.0)

    def _compute_positional_alignment_score(self, words1: List[str], words2: List[str]) -> float:
        """
        Compute alignment score using position-aware hierarchical matching.
        Updated to use reverse abbreviation mapping.
        """
        # Get reverse abbreviation mapping
        reverse_abbrev_map = self._get_journal_reverse_abbreviation_map()
        
        # Try sequential alignment first (common case for journal abbreviations)
        sequential_score = self._score_sequential_alignment(words1, words2, reverse_abbrev_map)
        
        # If sequential alignment is good enough, use it
        if sequential_score >= 0.8:
            return sequential_score
        
        # Otherwise, try optimal alignment (handles word reordering)
        optimal_score = self._score_optimal_alignment(words1, words2, reverse_abbrev_map)
        
        # Return the better score
        return max(sequential_score, optimal_score)

    def _score_sequential_alignment(self, words1: List[str], words2: List[str], 
                                  reverse_abbrev_map: Dict[str, str]) -> float:
        """
        Score by aligning words in sequence (position 0 to 0, 1 to 1, etc.)
        Best for typical journal abbreviations that maintain word order.
        """
        total_score = 0.0
        max_length = max(len(words1), len(words2))
        
        for i in range(max_length):
            word1 = words1[i] if i < len(words1) else None
            word2 = words2[i] if i < len(words2) else None
            
            # Calculate match score for this position
            match_score = self._score_word_pair_hierarchical(word1, word2, reverse_abbrev_map)
            total_score += match_score
        
        # Normalize by number of positions
        return total_score / max_length

    def _score_optimal_alignment(self, words1: List[str], words2: List[str],
                               reverse_abbrev_map: Dict[str, str]) -> float:
        """
        Find optimal word alignment using greedy approach.
        Handles cases where words might be reordered or missing.
        """
        # Create scoring matrix
        match_matrix = []
        for word1 in words1:
            row = []
            for word2 in words2:
                score = self._score_word_pair_hierarchical(word1, word2, reverse_abbrev_map)
                row.append(score)
            match_matrix.append(row)
        
        # Find best alignment using greedy approach
        used_indices_2 = set()
        total_score = 0.0
        
        for i, word1 in enumerate(words1):
            best_match_score = 0.0
            best_j = -1
            
            # Find best unused match in words2
            for j, word2 in enumerate(words2):
                if j not in used_indices_2:
                    score = match_matrix[i][j]
                    if score > best_match_score:
                        best_match_score = score
                        best_j = j
            
            # If we found a decent match, use it
            if best_match_score > 0.3:  # Threshold for considering a match
                used_indices_2.add(best_j)
                total_score += best_match_score
        
        # Normalize by number of words in shorter sequence
        return total_score / min(len(words1), len(words2)) if words1 and words2 else 0.0

    def _score_word_pair_hierarchical(self, word1: Optional[str], word2: Optional[str],
                                    reverse_abbrev_map: Dict[str, str]) -> float:
        """
        Score a pair of words using hierarchical matching approach with reverse mapping.
        
        Hierarchy (highest to lowest priority):
        1. Exact match (1.0)
        2. Special acronym match (0.95)
        3. Both map to same abbreviation (0.95)
        4. One is abbreviation, other is full word (0.90)
        5. Prefix match (0.75-0.90 depending on quality)
        6. Fuzzy similarity (0.4-0.8)
        7. No match (0.0)
        """
        if not word1 or not word2:
            return 0.0
        
        # 1. Exact match
        if word1 == word2:
            return 1.0
        
        # 2. Special acronym handling (highest priority for known cases)
        special_score = self._handle_special_acronyms(word1, word2)
        if special_score > 0:
            return special_score
        
        # 3. Both words map to the same abbreviation
        abbrev1 = reverse_abbrev_map.get(word1)
        abbrev2 = reverse_abbrev_map.get(word2)
        
        if abbrev1 and abbrev2 and abbrev1 == abbrev2:
            return 0.95  # Both map to same abbreviation (e.g., "medical" and "medicine" → "med")
        
        # 4. One word is an abbreviation, the other is the full word it maps to
        if abbrev1 and abbrev1 == word2:
            return 0.90  # word1 is abbreviation of word2
        if abbrev2 and abbrev2 == word1:
            return 0.90  # word2 is abbreviation of word1
        
        # 5. Prefix matching (for unmapped abbreviations)
        prefix_score = self._score_prefix_match(word1, word2)
        if prefix_score > 0:
            return prefix_score
        
        # 6. Fuzzy similarity fallback
        fuzzy_score = self._score_fuzzy_similarity(word1, word2)
        return fuzzy_score

    def _score_prefix_match(self, word1: str, word2: str) -> float:
        """
        Score prefix matching with quality assessment.
        Handles cases like "syst" -> "systematic" that aren't in abbreviation map.
        """
        if len(word1) < 3 or len(word2) < 3:
            return 0.0
        
        shorter = min(word1, word2, key=len)
        longer = max(word1, word2, key=len)
        
        # Check if shorter word is prefix of longer word
        if longer.startswith(shorter):
            # Quality scoring based on prefix length and completeness
            prefix_ratio = len(shorter) / len(longer)
            
            # High-quality prefixes (like "syst" -> "systematic")
            if len(shorter) >= 4 and prefix_ratio >= 0.5:
                return 0.90
            # Medium-quality prefixes (like "int" -> "international")  
            elif len(shorter) >= 3 and prefix_ratio >= 0.25:
                return 0.85
            # Lower-quality but acceptable prefixes
            elif len(shorter) >= 3 and prefix_ratio >= 0.2:
                return 0.75
            # Very short prefixes (discouraged)
            else:
                return 0.60
        
        return 0.0

    def _score_fuzzy_similarity(self, word1: str, word2: str) -> float:
        """
        Score fuzzy string similarity as last resort.
        """
        from difflib import SequenceMatcher
        fuzzy_score = SequenceMatcher(None, word1, word2).ratio()
        
        # Apply threshold and scaling
        if fuzzy_score >= 0.8:
            return fuzzy_score * 0.7  # Scale down since it's fuzzy
        elif fuzzy_score >= 0.6:
            return fuzzy_score * 0.5  # Lower confidence
        else:
            return 0.0  # Too dissimilar

    def _normalize_journal_name(self, name: str) -> str:
        """
        Normalize journal name for comparison - focuses on general text cleanup only.
        Abbreviation handling is done separately in the comparison logic.
        """
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = name.lower().strip()
        
        # Replace dots with spaces (KEEP THIS - it was the fix for J.IMMUNOL)
        normalized = re.sub(r'\.', ' ', normalized)
        
        # Handle other punctuation
        normalized = re.sub(r'[:/]', ' ', normalized)  # Replace : and / with spaces
        normalized = re.sub(r'[,;\"\'`\(\)]', '', normalized)  # Remove other punctuation
        normalized = re.sub(r'[—–]', '-', normalized)  # Replace em/en dashes
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove ONLY basic stop words that don't distinguish publications
        # Do NOT remove journal-specific terms - those are handled in comparison
        basic_stop_words = ['the', 'of', 'and', 'for', 'in', 'on', 'at', 'to', 'from', 'with', 'by']
        words = [w for w in normalized.split() if w not in basic_stop_words]
        
        result = ' '.join(words)
        return result if len(result) >= 2 else ""  # Minimum length check
    
    def preprocess_field(self, field_name: str, value: str, preprocessing_func: Optional[str] = None) -> Any:
        """Apply preprocessing to a field value"""
        if not value or not preprocessing_func:
            return value
        
        # Get the preprocessing function by name
        if hasattr(self, preprocessing_func):
            func = getattr(self, preprocessing_func)
            return func(value)
        
        return value
    
    def build_query_params(self, api: str, field_combination: List[str], ner_entities: Dict[str, List[str]], 
                          search_capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Build API-specific query parameters with comprehensive error handling"""
        query_params = {}
        
        if self.debug:
            print(f"\n BUILDING QUERY PARAMS for {api}")
            print(f"   Field combination: {field_combination}")
            print(f"   Available NER entities: {list(ner_entities.keys())}")
        
        for field in field_combination:
            if self.debug:
                print(f"\n   Processing field: {field}")
            
            try:
                # Get the value from NER entities with better handling
                # When processing TITLE_SEGMENTED, use TITLE data
                if field == "TITLE_SEGMENTED":
                    entity_values = ner_entities.get("TITLE", [])  # Use TITLE data
                    if self.debug:
                        print(f"     Using TITLE data for TITLE_SEGMENTED: {entity_values}")
                else:
                    entity_values = ner_entities.get(field, [])
                    if self.debug:
                        print(f"     Raw entity values: {entity_values}")
                
                if not entity_values:
                    if self.debug:
                        print(f"      No values found for {field}, skipping")
                    continue
                
                # Handle nested lists and get first valid value
                value = None
                for val in entity_values:
                    if isinstance(val, list) and val:
                        # Handle nested lists (e.g., [["value"]])
                        inner_val = val[0] if val else None
                        if isinstance(inner_val, str) and inner_val.strip():
                            value = inner_val
                            if self.debug:
                                print(f"     Extracted from nested list: '{value}'")
                            break
                    elif isinstance(val, str) and val.strip():
                        value = val
                        if self.debug:
                            print(f"     Using string value: '{value}'")
                        break
                
                if not value:
                    if self.debug:
                        print(f"      No valid value extracted for {field}, skipping")
                    continue
                
                # Get search field configuration
                search_config = search_capabilities.get(field)
                if not search_config:
                    if self.debug:
                        print(f"      No search config found for {field}, skipping")
                    continue
                
                if self.debug:
                    print(f"     API field name: {search_config.api_field_name}")
                    print(f"     Preprocessing: {search_config.required_preprocessing}")
                
                # Apply preprocessing if specified
                if search_config.required_preprocessing:
                    if self.debug:
                        print(f"      Applying preprocessing: {search_config.required_preprocessing}")
                        print(f"     Before: '{value}'")
                    
                    try:
                        processed_value = self.preprocess_field(field, value, search_config.required_preprocessing)
                        
                        if self.debug:
                            print(f"     After: '{processed_value}'")
                        
                        # Handle special cases that return tuples (like date ranges)
                        if isinstance(processed_value, tuple):
                            if field == "PUBLICATION_YEAR" and api == "openaire":
                                from_date, to_date = processed_value
                                if from_date and to_date:
                                    query_params["fromPublicationDate"] = from_date
                                    query_params["toPublicationDate"] = to_date
                                    if self.debug:
                                        print(f"      Added date range: {from_date} to {to_date}")
                            elif field == "PUBLICATION_YEAR" and api == "crossref":
                                # CrossRef filter format
                                if processed_value and isinstance(processed_value, str):
                                    query_params["filter"] = processed_value
                                    if self.debug:
                                        print(f"      Added CrossRef filter: {processed_value}")
                            continue
                        
                        if processed_value is None:
                            if self.debug:
                                print(f"      Preprocessing returned None, skipping")
                            continue
                        value = processed_value
                        
                    except Exception as e:
                        if self.debug:
                            print(f"      Preprocessing error: {e}")
                        continue
                
                # Get API-specific parameter name
                api_field_name = search_config.api_field_name
                
                if self.debug:
                    print(f"     Final value: '{value}'")
                    print(f"     Target parameter: '{api_field_name}'")
                
                # Handle special field formatting with error handling
                try:
                    if api == "openalex":
                        # OpenAlex field handling - USE PREPROCESSED VALUES
                        if field == "DOI":
                            # DOI already cleaned by preprocessing
                            query_params["doi"] = value
                            if self.debug:
                                print(f"      Added OpenAlex DOI: {value}")
                        elif field == "JOURNAL":
                            # Journal already resolved to ID by preprocessing
                            if value:  # Only add if we got a valid journal ID
                                query_params["locations.source.id"] = value
                                if self.debug:
                                    print(f"      Added OpenAlex journal ID: {value}")
                            else:
                                if self.debug:
                                    print(f"      Journal resolution failed, skipping")
                        elif field == "TITLE" and "|" in str(value):
                            # Handle OR syntax for segmented titles
                            query_params["title.search"] = value
                            if self.debug:
                                print(f"      Added OpenAlex OR title search: {value}")
                        else:
                            # Use preprocessed value directly
                            query_params[api_field_name] = value
                            if self.debug:
                                print(f"      Added OpenAlex field {api_field_name}: {value}")
                    
                    # ... (similar debug prints for other APIs)
                    elif api == "openaire":
                        # OpenAIRE Graph API special handling
                        if field == "DOI":
                            query_params["pid"] = value
                            if self.debug:
                                print(f"      Added OpenAIRE DOI: {value}")
                        elif field == "TITLE":
                            query_params["mainTitle"] = value
                            if self.debug:
                                print(f"      Added OpenAIRE title: {value}")
                        elif field == "AUTHORS":
                            query_params["authorFullName"] = value
                            if self.debug:
                                print(f"      Added OpenAIRE author: {value}")
                        elif field == "PUBLICATION_YEAR":
                            # Already handled above in tuple processing
                            continue
                        else:
                            query_params[api_field_name] = value
                            if self.debug:
                                print(f"      Added OpenAIRE field {api_field_name}: {value}")
                    
                    else:
                        # Default handling for other APIs
                        query_params[api_field_name] = value
                        if self.debug:
                            print(f"      Added {api} field {api_field_name}: {value}")
                            
                except Exception as e:
                    if self.debug:
                        print(f"      Field formatting error for {field}: {e}")
                    continue
                    
            except Exception as e:
                if self.debug:
                    print(f"      General error processing {field}: {e}")
                continue
        
        if self.debug:
            print(f"\n FINAL QUERY PARAMS: {query_params}")
        
        return query_params

    def extract_response_field(self, data: Any, field_config: Any, api: str) -> Any:
        """Extract a field from API response using the field configuration"""
        if not data or not field_config:
            return None
        
        path = field_config.path
        
        try:
            if api == "pubmed":
                # Handle both dict structure and XML fallback
                if isinstance(data, dict):
                    # Try direct field access first (from parsed structure)
                    if path == "PubmedArticle/MedlineCitation/PMID":
                        return data.get('pmid') or data.get('id')
                    elif path == "PubmedArticle/MedlineCitation/Article/ELocationID[@EIdType='doi']":
                        return data.get('doi')
                    elif path == "PubmedArticle/MedlineCitation/Article/ArticleTitle":
                        return data.get('title')
                    elif path == "PubmedArticle/MedlineCitation/Article/AuthorList/Author":
                        return data.get('authors', [])
                    elif path == "PubmedArticle/MedlineCitation/Article/Journal/JournalIssue/PubDate/Year":
                        return data.get('publication_year')
                    elif path == "PubmedArticle/MedlineCitation/Article/Journal/Title":
                        return data.get('journal')
                    elif path == "PubmedArticle/MedlineCitation/Article/Journal/JournalIssue/Volume":
                        return data.get('volume')
                    elif path == "PubmedArticle/MedlineCitation/Article/Journal/JournalIssue/Issue":
                        return data.get('issue')
                    
                    # Fallback to XML parsing if xml_content exists
                    if 'xml_content' in data:
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(data['xml_content'])
                        return root.findtext(path)
                
                return None
            
            # Handle JSON path extraction for other APIs
            return self._extract_json_path(data, path)
            
        except Exception:
            # Try fallback paths if available
            if field_config.fallback_paths:
                for fallback_path in field_config.fallback_paths:
                    try:
                        return self._extract_json_path(data, fallback_path)
                    except Exception:
                        continue
            return None
    
    def _extract_json_path(self, data: Any, path: str) -> Any:
        """Extract value from nested JSON using dot notation path"""
        if not path:
            return data
        
        parts = path.split('.')
        current = data
        
        for part in parts:
            if current is None:
                return None
            
            # Handle array indexing like "title[0]"
            if '[' in part and ']' in part:
                field_name = part.split('[')[0]
                index_str = part.split('[')[1].split(']')[0]
                
                if isinstance(current, dict) and field_name in current:
                    current = current[field_name]
                    if isinstance(current, list):
                        try:
                            index = int(index_str)
                            current = current[index] if 0 <= index < len(current) else None
                        except (ValueError, IndexError):
                            current = None
                    else:
                        current = None
                else:
                    current = None
            else:
                # Regular field access
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    current = None
        
        return current
