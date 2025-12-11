# search_api.py - Enhanced with multiple DOI support and fixed PubMed handling
import requests
import re
import time
import xml.etree.ElementTree as ET
import urllib
from typing import Dict, List, Any, Optional

# Import enhanced components
from .api_capabilities import APICapabilities
from .field_mapper import FieldMapper, DOIResult
from .progressive_search import SearchOrchestrator, ProgressiveSearchStrategy, ResultDeduplicator

class BaseAPIStrategy:
    """Base class for API search strategies with enhanced DOI support"""
        
    def __init__(self, debug:bool = False):
        self.field_mapper = FieldMapper(debug=debug)
        self.deduplicator = ResultDeduplicator(self.field_mapper)
        self.debug = debug

    def encode_query_value(self, value: str, encoding_type: str = "quote") -> str:
        """Encode a query value for URL safety"""
        if not value:
            return ""
        
        if encoding_type == "quote_plus":
            return urllib.parse.quote_plus(str(value))
        else:
            return urllib.parse.quote(str(value))
    
    def encode_query_params(self, params: Dict[str, Any], encoding_type: str = "quote") -> Dict[str, str]:
        """Encode all query parameters"""
        encoded_params = {}
        for key, value in params.items():
            if value is not None and str(value).strip():
                encoded_params[key] = self.encode_query_value(value, encoding_type)
        return encoded_params

    def parse_response(self, response: requests.Response) -> List[Dict]:
        """Parse API response into standardized dict format"""
        raise NotImplementedError("Subclasses must implement parse_response")

    def enhance_result_with_dois(self, result: Dict[str, Any], api_name: str) -> Dict[str, Any]:
        """
        Enhanced result with multiple DOI information - Fixed for PubMed XML strings
        """
        # Handle PubMed XML string results
        if api_name == "pubmed" and isinstance(result, str):
            # Create dict wrapper for XML string to avoid assignment errors
            xml_content = result
            result = {
                'xml_content': xml_content,
                'api_source': 'pubmed',
                'is_xml': True
            }
        
        enhanced_result = result.copy() if isinstance(result, dict) else {'xml_content': result}
        
        try:
            # Extract DOI information using our enhanced field mapper
            doi_result = self.field_mapper.extract_dois_from_result(result, api_name)
            
            # Add DOI information to the result
            enhanced_result['main_doi'] = doi_result.main_doi
            enhanced_result['alternative_dois'] = doi_result.alternative_dois
            enhanced_result['total_dois'] = doi_result.total_count
            enhanced_result['all_dois'] = self.field_mapper.get_all_dois_from_result(doi_result)
            
            # Maintain backward compatibility - set 'doi' field to main DOI
            if doi_result.main_doi:
                enhanced_result['doi'] = doi_result.main_doi
            
            # Add API source information
            enhanced_result['api_source'] = api_name
            enhanced_result['supports_multiple_dois'] = APICapabilities.supports_multiple_dois(api_name)
            
        except Exception as e:
            # Fallback: try to get DOI from existing field
            existing_doi = None
            if isinstance(result, dict):
                existing_doi = result.get('doi') or result.get('DOI')
            
            if existing_doi:
                cleaned_doi = self.field_mapper.clean_doi(existing_doi)
                enhanced_result['main_doi'] = cleaned_doi
                enhanced_result['alternative_dois'] = []
                enhanced_result['total_dois'] = 1 if cleaned_doi else 0
                enhanced_result['all_dois'] = [cleaned_doi] if cleaned_doi else []
                enhanced_result['doi'] = cleaned_doi  # Backward compatibility
            else:
                enhanced_result['main_doi'] = None
                enhanced_result['alternative_dois'] = []
                enhanced_result['total_dois'] = 0
                enhanced_result['all_dois'] = []
                enhanced_result['doi'] = None
            
            enhanced_result['api_source'] = api_name
            enhanced_result['supports_multiple_dois'] = False
        
        return enhanced_result
           
    def search(self, ner_entities: Dict[str, List[str]], target_count: int = 1, **kwargs) -> List[Dict]:
        """Main search method - now just processes results, orchestration handled by SearchAPI"""
        api_name = self.get_api_name()
        
        if self.debug:
            print(f"\n--- DEBUG: {api_name} Strategy Search ---")
            print(f"NER entities: {ner_entities}")
        
        # This will be called by SearchOrchestrator, so we don't create another one here
        # Just indicate that this method should not be called directly
        raise NotImplementedError("BaseAPIStrategy.search() should not be called directly. Use SearchAPI.search_api() instead.")
    
    def get_api_name(self) -> str:
        """Return the API name for this strategy"""
        raise NotImplementedError("Subclasses must implement get_api_name")
    
    def _build_api_url(self, query_params: Dict[str, Any]) -> str:
        """Build API-specific URL from query parameters"""
        raise NotImplementedError("Subclasses must implement _build_api_url")

class OpenAlexStrategy(BaseAPIStrategy):
    """OpenAlex API search strategy with enhanced DOI support"""
    
    def get_api_name(self) -> str:
        return "openalex"

    def parse_response(self, response: requests.Response) -> List[Dict]:
        """Parse OpenAlex API response"""
        try:
            data = response.json()
            if data is None:
                return []
            
            results = data.get("results", [])
            return results if isinstance(results, list) else []
            
        except (ValueError, requests.exceptions.JSONDecodeError) as e:
            if self.debug:
                print(f"OpenAlex JSON parse error: {e}")
            return []
        except Exception as e:
            if self.debug:
                print(f"OpenAlex parse error: {e}")
            return []

    def _build_api_url(self, query_params: Dict[str, Any], per_page: int = 1) -> str:
        """Build OpenAlex API URL from query parameters"""
        base_url = "https://api.openalex.org/works"
        
        # Convert query parameters to OpenAlex filter format
        filter_parts = []
        for key, value in query_params.items():
            if not value:
                continue
            
            # Clean and encode value for OpenAlex compatibility
            if key == "title.search" and "|" in str(value):
                # Handle OR syntax for segmented titles - don't clean as aggressively
                segments = str(value).split("|")
                cleaned_segments = []
                for segment in segments:
                    cleaned_segment = self._clean_openalex_filter_value(segment)
                    if cleaned_segment:
                        cleaned_segments.append(cleaned_segment)
                
                if cleaned_segments:
                    # Encode the OR query properly
                    or_query = "|".join(cleaned_segments)
                    encoded_value = self.encode_query_value(or_query)
                    filter_parts.append(f"{key}:{encoded_value}")
            
            elif key in ["biblio.volume", "biblio.issue", "biblio.first_page", "biblio.last_page", "raw_author_name.search"]:
                # Don't clean bibliographic fields (short numbers) or author surnames (can be short)
                value_str = str(value).strip()
                if value_str:  # Just check it's not empty
                    encoded_value = self.encode_query_value(value_str)
                    filter_parts.append(f"{key}:{encoded_value}")
            
            else:
                # Regular cleaning for non-OR, non-bibliographic queries
                cleaned_value = self._clean_openalex_filter_value(str(value))
                if not cleaned_value:
                    continue
                encoded_value = self.encode_query_value(cleaned_value)
                
                if key == "locations.source.id":
                    filter_parts.append(f"{key}:{encoded_value}")
                elif key in ["raw_author_name.search"]:
                    filter_parts.append(f"{key}:{encoded_value}")
                elif key in ["title.search"]:
                    filter_parts.append(f'{key}:"{encoded_value}"')
                elif key in ["publication_year", "doi"]:
                    filter_parts.append(f"{key}:{encoded_value}")
        
        if filter_parts:
            filter_string = ",".join(filter_parts)
            final_url = f"{base_url}?per-page={per_page}&filter={filter_string}&mailto=info@sirisacademic.com"
            if self.debug:
                print(f"OpenAlex URL: {final_url}")
            return final_url
        
        return f"{base_url}?per-page={per_page}"

    def _clean_openalex_filter_value(self, value: str) -> str:
        """Clean filter values for OpenAlex compatibility"""
        if not value:
            return ""
        
        import re
        
        # Remove parentheses and their contents
        cleaned = re.sub(r'\([^)]*\)', '', value)
        
        # Replace problematic punctuation with spaces instead of removing
        cleaned = re.sub(r'[:/]', ' ', cleaned)  # Replace : and / with spaces
        cleaned = re.sub(r'[,;\"\'`]', '', cleaned)  # Remove other punctuation
        cleaned = re.sub(r'[—–]', '-', cleaned)  # Replace em/en dashes
        
        # Keep more characters for scientific terms
        cleaned = re.sub(r'[^\w\s\-.\(\)]', '', cleaned)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned if len(cleaned) >= 3 else ""

class OpenAIREStrategy(BaseAPIStrategy):
    def get_api_name(self) -> str:
        return "openaire"

    def parse_response(self, response: requests.Response) -> List[Dict]:
        """Parse OpenAIRE API response"""
        try:
            data = response.json()
            if data is None:
                return []
            
            results = data.get("results", [])
            return results if isinstance(results, list) else []
            
        except (ValueError, requests.exceptions.JSONDecodeError) as e:
            if self.debug:
                print(f"OpenAIRE JSON parse error: {e}")
            return []
        except Exception as e:
            if self.debug:
                print(f"OpenAIRE parse error: {e}")
            return []

    def _build_api_url(self, query_params: Dict[str, Any], per_page: int = 1) -> str:
        """Build OpenAIRE Graph API URL from query parameters"""
        base_url = "https://api.openaire.eu/graph/v1/researchProducts"
        
        # Handle OR syntax conversion for OpenAIRE
        processed_params = {}
        for key, value in query_params.items():
            if key == "mainTitle" and "|" in str(value):
                # Convert OpenAlex OR syntax (|) to OpenAIRE OR syntax ( OR )
                openaire_or_query = "(" + str(value).replace("|", ") OR (") + ")"
                processed_params[key] = openaire_or_query
            else:
                processed_params[key] = value
        
        # Use parent class encoding method with quote_plus for OpenAIRE
        encoded_params = self.encode_query_params(processed_params, "quote_plus")
        
        if encoded_params:
            params_list = [f"{key}={value}" for key, value in encoded_params.items()]
            query_string = "&".join(params_list)
            final_url = f"{base_url}?{query_string}&pageSize={per_page}"
            #print(f"DEBUG. URL={final_url}")
            return final_url
        
        return f"{base_url}?pageSize={per_page}"

class PubMedStrategy(BaseAPIStrategy):
    """PubMed API search strategy with custom two-step process"""
    
    def get_api_name(self) -> str:
        return "pubmed"
    
    def _build_api_url(self, query_params: Dict[str, Any], per_page: int = 1) -> str:
        """Build PubMed search URL (step 1 - get PMIDs)"""
        search_url = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        
        # Build search term from field parameters
        term_parts = []
        for key, value in query_params.items():
            if not value:
                continue
                
            # DON'T encode here - use raw values
            if key == "doi":
                term_parts.append(f"{value}[doi]")
            elif key in ["title", "author", "journal", "pdat", "volume", "issue", "first_page", "last_page"]:
                term_parts.append(f"{value}[{key}]")
        
        if term_parts:
            search_term = " AND ".join(term_parts)
            # Only encode once, at the end
            encoded_search_term = self.encode_query_value(search_term, "quote_plus")
            final_url = f"{search_url}?db=pubmed&retmode=json&retmax={per_page}&term={encoded_search_term}"
            return final_url
        
        return f"{search_url}?db=pubmed&retmode=json&retmax={per_page}"
    
    def parse_response(self, response: requests.Response) -> List[Dict]:
        """Handle PubMed's two-step process: search for PMIDs, then fetch articles"""
        try:
            # Step 1: Parse search response to get PMIDs
            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            
            if not pmids:
                if self.debug:
                    print("No PMIDs found in search response")
                return []
            
            if self.debug:
                print(f"Found {len(pmids)} PMIDs, fetching articles...")
            
            # Step 2: Fetch and parse articles
            return self._fetch_and_parse_articles(pmids)
            
        except (ValueError, requests.exceptions.JSONDecodeError) as e:
            if self.debug:
                print(f"PubMed JSON parse error: {e}")
            return []
        except Exception as e:
            if self.debug:
                print(f"PubMed parse error: {e}")
            return []
    
    def _fetch_and_parse_articles(self, pmids: List[str]) -> List[Dict]:
        """Fetch full PubMed articles for given PMIDs and parse to dicts"""
        fetch_url = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        parsed_articles = []
        
        try:
            for pmid in pmids[:20]:  # Limit to avoid timeouts
                if self.debug:
                    print(f"Fetching article for PMID: {pmid}")
                
                fetch_params = {
                    "db": "pubmed",
                    "retmode": "xml",
                    "id": pmid,
                }
                
                fetch_response = requests.get(fetch_url, params=fetch_params, timeout=60)
                fetch_response.raise_for_status()
                
                if fetch_response.text:
                    parsed_article = self._parse_pubmed_xml_to_dict(fetch_response.text)
                    if parsed_article:
                        parsed_articles.append(parsed_article)
                        if self.debug:
                            title = parsed_article.get('title', 'No title')[:50]
                            print(f"  Parsed: {title}...")
                
                # Small delay to be respectful to NCBI
                time.sleep(0.1)
                
        except Exception as e:
            if self.debug:
                print(f"Error fetching PubMed articles: {e}")
        
        if self.debug:
            print(f"Successfully parsed {len(parsed_articles)} PubMed articles")
        
        return parsed_articles
    
    def _parse_pubmed_xml_to_dict(self, xml_content: str) -> Dict[str, Any]:
        """Parse PubMed XML content into standardized dict structure"""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_content)
            
            # Initialize result dict
            result = {
                'xml_content': xml_content,  # Keep original for fallback
                'api_source': 'pubmed'
            }
            
            # Find the article element
            article = root.find("PubmedArticle/MedlineCitation/Article")
            if article is None:
                return result
            
            # Extract PMID (ID)
            pmid_elem = root.find("PubmedArticle/MedlineCitation/PMID")
            if pmid_elem is not None:
                result['id'] = pmid_elem.text
                result['pmid'] = pmid_elem.text
            
            # Extract DOI
            doi_elem = article.find(".//ELocationID[@EIdType='doi']")
            if doi_elem is not None:
                result['doi'] = doi_elem.text
            
            # Extract title
            title_elem = article.find("ArticleTitle")
            if title_elem is not None:
                result['title'] = title_elem.text or ""
            
            # Extract authors
            authors = []
            author_list = article.find("AuthorList")
            if author_list is not None:
                for author in author_list.findall("Author"):
                    last_name = author.findtext("LastName", "")
                    fore_name = author.findtext("ForeName", "")
                    initials = author.findtext("Initials", "")
                    
                    # Build author name
                    if last_name:
                        author_name = f"{last_name}, {initials or (fore_name[0] if fore_name else '')}."
                        authors.append({
                            'full_name': author_name,
                            'last_name': last_name,
                            'fore_name': fore_name,
                            'initials': initials
                        })
            result['authors'] = authors
            
            # Extract publication year
            pub_date = article.find("Journal/JournalIssue/PubDate")
            if pub_date is not None:
                year_elem = pub_date.find("Year")
                if year_elem is not None:
                    result['publication_year'] = year_elem.text
                else:
                    # Try MedlineDate format
                    medline_date = pub_date.findtext("MedlineDate", "")
                    if medline_date:
                        # Extract year from formats like "2020 Jan-Feb" or "2020"
                        import re
                        year_match = re.search(r'(\d{4})', medline_date)
                        if year_match:
                            result['publication_year'] = year_match.group(1)
            
            # Extract journal information
            journal_elem = article.find("Journal")
            if journal_elem is not None:
                # Journal title
                journal_title = journal_elem.findtext("Title", "")
                if not journal_title:
                    journal_title = journal_elem.findtext("ISOAbbreviation", "")
                result['journal'] = journal_title
                
                # Volume and Issue
                journal_issue = journal_elem.find("JournalIssue")
                if journal_issue is not None:
                    result['volume'] = journal_issue.findtext("Volume", "")
                    result['issue'] = journal_issue.findtext("Issue", "")
            
            # Extract pagination
            pagination = article.find("Pagination")
            if pagination is not None:
                result['first_page'] = pagination.findtext("StartPage", "")
                result['last_page'] = pagination.findtext("EndPage", "")
                
                # Also check for MedlinePgn format
                if not result.get('first_page'):
                    medline_pgn = pagination.findtext("MedlinePgn", "")
                    if medline_pgn:
                        # Handle formats like "123-456" or "e123456"
                        if '-' in medline_pgn:
                            parts = medline_pgn.split('-')
                            result['first_page'] = parts[0]
                            result['last_page'] = parts[1] if len(parts) > 1 else ""
                        else:
                            result['first_page'] = medline_pgn
            
            # Extract abstract (optional)
            abstract_elem = article.find("Abstract/AbstractText")
            if abstract_elem is not None:
                result['abstract'] = abstract_elem.text or ""
            
            return result
            
        except Exception as e:
            if self.debug:
                print(f"Error parsing PubMed XML: {e}")
            # Return minimal dict with original XML on parse error
            return {
                'xml_content': xml_content,
                'api_source': 'pubmed',
                'parse_error': str(e)
            }

class CrossRefStrategy(BaseAPIStrategy):
    """CrossRef API search strategy with enhanced DOI support"""
    
    def get_api_name(self) -> str:
        return "crossref"

    def parse_response(self, response: requests.Response) -> List[Dict]:
        """Parse CrossRef API response"""
        try:
            data = response.json()
            if data is None:
                return []
            
            message = data.get("message")
            if message is None:
                return []
            
            items = message.get("items", [])
            return items if isinstance(items, list) else []
            
        except (ValueError, requests.exceptions.JSONDecodeError) as e:
            if self.debug:
                print(f"CrossRef JSON parse error: {e}")
            return []
        except Exception as e:
            if self.debug:
                print(f"CrossRef parse error: {e}")
            return []

    def _build_api_url(self, query_params: Dict[str, Any], per_page: int = 1) -> str:
        """Build CrossRef API URL from query parameters"""
        base_url = "https://api.crossref.org/works"
        
        # Build query string from parameters
        encoded_params = self.encode_query_params(query_params, "quote_plus")
        
        if encoded_params:
            params_list = [f"{key}={value}" for key, value in encoded_params.items()]
            query_string = "&".join(params_list)
            final_url = f"{base_url}?{query_string}&rows={per_page}"
            #print(f"DEBUG. URL={final_url}")
            return final_url
        
        return f"{base_url}?rows={per_page}"

class HALSearchStrategy(BaseAPIStrategy):
    """HAL API search strategy with enhanced DOI support"""
    
    def get_api_name(self) -> str:
        return "hal"

    def parse_response(self, response: requests.Response) -> List[Dict]:
        """Parse HAL API response"""
        try:
            data = response.json()
            if data is None:
                return []
            
            response_data = data.get("response")
            if response_data is None:
                return []
            
            docs = response_data.get("docs", [])
            return docs if isinstance(docs, list) else []
            
        except (ValueError, requests.exceptions.JSONDecodeError) as e:
            if self.debug:
                print(f"HAL JSON parse error: {e}")
            return []
        except Exception as e:
            if self.debug:
                print(f"HAL parse error: {e}")
            return []

    def _build_api_url(self, query_params: Dict[str, Any], per_page: int = 1) -> str:
        """Build HAL API URL from query parameters"""
        base_url = "https://api.archives-ouvertes.fr/search/"
        
        # Build query parts WITHOUT pre-encoding individual values
        query_parts = []
        for key, value in query_params.items():
            if not value:
                continue
            
            # Don't encode individual values - just build the query structure
            if key.endswith("_t"):  # Full-text fields - add quotes
                query_parts.append(f'{key}:"{value}"')
            else:  # Exact fields - add quotes  
                query_parts.append(f'{key}:"{value}"')
        
        if query_parts:
            query_string = " AND ".join(query_parts)
            # Only encode once - the complete query string
            encoded_query = self.encode_query_value(query_string, "quote_plus")
            final_url = f"{base_url}?q={encoded_query}&fl=*&wt=json&rows={per_page}"
            #print(f"DEBUG. URL={final_url}")
            return final_url
        
        return f"{base_url}?fl=*&wt=json&rows={per_page}"

class SearchAPI:
    """Main search API coordinator with enhanced DOI support and reduced verbosity"""
    
    def __init__(self, debug: bool = False):
        self.strategies = {
            "openalex": OpenAlexStrategy(debug=debug),
            "openaire": OpenAIREStrategy(debug=debug),
            "pubmed": PubMedStrategy(debug=debug),
            "crossref": CrossRefStrategy(debug=debug),
            "hal": HALSearchStrategy(debug=debug),
        }
        self.debug = debug
        self.orchestrator = SearchOrchestrator(debug=debug)
           
    def search_api(self, ner_entities: Dict[str, List[str]], api: str = "openalex", 
                  target_count: int = 1, **kwargs) -> List[Dict]:
        
        if self.debug:
            print(f"\n=== DEBUG: SearchAPI.search_api for {api} ===")
            print(f"NER entities: {ner_entities}")
            print(f"Target count: {target_count}")
        
        if api not in self.strategies:
            raise ValueError(f"Unsupported API: {api}")
        
        if api not in APICapabilities.get_supported_apis():
            raise ValueError(f"API {api} not configured in APICapabilities")
        
        strategy = self.strategies[api]
        strategy.debug = self.debug
        
        # Use the orchestrator to do the actual search
        self.orchestrator.debug = self.debug
        results = self.orchestrator.search_single_api(ner_entities, api, target_count, strategy)
        
        # Enhance results with DOI information
        enhanced_results = []
        for i, result in enumerate(results or []):
            enhanced_result = strategy.enhance_result_with_dois(result, api)
            enhanced_results.append(enhanced_result)
        
        if self.debug:
            print(f"Search returned {len(enhanced_results)} results")
        
        return enhanced_results
           
    def search_multiple_apis(self, ner_entities: Dict[str, List[str]], 
                           apis: List[str], target_count_per_api: int = 5) -> Dict[str, List[Dict]]:
        """Search multiple APIs and return results by API with enhanced DOI support"""
        return self.orchestrator.search_multiple_apis(
            ner_entities, apis, target_count_per_api, self.strategies
        )
