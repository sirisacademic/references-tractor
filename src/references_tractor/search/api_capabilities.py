# search/api_capabilities.py
"""
Defines the search capabilities and field mappings for each supported API.
Enhanced with DOI extraction configurations for multiple DOI support.
"""

from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

@dataclass
class RetryConfig:
    """Configuration for API retry behavior"""
    max_retries: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 30.0  # Maximum delay in seconds
    backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    rate_limit_delay: float = 0.5  # Delay between successful calls
    timeout: int = 30  # Request timeout in seconds

@dataclass
class SearchFieldConfig:
    """Configuration for a search field in an API"""
    api_field_name: str  # The actual parameter name for the API
    field_type: str  # 'exact', 'search', 'filter', 'range'
    required_preprocessing: Optional[str] = None  # Function name for preprocessing
    supports_multiple: bool = False  # Whether field accepts multiple values

@dataclass
class ResponseFieldConfig:
    """Configuration for extracting fields from API responses"""
    path: str  # JSONPath or XPath to the field
    field_type: str  # 'string', 'list', 'nested'
    fallback_paths: Optional[List[str]] = None  # Alternative paths to try

@dataclass
class DOIExtractionConfig:
    """Configuration for DOI extraction from API responses"""
    supports_multiple_dois: bool  # Whether this API can return multiple DOIs per result
    main_doi_path: str  # Path to the primary DOI
    alternative_doi_paths: List[str]  # Paths to alternative DOIs
    extraction_notes: str  # Notes about this API's DOI patterns

class APICapabilities:
    """Defines search and response capabilities for each API"""
    
    # DOI extraction configurations for each API
    DOI_EXTRACTION_CONFIGS = {
        "openalex": DOIExtractionConfig(
            supports_multiple_dois=False,
            main_doi_path="doi",
            alternative_doi_paths=[],
            extraction_notes="OpenAlex typically returns one DOI per result. Multiple results may contain different DOIs for the same paper."
        ),
        "openaire": DOIExtractionConfig(
            supports_multiple_dois=True,
            main_doi_path="pids[?(@.scheme=='doi')].value",
            alternative_doi_paths=[
                "instances[*].pids[?(@.scheme=='doi')].value"
            ],
            extraction_notes="OpenAIRE often has multiple DOIs in pids array and instance-level pids. Same paper may have conference and journal DOIs."
        ),
        "crossref": DOIExtractionConfig(
            supports_multiple_dois=False,
            main_doi_path="DOI",
            alternative_doi_paths=[],
            extraction_notes="CrossRef typically has one DOI per result. References contain cited DOIs, not alternatives for the same paper."
        ),
        "pubmed": DOIExtractionConfig(
            supports_multiple_dois=False,
            main_doi_path="PubmedArticle/MedlineCitation/Article/ELocationID[@EIdType='doi']",
            alternative_doi_paths=[],
            extraction_notes="PubMed usually has one DOI per article in XML format."
        ),
        "hal": DOIExtractionConfig(
            supports_multiple_dois=False,
            main_doi_path="doiId_s",
            alternative_doi_paths=[],
            extraction_notes="HAL typically has one DOI per entry."
        )
    }
    
    # Mapping from NER entities to API-specific search configurations
    SEARCH_CAPABILITIES = {
        "openalex": {
            "DOI": SearchFieldConfig("doi", "exact", "clean_doi"),
            "TITLE": SearchFieldConfig("title.search", "search", "clean_title"),
            "AUTHORS": SearchFieldConfig("raw_author_name.search", "search", "extract_author_surname_boolean"),
            "PUBLICATION_YEAR": SearchFieldConfig("publication_year", "exact"),
            "JOURNAL": SearchFieldConfig("locations.source.id", "exact", "resolve_journal_id"),
            "VOLUME": SearchFieldConfig("biblio.volume", "exact"),
            "ISSUE": SearchFieldConfig("biblio.issue", "exact"),
            "PAGE_FIRST": SearchFieldConfig("biblio.first_page", "exact"),
            "PAGE_LAST": SearchFieldConfig("biblio.last_page", "exact"),
            "TITLE_SEGMENTED": SearchFieldConfig("title.search", "search", "segment_title_for_or_search"),
        },
        "pubmed": {
            "DOI": SearchFieldConfig("doi", "exact", "clean_doi"),
            "TITLE": SearchFieldConfig("title", "search"),
            "AUTHORS": SearchFieldConfig("author", "search", "extract_author_surname"),
            "PUBLICATION_YEAR": SearchFieldConfig("pdat", "exact"),
            "JOURNAL": SearchFieldConfig("journal", "search"),
            "VOLUME": SearchFieldConfig("volume", "exact"),
            "ISSUE": SearchFieldConfig("issue", "exact"),
            "PAGE_FIRST": SearchFieldConfig("first_page", "exact"),
            "PAGE_LAST": SearchFieldConfig("last_page", "exact"),
        },
        "openaire": {
            "DOI": SearchFieldConfig("pid", "exact", "clean_doi"),
            "TITLE": SearchFieldConfig("mainTitle", "search"),
            "AUTHORS": SearchFieldConfig("authorFullName", "search", "extract_author_surname_boolean"),
            "PUBLICATION_YEAR": SearchFieldConfig("fromPublicationDate", "range", "year_to_date_range"),
            "TITLE_SEGMENTED": SearchFieldConfig("mainTitle", "search", "segment_title_for_or_search"),
        },
        "crossref": {
            "DOI": SearchFieldConfig("query", "search", "clean_doi"),
            "TITLE": SearchFieldConfig("query.title", "search"),
            "AUTHORS": SearchFieldConfig("query.author", "search", "extract_author_surname"),
            "PUBLICATION_YEAR": SearchFieldConfig("filter", "filter", "year_to_filter"),
            "JOURNAL": SearchFieldConfig("query.container-title", "search"),
        },
        "hal": {
            "DOI": SearchFieldConfig("doiId_s", "exact", "clean_doi"),
            "TITLE": SearchFieldConfig("title_t", "search"),
            "AUTHORS": SearchFieldConfig("authFullName_t", "search", "extract_author_surname"),
            "PUBLICATION_YEAR": SearchFieldConfig("publicationDateY_s", "exact"),
            "JOURNAL": SearchFieldConfig("journalTitle_t", "search"),
            "VOLUME": SearchFieldConfig("volume_s", "exact"),
            "ISSUE": SearchFieldConfig("issue_s", "exact"),
            "PAGE_FIRST": SearchFieldConfig("page_s", "search"),
            "PAGE_LAST": SearchFieldConfig("page_s", "search"),
        }
    }

    # API-specific retry configurations
    RETRY_CONFIGS = {
        "openalex": RetryConfig(
            max_retries=3,
            base_delay=1.0,
            rate_limit_delay=0.5,
            timeout=20
        ),
        "openaire": RetryConfig(
            max_retries=5,
            base_delay=2.0,
            rate_limit_delay=1.0,
            timeout=45
        ),
        "crossref": RetryConfig(
            max_retries=5,
            base_delay=2.0,
            rate_limit_delay=1.0,
            timeout=45
        ),
        "pubmed": RetryConfig(
            max_retries=5,
            base_delay=2.0,
            rate_limit_delay=1.2,
            timeout=60
        ),
        "hal": RetryConfig(
            max_retries=3,
            base_delay=1.5,
            rate_limit_delay=0.7,
            timeout=25
        )
    }
    
    # API-specific field combinations ordered by precision (restrictive -> broad)
    FIELD_COMBINATIONS = {
        "openalex": [
            # Tier 1: DOI (highest precision)
            ["DOI"],
            
            # Tier 2: Title combinations (high precision)
            ["TITLE", "PUBLICATION_YEAR", "AUTHORS", "JOURNAL"],
            ["TITLE", "PUBLICATION_YEAR", "AUTHORS"],
            ["TITLE", "PUBLICATION_YEAR", "JOURNAL"],
            ["TITLE", "PUBLICATION_YEAR"],
            ["TITLE", "AUTHORS"],
            ["TITLE"],
            
            # Tier 3: Segmented title combinations (medium-high precision)
            ["TITLE_SEGMENTED", "PUBLICATION_YEAR", "AUTHORS", "JOURNAL"],
            ["TITLE_SEGMENTED", "PUBLICATION_YEAR", "AUTHORS"],
            ["TITLE_SEGMENTED", "PUBLICATION_YEAR", "JOURNAL"],
            ["TITLE_SEGMENTED", "PUBLICATION_YEAR"],
            ["TITLE_SEGMENTED", "AUTHORS"],
            ["TITLE_SEGMENTED"],
            
            # Tier 4: Bibliographic fingerprint combinations (medium precision)
            ["AUTHORS", "JOURNAL", "VOLUME", "ISSUE", "PAGE_FIRST", "PUBLICATION_YEAR"],
            ["JOURNAL", "VOLUME", "ISSUE", "PAGE_FIRST", "PUBLICATION_YEAR"],
            ["AUTHORS", "JOURNAL", "VOLUME", "PAGE_FIRST", "PUBLICATION_YEAR"],
            ["JOURNAL", "VOLUME", "PAGE_FIRST", "PUBLICATION_YEAR"],
            
            # Tier 5: Author + bibliographic combinations (medium precision)
            ["AUTHORS", "PUBLICATION_YEAR", "JOURNAL", "VOLUME", "PAGE_FIRST"],
            ["AUTHORS", "PUBLICATION_YEAR", "JOURNAL", "VOLUME"],
            ["AUTHORS", "PUBLICATION_YEAR", "JOURNAL", "ISSUE", "PAGE_FIRST"],
            ["AUTHORS", "PUBLICATION_YEAR", "JOURNAL"],
            
            # Tier 6: Journal + bibliographic combinations (medium-low precision)
            ["JOURNAL", "PUBLICATION_YEAR", "VOLUME", "ISSUE", "PAGE_FIRST"],
            ["JOURNAL", "PUBLICATION_YEAR", "VOLUME", "ISSUE"],
            ["JOURNAL", "PUBLICATION_YEAR", "VOLUME"],
            
            # Tier 7: Basic combinations (lower precision)
            ["AUTHORS", "PUBLICATION_YEAR"],
            ["JOURNAL", "PUBLICATION_YEAR"],
            ["AUTHORS", "JOURNAL"],
        ],
        
        "pubmed": [
            # Tier 1: DOI (highest precision)
            ["DOI"],
            
            # Tier 2: Title combinations (high precision)
            ["TITLE", "AUTHORS", "JOURNAL", "PUBLICATION_YEAR"],
            ["TITLE", "JOURNAL", "PUBLICATION_YEAR"],
            ["TITLE", "AUTHORS", "PUBLICATION_YEAR"],
            ["TITLE", "PUBLICATION_YEAR"],
            ["TITLE", "AUTHORS"],
            ["TITLE"],
            
            # Tier 3: Bibliographic fingerprint combinations (medium-high precision)
            # PubMed is excellent for biomedical journals with volume/issue/page
            ["AUTHORS", "JOURNAL", "VOLUME", "ISSUE", "PAGE_FIRST", "PUBLICATION_YEAR"],
            ["JOURNAL", "VOLUME", "ISSUE", "PAGE_FIRST", "PUBLICATION_YEAR"],
            ["AUTHORS", "JOURNAL", "VOLUME", "PAGE_FIRST", "PUBLICATION_YEAR"],
            ["JOURNAL", "VOLUME", "PAGE_FIRST", "PUBLICATION_YEAR"],
            
            # Tier 4: Author + journal combinations (medium precision)
            ["AUTHORS", "JOURNAL", "PUBLICATION_YEAR", "VOLUME"],
            ["AUTHORS", "JOURNAL", "PUBLICATION_YEAR", "PAGE_FIRST"],
            ["AUTHORS", "JOURNAL", "PUBLICATION_YEAR"],
            
            # Tier 5: Journal + bibliographic combinations (medium-low precision)
            ["JOURNAL", "PUBLICATION_YEAR", "VOLUME", "ISSUE"],
            ["JOURNAL", "PUBLICATION_YEAR", "VOLUME"],
            ["JOURNAL", "PUBLICATION_YEAR", "PAGE_FIRST"],
            
            # Tier 6: Basic combinations (lower precision)
            ["AUTHORS", "PUBLICATION_YEAR"],
            ["JOURNAL", "PUBLICATION_YEAR"],
        ],
        
        "openaire": [
            # Tier 1: DOI (highest precision)
            ["DOI"],
            
            # Tier 2: Title combinations (high precision)
            ["TITLE", "AUTHORS", "PUBLICATION_YEAR"],
            ["TITLE", "PUBLICATION_YEAR"],
            ["TITLE", "AUTHORS"],
            ["TITLE"],
            
            # Tier 3: Segmented title combinations (medium precision)
            ["TITLE_SEGMENTED", "AUTHORS", "PUBLICATION_YEAR"],
            ["TITLE_SEGMENTED", "PUBLICATION_YEAR"],
            ["TITLE_SEGMENTED", "AUTHORS"],
            ["TITLE_SEGMENTED"],
            
            # Tier 4: Non-title combinations (lower precision)
            # OpenAIRE has limited bibliographic search capabilities
            ["AUTHORS", "PUBLICATION_YEAR"],
        ],
        
        "crossref": [
            # Tier 1: DOI (highest precision)
            ["DOI"],
            
            # Tier 2: Title combinations (high precision)
            ["TITLE", "AUTHORS", "PUBLICATION_YEAR"],
            ["TITLE", "PUBLICATION_YEAR"],
            ["TITLE", "AUTHORS"],
            ["TITLE"],
            
            # Tier 3: Non-title combinations (lower precision)
            # CrossRef search is more limited for bibliographic details
            ["AUTHORS", "PUBLICATION_YEAR"],
            ["JOURNAL", "PUBLICATION_YEAR"],  # Using container-title
        ],
        
        "hal": [
            # Tier 1: DOI (highest precision)
            ["DOI"],
            
            # Tier 2: Title combinations (high precision)
            ["TITLE", "AUTHORS", "PUBLICATION_YEAR", "JOURNAL"],
            ["TITLE", "AUTHORS", "PUBLICATION_YEAR"],
            ["TITLE", "PUBLICATION_YEAR", "JOURNAL"],
            ["TITLE", "PUBLICATION_YEAR"],
            ["TITLE", "AUTHORS"],
            ["TITLE"],
            
            # Tier 3: Bibliographic fingerprint combinations (medium precision)
            # HAL supports good bibliographic search
            ["AUTHORS", "JOURNAL", "VOLUME", "ISSUE", "PAGE_FIRST", "PUBLICATION_YEAR"],
            ["JOURNAL", "VOLUME", "ISSUE", "PAGE_FIRST", "PUBLICATION_YEAR"],
            ["AUTHORS", "JOURNAL", "VOLUME", "PAGE_FIRST", "PUBLICATION_YEAR"],
            ["JOURNAL", "VOLUME", "PAGE_FIRST", "PUBLICATION_YEAR"],
            
            # Tier 4: Author + journal combinations (medium precision)
            ["AUTHORS", "JOURNAL", "PUBLICATION_YEAR", "VOLUME"],
            ["AUTHORS", "JOURNAL", "PUBLICATION_YEAR", "PAGE_FIRST"],
            ["AUTHORS", "JOURNAL", "PUBLICATION_YEAR"],
            
            # Tier 5: Journal + bibliographic combinations (medium-low precision)
            ["JOURNAL", "PUBLICATION_YEAR", "VOLUME", "ISSUE", "PAGE_FIRST"],
            ["JOURNAL", "PUBLICATION_YEAR", "VOLUME", "ISSUE"],
            ["JOURNAL", "PUBLICATION_YEAR", "VOLUME"],
            
            # Tier 6: Basic combinations (lower precision)
            ["AUTHORS", "PUBLICATION_YEAR"],
            ["JOURNAL", "PUBLICATION_YEAR"],
        ]
    }
    
    # Response field extraction configurations
    RESPONSE_FIELDS = {
        "openalex": {
            "id": ResponseFieldConfig("id", "string"),
            "doi": ResponseFieldConfig("doi", "string"),
            "title": ResponseFieldConfig("title", "string"),
            "authors": ResponseFieldConfig("authorships", "list"),
            "publication_year": ResponseFieldConfig("publication_year", "string"),
            "journal": ResponseFieldConfig("primary_location.source.display_name", "string"),
            "volume": ResponseFieldConfig("biblio.volume", "string"),
            "issue": ResponseFieldConfig("biblio.issue", "string"),
            "first_page": ResponseFieldConfig("biblio.first_page", "string"),
            "last_page": ResponseFieldConfig("biblio.last_page", "string"),
        },
        "openaire": {
            "id": ResponseFieldConfig("id", "string"),
            "doi": ResponseFieldConfig("pids", "nested"),
            "title": ResponseFieldConfig("mainTitle", "string"), 
            "authors": ResponseFieldConfig("authors", "list"),
            "publication_year": ResponseFieldConfig("publicationDate", "string"),
            "journal": ResponseFieldConfig("container.name", "string"),
        },
        "pubmed": {
            "id": ResponseFieldConfig("PubmedArticle/MedlineCitation/PMID", "string"),
            "doi": ResponseFieldConfig("PubmedArticle/MedlineCitation/Article/ELocationID[@EIdType='doi']", "string"),
            "title": ResponseFieldConfig("PubmedArticle/MedlineCitation/Article/ArticleTitle", "string"),
            "authors": ResponseFieldConfig("PubmedArticle/MedlineCitation/Article/AuthorList/Author", "list"),
            "publication_year": ResponseFieldConfig("PubmedArticle/MedlineCitation/Article/Journal/JournalIssue/PubDate/Year", "string"),
            "journal": ResponseFieldConfig("PubmedArticle/MedlineCitation/Article/Journal/Title", "string"),
            "volume": ResponseFieldConfig("PubmedArticle/MedlineCitation/Article/Journal/JournalIssue/Volume", "string"),
            "issue": ResponseFieldConfig("PubmedArticle/MedlineCitation/Article/Journal/JournalIssue/Issue", "string"),
        },
        "crossref": {
            "id": ResponseFieldConfig("DOI", "string"),
            "doi": ResponseFieldConfig("DOI", "string"),
            "title": ResponseFieldConfig("title[0]", "string"),
            "authors": ResponseFieldConfig("author", "list"),
            "publication_year": ResponseFieldConfig("issued.date-parts[0][0]", "string"),
            "journal": ResponseFieldConfig("container-title[0]", "string"),
            "volume": ResponseFieldConfig("volume", "string"),
            "issue": ResponseFieldConfig("issue", "string"),
            "pages": ResponseFieldConfig("page", "string"),
        },
        "hal": {
            "id": ResponseFieldConfig("halId_s", "string"),
            "doi": ResponseFieldConfig("doiId_s", "string"),
            "title": ResponseFieldConfig("title_s[0]", "string"),
            "authors": ResponseFieldConfig("authFullName_s", "list"),
            "publication_year": ResponseFieldConfig("publicationDateY_s", "string"),
            "journal": ResponseFieldConfig("journalTitle_s", "string"),
            "volume": ResponseFieldConfig("volume_s", "string"),
            "issue": ResponseFieldConfig("issue_s[0]", "string"),
            "pages": ResponseFieldConfig("page_s", "string"),
        }
    }
    
    @classmethod
    def get_search_fields(cls, api: str) -> Dict[str, SearchFieldConfig]:
        """Get search field configurations for an API"""
        return cls.SEARCH_CAPABILITIES.get(api, {})
    
    @classmethod
    def get_field_combinations(cls, api: str) -> List[List[str]]:
        """Get field combination strategies for an API"""
        return cls.FIELD_COMBINATIONS.get(api, [])
    
    @classmethod
    def get_response_fields(cls, api: str) -> Dict[str, ResponseFieldConfig]:
        """Get response field extraction configurations for an API"""
        return cls.RESPONSE_FIELDS.get(api, {})
    
    @classmethod
    def get_doi_extraction_config(cls, api: str) -> Optional[DOIExtractionConfig]:
        """Get DOI extraction configuration for an API"""
        return cls.DOI_EXTRACTION_CONFIGS.get(api)
    
    @classmethod
    def supports_multiple_dois(cls, api: str) -> bool:
        """Check if an API supports multiple DOIs per result"""
        config = cls.get_doi_extraction_config(api)
        return config.supports_multiple_dois if config else False
    
    @classmethod
    def supports_field(cls, api: str, field: str) -> bool:
        """Check if an API supports searching by a specific field"""
        search_fields = cls.get_search_fields(api)
        return field in search_fields
    
    @classmethod
    def get_supported_apis(cls) -> List[str]:
        """Get list of all supported APIs"""
        return list(cls.SEARCH_CAPABILITIES.keys())
    
    @classmethod
    def get_apis_with_multiple_dois(cls) -> List[str]:
        """Get list of APIs that support multiple DOIs per result"""
        return [api for api in cls.get_supported_apis() if cls.supports_multiple_dois(api)]

    @classmethod
    def get_retry_config(cls, api: str) -> RetryConfig:
        """Get retry configuration for an API"""
        return cls.RETRY_CONFIGS.get(api, RetryConfig())
    
    @classmethod
    def get_api_timeout(cls, api: str) -> int:
        """Get timeout for a specific API"""
        return cls.get_retry_config(api).timeout
    
    @classmethod
    def get_api_delay_settings(cls, api: str) -> Dict[str, float]:
        """Get delay settings for an API"""
        config = cls.get_retry_config(api)
        return {
            'base_delay': config.base_delay,
            'max_delay': config.max_delay,
            'backoff_multiplier': config.backoff_multiplier,
            'rate_limit_delay': config.rate_limit_delay
        }

    @classmethod
    def print_doi_capabilities_summary(cls):
        """Print a summary of DOI capabilities for each API"""
        print("DOI Extraction Capabilities Summary:")
        print("=" * 50)
        
        for api in cls.get_supported_apis():
            config = cls.get_doi_extraction_config(api)
            if config:
                print(f"\n{api.upper()}:")
                print(f"  Multiple DOIs: {config.supports_multiple_dois}")
                print(f"  Main DOI Path: {config.main_doi_path}")
                if config.alternative_doi_paths:
                    print(f"  Alternative Paths: {', '.join(config.alternative_doi_paths)}")
                print(f"  Notes: {config.extraction_notes}")
