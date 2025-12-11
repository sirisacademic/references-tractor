# Development Examples

This document provides detailed examples for extending References Tractor with new features, APIs, and models.

## Adding New APIs

This section demonstrates how to add support for a new academic database by implementing all required components.

### Example: Adding Support for Semantic Scholar

Here's a complete walkthrough of adding Semantic Scholar API support:

#### 1. API Configuration

First, add the API configuration to define search capabilities and DOI extraction:

```python
# search/api_capabilities.py

# Add to SEARCH_CAPABILITIES
SEARCH_CAPABILITIES["semantic_scholar"] = {
    "DOI": SearchFieldConfig("doi", "exact", "clean_doi"),
    "TITLE": SearchFieldConfig("query", "search"),
    "AUTHORS": SearchFieldConfig("query", "search", "extract_author_surname"),
    "PUBLICATION_YEAR": SearchFieldConfig("year", "exact"),
    "JOURNAL": SearchFieldConfig("query", "search"),
}

# Add to DOI_EXTRACTION_CONFIGS
DOI_EXTRACTION_CONFIGS["semantic_scholar"] = DOIExtractionConfig(
    supports_multiple_dois=False,
    main_doi_path="externalIds.DOI",
    alternative_doi_paths=[],
    extraction_notes="Semantic Scholar has consistent DOI structure in externalIds"
)

# Add to FIELD_COMBINATIONS
FIELD_COMBINATIONS["semantic_scholar"] = [
    ["DOI"],
    ["TITLE", "AUTHORS", "PUBLICATION_YEAR"],
    ["TITLE", "PUBLICATION_YEAR"],
    ["TITLE", "AUTHORS"],
    ["TITLE"],
    ["AUTHORS", "PUBLICATION_YEAR"],
]

# Add to RETRY_CONFIGS
RETRY_CONFIGS["semantic_scholar"] = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    rate_limit_delay=0.3,
    timeout=25
)
```

#### 2. Search Strategy Implementation

Create the search strategy class:

```python
# search/search_api.py

class SemanticScholarStrategy(BaseAPIStrategy):
    def get_api_name(self) -> str:
        return "semantic_scholar"
        
    def _build_api_url(self, query_params: Dict[str, Any]) -> str:
        """Build Semantic Scholar API URL from query parameters"""
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        # Handle different query types
        query_parts = []
        
        if "doi" in query_params:
            # DOI search - use specific endpoint
            doi = query_params["doi"]
            return f"https://api.semanticscholar.org/graph/v1/paper/{doi}"
        
        if "query" in query_params:
            # General query search
            query = str(query_params["query"])
            encoded_query = self.encode_query_value(query)
            query_parts.append(f"query={encoded_query}")
        
        if "year" in query_params:
            # Year filtering
            year = query_params["year"]
            query_parts.append(f"year={year}")
        
        # Add required fields for response
        fields = "paperId,externalIds,title,authors,year,venue,publicationDate"
        query_parts.append(f"fields={fields}")
        
        # Set result limit
        query_parts.append("limit=10")
        
        if query_parts:
            query_string = "&".join(query_parts)
            return f"{base_url}?{query_string}"
        
        return base_url
    
    def _parse_api_response(self, response: requests.Response) -> List[Dict]:
        """Parse Semantic Scholar API response"""
        try:
            data = response.json()
            
            # Handle different response formats
            if "data" in data:
                # Search results
                return data.get("data", [])
            elif "paperId" in data:
                # Single paper result
                return [data]
            else:
                return []
                
        except Exception as e:
            print(f"Error parsing Semantic Scholar response: {e}")
            return []
```

#### 3. Citation Formatter

Implement APA citation formatting for Semantic Scholar data:

```python
# search/citation_formatter.py

class SemanticScholarFormatter(CitationFormatter):
    def generate_apa_citation(self, data):
        """Generate APA citation from Semantic Scholar data"""
        
        # Extract authors
        authors = []
        for author in data.get('authors', []):
            author_name = author.get('name', '')
            if author_name:
                authors.append(author_name)
        
        # Format author string
        if len(authors) > 3:
            authors_str = ", ".join(authors[:3]) + ", et al."
        else:
            authors_str = ", ".join(authors)
        
        # Extract other fields
        title = data.get('title', 'Unknown Title')
        year = data.get('year', 'n.d.')
        venue = data.get('venue', '')
        
        # Extract DOI information
        main_doi, alternative_dois = self._get_doi_info(data)
        
        # If no enhanced structure, extract from externalIds
        if not main_doi:
            external_ids = data.get('externalIds', {})
            if external_ids and isinstance(external_ids, dict):
                main_doi = external_ids.get('DOI')
                alternative_dois = []
        
        doi_section = self._format_doi_section(main_doi, alternative_dois)
        
        # Build citation parts
        citation_parts = [
            f"{authors_str} ({year})." if authors_str else f"({year}).",
            f"{title}." if title else "",
            f"{venue}." if venue else "",
            doi_section
        ]
        
        return " ".join(part for part in citation_parts if part).strip()
```

#### 4. Register the New API

Add the new API to the main system:

```python
# search/search_api.py - in SearchAPI.__init__
self.strategies = {
    "openalex": OpenAlexStrategy(),
    "openaire": OpenAIREStrategy(),
    "pubmed": PubMedStrategy(),
    "crossref": CrossRefStrategy(),
    "hal": HALSearchStrategy(),
    "semantic_scholar": SemanticScholarStrategy(),  # Add this line
}

# search/citation_formatter.py - in CitationFormatterFactory
formatters = {
    "openalex": OpenAlexFormatter(),
    "openaire": OpenAIREFormatter(),
    "pubmed": PubMedFormatter(),
    "crossref": CrossrefFormatter(),
    "hal": HALFormatter(),
    "semantic_scholar": SemanticScholarFormatter(),  # Add this line
}
```

#### 5. Add Tests

Create comprehensive tests for the new API:

```python
# tests/unit/test_semantic_scholar.py
import pytest
from unittest.mock import Mock, patch
from references_tractor.search.search_api import SemanticScholarStrategy

class TestSemanticScholarStrategy:
    def setup_method(self):
        self.strategy = SemanticScholarStrategy()
    
    def test_api_name(self):
        assert self.strategy.get_api_name() == "semantic_scholar"
    
    def test_build_url_with_doi(self):
        query_params = {"doi": "10.1038/nature12373"}
        url = self.strategy._build_api_url(query_params)
        
        assert "api.semanticscholar.org" in url
        assert "10.1038/nature12373" in url
    
    def test_build_url_with_query(self):
        query_params = {"query": "machine learning", "year": "2020"}
        url = self.strategy._build_api_url(query_params)
        
        assert "query=machine%20learning" in url
        assert "year=2020" in url
        assert "fields=" in url

# tests/integration/test_semantic_scholar_integration.py
import pytest
from references_tractor import ReferencesTractor

@pytest.mark.integration
class TestSemanticScholarIntegration:
    def setup_method(self):
        self.ref_tractor = ReferencesTractor()
    
    def test_famous_paper_linking(self):
        """Test linking a well-known paper"""
        citation = "Attention Is All You Need. Vaswani et al. NIPS 2017."
        
        result = self.ref_tractor.link_citation(
            citation, 
            api_target="semantic_scholar"
        )
        
        # Should find the transformer paper
        if result:  # Only assert if API returned a result
            assert "attention" in result.get('result', '').lower()
            assert result.get('score', 0) > 0.5
    
    def test_doi_search(self):
        """Test DOI-based search"""
        citation = "DOI: 10.1038/nature12373"
        
        result = self.ref_tractor.link_citation(
            citation,
            api_target="semantic_scholar"
        )
        
        if result:
            assert result.get('doi') is not None
    
    def test_no_match_citation(self):
        """Test citation that should return no match"""
        citation = "This is not a real citation at all."
        
        result = self.ref_tractor.link_citation(
            citation,
            api_target="semantic_scholar"
        )
        
        # Should return empty dict for no match
        assert result == {}

# tests/unit/test_semantic_scholar_formatter.py
import pytest
from references_tractor.search.citation_formatter import SemanticScholarFormatter

class TestSemanticScholarFormatter:
    def setup_method(self):
        self.formatter = SemanticScholarFormatter()
    
    def test_format_complete_paper(self):
        """Test formatting with complete paper data"""
        data = {
            "title": "Attention Is All You Need",
            "authors": [
                {"name": "Ashish Vaswani"},
                {"name": "Noam Shazeer"},
                {"name": "Niki Parmar"},
                {"name": "Jakob Uszkoreit"}  # Will be truncated to "et al."
            ],
            "year": 2017,
            "venue": "NIPS",
            "externalIds": {
                "DOI": "10.48550/arXiv.1706.03762"
            }
        }
        
        citation = self.formatter.generate_apa_citation(data)
        
        assert "Vaswani" in citation
        assert "et al." in citation
        assert "(2017)" in citation
        assert "Attention Is All You Need" in citation
        assert "DOI: 10.48550/arXiv.1706.03762" in citation
    
    def test_format_minimal_paper(self):
        """Test formatting with minimal data"""
        data = {
            "title": "Test Paper",
            "year": 2020
        }
        
        citation = self.formatter.generate_apa_citation(data)
        
        assert "(2020)" in citation
        assert "Test Paper" in citation
```

#### 6. Update Documentation

Add the new API to documentation:

```python
# Update README.md supported APIs table
| API | Coverage | Specialization |
|-----|----------|----------------|
| **Semantic Scholar** | Computer science focus | AI/ML publications |

# Update docs/api.md with examples
# Update docs/evaluation.md with API-specific notes
```

## Adding New Models

This section shows how to integrate new transformer models into the system.

### Example: Adding a Custom NER Model

#### 1. Model Integration

```python
# core.py
class ReferencesTractor:
    def __init__(
        self,
        ner_model_path: str = "SIRIS-Lab/citation-parser-ENTITY",
        custom_ner_model_path: Optional[str] = None,  # Add new parameter
        # ... other parameters
    ):
        # Initialize standard NER model
        self.ner_pipeline = self._init_pipeline("ner", ner_model_path, device, agg_strategy="simple")
        
        # Initialize custom NER model if provided
        if custom_ner_model_path:
            self.custom_ner_pipeline = self._init_pipeline(
                "ner", custom_ner_model_path, device, agg_strategy="simple"
            )
        else:
            self.custom_ner_pipeline = None
    
    def process_ner_entities_custom(self, citation: str) -> Dict[str, List[str]]:
        """Extract entities using custom NER model"""
        if not self.custom_ner_pipeline:
            raise ValueError("Custom NER model not initialized")
        
        output = self.custom_ner_pipeline(citation)
        entities = {}
        
        for entity in output:
            key = entity.get("entity_group")
            entities.setdefault(key, []).append(entity.get("word", ""))
        
        # Apply same cleaning as standard NER
        from .utils.entity_validation import EntityValidator
        cleaned_entities = EntityValidator.validate_and_clean_entities(entities)
        
        return cleaned_entities
    
    def process_ner_entities_ensemble(self, citation: str) -> Dict[str, List[str]]:
        """Combine results from multiple NER models"""
        # Get results from standard model
        standard_entities = self.process_ner_entities(citation)
        
        # Get results from custom model if available
        if self.custom_ner_pipeline:
            custom_entities = self.process_ner_entities_custom(citation)
            
            # Merge results (prefer custom model when available)
            merged_entities = {}
            all_keys = set(standard_entities.keys()) | set(custom_entities.keys())
            
            for key in all_keys:
                standard_values = standard_entities.get(key, [])
                custom_values = custom_entities.get(key, [])
                
                # Use custom values if available, otherwise use standard
                if custom_values:
                    merged_entities[key] = custom_values
                else:
                    merged_entities[key] = standard_values
            
            return merged_entities
        
        return standard_entities
```

#### 2. Model Configuration

```python
# Add configuration options
class ModelConfig:
    """Configuration for model selection and ensemble"""
    
    def __init__(
        self,
        use_custom_ner: bool = False,
        use_ensemble_ner: bool = False,
        custom_ner_weight: float = 0.7,
        fallback_to_standard: bool = True
    ):
        self.use_custom_ner = use_custom_ner
        self.use_ensemble_ner = use_ensemble_ner
        self.custom_ner_weight = custom_ner_weight
        self.fallback_to_standard = fallback_to_standard

# Usage
ref_tractor = ReferencesTractor(
    custom_ner_model_path="path/to/custom/model",
    model_config=ModelConfig(use_ensemble_ner=True)
)
```

## Adding New Evaluation Metrics

This section demonstrates adding custom evaluation metrics to the system.

### Example: Adding Precision/Recall Metrics

#### 1. Extend the Evaluator

```python
# utils/citation_evaluator.py

def calculate_precision_recall_metrics(self, evaluation_mode: str = "strict") -> Dict[str, Dict[str, float]]:
    """Calculate precision, recall, and F1-score for each API"""
    metrics = {}
    
    for api in self.apis + ['ensemble']:
        # Initialize counters
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        
        for result in self.results:
            gold_standard = result['gold_standard']
            
            # Get expected value for this approach
            if api == 'ensemble':
                expected_value = gold_standard.get('doi')
                api_result = result.get('ensemble', {})
            else:
                expected_value = gold_standard.get(api)
                api_result = result.get('api_results', {}).get(api, {})
            
            final_evaluation = api_result.get('final_evaluation', 'ERROR')
            
            # Skip errors
            if final_evaluation == 'ERROR':
                continue
            
            # Determine expected and actual outcomes
            expected_positive = bool(expected_value)
            actual_positive = (final_evaluation == 'CORRECT')
            
            # Count outcomes
            if expected_positive and actual_positive:
                true_positives += 1
            elif not expected_positive and actual_positive:
                false_positives += 1
            elif expected_positive and not actual_positive:
                false_negatives += 1
            else:  # not expected_positive and not actual_positive
                true_negatives += 1
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate additional metrics
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        
        metrics[api] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'specificity': specificity,
            'accuracy': accuracy,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives
        }
    
    return metrics

def calculate_confidence_calibration(self) -> Dict[str, Dict[str, float]]:
    """Calculate confidence calibration metrics"""
    metrics = {}
    
    for api in self.apis:
        confidence_scores = []
        correctness = []
        
        for result in self.results:
            api_result = result.get('api_results', {}).get(api, {})
            
            score = api_result.get('score')
            final_eval = api_result.get('final_evaluation')
            
            if score is not None and final_eval != 'ERROR':
                confidence_scores.append(float(score))
                correctness.append(1 if final_eval == 'CORRECT' else 0)
        
        if confidence_scores:
            # Calculate calibration metrics
            import numpy as np
            
            # Bin scores and calculate calibration
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0  # Expected Calibration Error
            mce = 0  # Maximum Calibration Error
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = [(conf >= bin_lower) and (conf < bin_upper) 
                         for conf in confidence_scores]
                prop_in_bin = sum(in_bin) / len(in_bin)
                
                if prop_in_bin > 0:
                    accuracy_in_bin = sum([correctness[i] for i, in_bin_i in enumerate(in_bin) if in_bin_i]) / sum(in_bin)
                    avg_confidence_in_bin = sum([confidence_scores[i] for i, in_bin_i in enumerate(in_bin) if in_bin_i]) / sum(in_bin)
                    
                    calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                    ece += prop_in_bin * calibration_error
                    mce = max(mce, calibration_error)
            
            metrics[api] = {
                'ece': ece,  # Expected Calibration Error
                'mce': mce,  # Maximum Calibration Error
                'avg_confidence': np.mean(confidence_scores),
                'avg_accuracy': np.mean(correctness)
            }
    
    return metrics
```

#### 2. Update Dashboard Generation

```python
def generate_enhanced_dashboard(self, evaluation_mode: str = "strict") -> str:
    """Generate dashboard with additional metrics"""
    # Get existing metrics
    classification_metrics = self.calculate_classification_metrics(evaluation_mode)
    pr_metrics = self.calculate_precision_recall_metrics(evaluation_mode)
    calibration_metrics = self.calculate_confidence_calibration()
    
    output = []
    
    # ... existing dashboard code ...
    
    # Add precision/recall section
    output.append("")
    output.append("PRECISION/RECALL METRICS:")
    output.append("="*80)
    
    header = f"{'API':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Specificity':<12}"
    output.append(header)
    output.append("-"*80)
    
    for api in self.apis + ['ensemble']:
        if api in pr_metrics:
            m = pr_metrics[api]
            row = f"{api.title():<12} {m['precision']:<10.3f} {m['recall']:<10.3f} {m['f1_score']:<10.3f} {m['specificity']:<12.3f}"
            output.append(row)
    
    # Add calibration section
    output.append("")
    output.append("CONFIDENCE CALIBRATION:")
    output.append("="*60)
    
    header = f"{'API':<12} {'ECE':<8} {'MCE':<8} {'Avg_Conf':<10} {'Avg_Acc':<10}"
    output.append(header)
    output.append("-"*60)
    
    for api in self.apis:
        if api in calibration_metrics:
            m = calibration_metrics[api]
            row = f"{api.title():<12} {m['ece']:<8.3f} {m['mce']:<8.3f} {m['avg_confidence']:<10.3f} {m['avg_accuracy']:<10.3f}"
            output.append(row)
    
    # Add interpretation guide
    output.append("")
    output.append("ADDITIONAL METRICS INTERPRETATION:")
    output.append("-"*50)
    output.append("PRECISION/RECALL:")
    output.append("  • Precision: Of predicted positives, how many are actually correct")
    output.append("  • Recall: Of actual positives, how many are correctly predicted")
    output.append("  • F1-Score: Harmonic mean of precision and recall")
    output.append("  • Specificity: Of actual negatives, how many are correctly identified")
    output.append("")
    output.append("CONFIDENCE CALIBRATION:")
    output.append("  • ECE: Expected Calibration Error (lower is better)")
    output.append("  • MCE: Maximum Calibration Error (lower is better)")
    output.append("  • Well-calibrated models have ECE and MCE close to 0")
    
    return "\n".join(output)
```

