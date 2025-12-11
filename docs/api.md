# API Reference

Complete API documentation for the References Tractor citation processing and linking system.

> **Installation & Setup**: For installation instructions and basic configuration, see the [README.md](README.md#quick-start) Quick Start guide.

## ReferencesTractor Class

The main class for citation processing and linking operations.

### Constructor

```python
ReferencesTractor(
    ner_model_path: str = "SIRIS-Lab/citation-parser-ENTITY",
    select_model_path: str = "SIRIS-Lab/citation-parser-SELECT", 
    prescreening_model_path: str = "SIRIS-Lab/citation-parser-TYPE",
    span_model_path: str = "SIRIS-Lab/citation-parser-SPAN",
    device: Union[int, str] = "auto",
    enable_caching: bool = True,
    cache_size_limit: int = 1000
)
```

**Parameters:**
- `ner_model_path` (str): Path to NER model for entity extraction
- `select_model_path` (str): Path to SELECT model for candidate ranking
- `prescreening_model_path` (str): Path to prescreening model for citation detection
- `span_model_path` (str): Path to span model for citation extraction from text
- `device` (str): Device for model inference ("auto", "cpu", "cuda", "mps")
- `enable_caching` (bool): Enable result caching for performance
- `cache_size_limit` (int): Maximum number of cached results

**Example:**
```python
from references_tractor import ReferencesTractor

# Initialize with default settings
ref_tractor = ReferencesTractor()

# Custom model paths and caching
ref_tractor = ReferencesTractor(
    ner_model_path="path/to/custom/ner/model",
    device="cuda",
    cache_size_limit=500
)
```

## Core Methods

### link_citation()

Link a single citation to a publication using a specific API.

```python
link_citation(
    citation: str,
    output: str = 'simple',
    api_target: str = 'openalex'
) -> Dict[str, Any]
```

**Parameters:**
- `citation` (str): Raw citation text to process
- `output` (str): Output detail level ('simple' or 'advanced')
- `api_target` (str): Target API ('openalex', 'openaire', 'pubmed', 'crossref', 'hal')

**Returns:**
- `Dict[str, Any]`: Citation linking results

**Simple Output Format:**
```python
{
    "result": "Smith, J. et al. (2019). Machine Learning in Healthcare. Nature Medicine, 25, 1234-1240. DOI: 10.1038/s41591-019-0123-4",
    "score": 0.95,
    "openalex_id": "W2963456789",
    "doi": "10.1038/s41591-019-0123-4",
    "url": "https://openalex.org/W2963456789"
}
```

**Advanced Output Format:**
```python
{
    "result": "Smith, J. et al. (2019). Machine Learning in Healthcare...",
    "score": 0.95,
    "openalex_id": "W2963456789", 
    "doi": "10.1038/s41591-019-0123-4",
    "main_doi": "10.1038/s41591-019-0123-4",
    "alternative_dois": ["10.1101/2019.123456"],
    "total_dois": 2,
    "all_dois": ["10.1038/s41591-019-0123-4", "10.1101/2019.123456"],
    "url": "https://openalex.org/W2963456789",
    "full-publication": { /* Complete metadata object */ }
}
```

**Example:**
```python
# Test different output formats
result_simple = ref_tractor.link_citation(
    "BERT: Pre-training of Deep Bidirectional Transformers",
    api_target="openalex"
)

result_advanced = ref_tractor.link_citation(
    "BERT: Pre-training of Deep Bidirectional Transformers", 
    output="advanced",
    api_target="openalex"
)

# Access metadata in advanced mode
if result_advanced:
    publication = result_advanced.get('full-publication', {})
    authors = publication.get('authors', [])
    venue = publication.get('venue', '')
```

### link_citation_ensemble()

Link citation using ensemble method with multiple APIs and DOI consensus voting.

```python
link_citation_ensemble(
    citation: str,
    output: str = 'simple',
    api_targets: List[str] = ['openalex', 'openaire', 'pubmed', 'crossref', 'hal']
) -> Dict[str, Any]
```

**Parameters:**
- `citation` (str): Raw citation text to process
- `output` (str): Output detail level ('simple' or 'advanced')
- `api_targets` (List[str]): List of APIs to use for consensus

**Returns:**
- `Dict[str, Any]`: Ensemble linking results with consensus information

**Output Format:**
```python
{
    "doi": "10.1038/s41591-019-0123-4",
    "external_ids": {
        "openalex_id": "W2963456789",
        "pubmed_id": "31234567", 
        "crossref_id": "10.1038/s41591-019-0123-4"
    },
    "ensemble_metadata": {
        "selected_doi_votes": 3,
        "total_dois_found": 1,
        "all_related_dois": ["10.1038/s41591-019-0123-4"],
        "contributing_apis": ["openalex", "pubmed", "crossref"],
        "doi_vote_breakdown": {"10.1038/s41591-019-0123-4": 3}
    }
}
```

**Example:**
```python
# Ensemble with custom API selection
biomedical_apis = ["openalex", "pubmed", "crossref"]
result = ref_tractor.link_citation_ensemble(
    "Effects of meditation on brain structure",
    api_targets=biomedical_apis
)

# Check consensus strength
metadata = result.get('ensemble_metadata', {})
confidence = metadata.get('selected_doi_votes', 0)
total_apis = len(metadata.get('contributing_apis', []))

print(f"Consensus: {confidence}/{total_apis} APIs agreed")
```

### extract_and_link_from_text()

Extract citations from text and link them to publications.

```python
extract_and_link_from_text(
    text: str,
    api_target: str = 'openalex'
) -> Dict[str, Dict[str, Any]]
```

**Parameters:**
- `text` (str): Input text containing citations
- `api_target` (str): Target API for linking extracted citations

**Returns:**
- `Dict[str, Dict[str, Any]]`: Mapping from extracted citations to linking results

**Example:**
```python
manuscript_text = """
The transformer architecture (Vaswani et al., 2017) revolutionized NLP.
BERT (Devlin et al., 2018) further advanced the field with bidirectional training.
GPT-3 (Brown et al., 2020) demonstrated the power of large language models.
"""

# Extract and link all citations
results = ref_tractor.extract_and_link_from_text(manuscript_text)

# Process results
for citation_text, link_result in results.items():
    if link_result:
        print(f"✓ Found: {citation_text}")
        print(f"  DOI: {link_result.get('doi', 'N/A')}")
        print(f"  Confidence: {link_result.get('score', 0):.2f}")
    else:
        print(f"✗ Not found: {citation_text}")
```

### process_ner_entities()

Extract named entities from citation text using NER model.

```python
process_ner_entities(citation: str) -> Dict[str, List[str]]
```

**Parameters:**
- `citation` (str): Raw citation text

**Returns:**
- `Dict[str, List[str]]`: Dictionary mapping entity types to extracted values

**Entity Types:**
- `TITLE`: Publication title
- `AUTHORS`: Author names  
- `DOI`: Digital Object Identifier
- `JOURNAL`: Journal or venue name
- `PUBLICATION_YEAR`: Publication year
- `VOLUME`: Volume number
- `ISSUE`: Issue number
- `PAGE_FIRST`: First page number
- `PAGE_LAST`: Last page number

**Example:**
```python
# Complex citation with multiple elements
citation = """
Smith, J., Johnson, A., & Brown, C. (2019). 
Machine Learning Applications in Healthcare: A Comprehensive Review. 
Nature Medicine, 25(8), 1234-1240. 
https://doi.org/10.1038/s41591-019-0123-4
"""

entities = ref_tractor.process_ner_entities(citation)

# Inspect extracted entities
for entity_type, values in entities.items():
    print(f"{entity_type}: {values}")

# Use entities for custom search logic
if entities.get('DOI'):
    print(f"Direct DOI available: {entities['DOI'][0]}")
elif entities.get('TITLE') and entities.get('AUTHORS'):
    print("Can search by title + authors")
else:
    print("Limited entity information available")
```

## Advanced Methods

### search_api()

Search a specific API with extracted entities (lower-level access).

```python
search_api(
    ner_entities: Dict[str, List[str]], 
    api: str = "openalex",
    target_count: int = 10
) -> List[dict]
```

**Parameters:**
- `ner_entities` (Dict[str, List[str]]): Extracted citation entities
- `api` (str): Target API name
- `target_count` (int): Number of candidates to retrieve

**Returns:**
- `List[dict]`: List of candidate publications

**Example:**
```python
# Manual entity extraction and search
citation = "BERT: Pre-training of Deep Bidirectional Transformers"
entities = ref_tractor.process_ner_entities(citation)

# Search specific APIs with custom parameters
openalex_candidates = ref_tractor.search_api(
    entities, 
    api="openalex", 
    target_count=5
)

pubmed_candidates = ref_tractor.search_api(
    entities, 
    api="pubmed", 
    target_count=3
)

# Compare results across APIs
print(f"OpenAlex found {len(openalex_candidates)} candidates")
print(f"PubMed found {len(pubmed_candidates)} candidates")
```

### generate_apa_citation()

Generate APA-style citation from publication metadata.

```python
generate_apa_citation(data: dict, api: str = "openalex") -> str
```

**Parameters:**
- `data` (dict): Publication metadata from API
- `api` (str): Source API for proper formatting

**Returns:**
- `str`: Formatted APA citation

**Example:**
```python
# Get full publication data
result = ref_tractor.link_citation(
    "Attention is all you need",
    output="advanced",
    api_target="openalex"
)

if result and 'full-publication' in result:
    # Generate formatted citation
    apa_citation = ref_tractor.generate_apa_citation(
        result['full-publication'], 
        api="openalex"
    )
    print("APA Citation:")
    print(apa_citation)
```

## Utility Methods

### get_cache_stats()

Get current caching performance statistics.

```python
get_cache_stats() -> Dict[str, Any]
```

**Returns:**
- `Dict[str, Any]`: Cache performance metrics

**Output Format:**
```python
{
    'cache_enabled': True,
    'cache_size': 245,
    'cache_limit': 1000,
    'cache_hits': 67,
    'cache_misses': 178,
    'hit_rate': '27.3%',
    'total_requests': 245
}
```

**Example:**
```python
# Monitor cache performance
stats = ref_tractor.get_cache_stats()

print(f"Cache status: {'Enabled' if stats['cache_enabled'] else 'Disabled'}")
print(f"Usage: {stats['cache_size']}/{stats['cache_limit']} entries")
print(f"Hit rate: {stats['hit_rate']}")

# Optimize cache usage
hit_rate = float(stats['hit_rate'].rstrip('%'))
if hit_rate < 30:
    print("Consider increasing cache size for better performance")
```

### clear_cache()

Clear the citation cache and reset statistics.

```python
clear_cache() -> None
```

**Example:**
```python
# Reset cache when switching datasets
ref_tractor.clear_cache()
print("Cache cleared - fresh start for new evaluation")
```

## Helper Functions

### get_uri()

Construct canonical URL for a publication.

```python
get_uri(pid: Optional[str], doi: Optional[str], api: str) -> Optional[str]
```

**Parameters:**
- `pid` (str): Publication ID
- `doi` (str): DOI
- `api` (str): Source API

**Returns:**
- `Optional[str]`: Canonical URL or None

### extract_id()

Extract publication ID from API response.

```python
extract_id(publication: dict, api: str) -> Optional[str]
```

**Parameters:**
- `publication` (dict): Publication metadata
- `api` (str): Source API

**Returns:**
- `Optional[str]`: Publication ID or None

### extract_doi()

Extract DOI from API response.

```python
extract_doi(publication: dict, api: str) -> Optional[str]
```

**Parameters:**
- `publication` (dict): Publication metadata  
- `api` (str): Source API

**Returns:**
- `Optional[str]`: DOI or None

## Error Handling

### Common Exceptions

#### ImportError
Raised when required dependencies are missing.

```python
try:
    from references_tractor import ReferencesTractor
except ImportError as e:
    print(f"Installation error: {e}")
    print("Please install with: pip install references-tractor")
```

#### ValueError
Raised for invalid API names or parameters.

```python
try:
    result = ref_tractor.link_citation(citation, api_target="invalid_api")
except ValueError as e:
    print(f"Invalid API: {e}")
    # Valid APIs: openalex, openaire, pubmed, crossref, hal
```

#### ConnectionError
Raised for network connectivity issues.

```python
try:
    result = ref_tractor.link_citation(citation)
except ConnectionError as e:
    print(f"Network error: {e}")
    print("Please check internet connectivity")
```

### Error Response Format

When processing fails, methods return error information:

```python
{
    "error": "This text is not a citation. Please introduce a valid citation.",
    "status": "ERROR"
}
```

**Empty Results:**
When no match is found, methods return empty dict:

```python
{}  # No match found
```

**Handling Errors in Code:**
```python
def safe_citation_linking(citation: str) -> Optional[str]:
    """Safely attempt citation linking with error handling."""
    try:
        result = ref_tractor.link_citation(citation)
        
        # Check for error status
        if result.get('status') == 'ERROR':
            print(f"Processing error: {result.get('error')}")
            return None
        
        # Check for empty result (no match)
        if not result:
            print("No match found")
            return None
        
        # Extract DOI
        return result.get('doi')
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

## Advanced Configuration

### Environment Variables

Set environment variables to customize behavior:

```bash
# Model paths
export REFERENCES_TRACTOR_NER_MODEL="path/to/ner/model"
export REFERENCES_TRACTOR_SELECT_MODEL="path/to/select/model"

# Cache settings
export REFERENCES_TRACTOR_CACHE_SIZE="2000"
export REFERENCES_TRACTOR_ENABLE_CACHE="true"

# Device selection
export REFERENCES_TRACTOR_DEVICE="cuda"
```

### Custom Model Configuration

```python
# Advanced model configuration
ref_tractor = ReferencesTractor(
    ner_model_path="custom/ner/model",
    select_model_path="custom/select/model",
    prescreening_model_path="custom/prescreening/model",
    span_model_path="custom/span/model",
    device="auto",
    enable_caching=True,
    cache_size_limit=2000
)

# Verify model loading
print(f"NER device: {ref_tractor.ner_pipeline.device}")
print(f"SELECT device: {ref_tractor.select_pipeline.device}")
```

### API-Specific Configuration

```python
# Configure retry behavior and timeouts
import os

# Set API-specific timeouts
os.environ['OPENALEX_TIMEOUT'] = '30'
os.environ['PUBMED_TIMEOUT'] = '25'

# Configure retry limits
os.environ['API_MAX_RETRIES'] = '3'
os.environ['API_RETRY_DELAY'] = '1.0'
```

## Batch Processing Patterns

### Sequential Processing

```python
def process_citations_sequentially(citations: List[str]) -> List[Dict]:
    """Process citations one by one with progress tracking."""
    results = []
    
    for i, citation in enumerate(citations):
        print(f"Processing {i+1}/{len(citations)}: {citation[:50]}...")
        
        try:
            result = ref_tractor.link_citation(citation)
            results.append({
                'citation': citation,
                'result': result,
                'status': 'success',
                'index': i
            })
        except Exception as e:
            results.append({
                'citation': citation,
                'error': str(e),
                'status': 'error',
                'index': i
            })
    
    return results
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_citations_parallel(citations: List[str], max_workers: int = 4) -> List[Dict]:
    """Process citations in parallel with controlled concurrency."""
    
    def process_single(citation_data):
        index, citation = citation_data
        try:
            result = ref_tractor.link_citation(citation)
            return {
                'index': index,
                'citation': citation,
                'result': result,
                'status': 'success'
            }
        except Exception as e:
            return {
                'index': index,
                'citation': citation,
                'error': str(e),
                'status': 'error'
            }
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single, (i, citation)): i 
            for i, citation in enumerate(citations)
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    # Sort by original order
    results.sort(key=lambda x: x['index'])
    return results
```

## Performance Monitoring

### Timing and Profiling

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(description: str):
    """Context manager for timing operations."""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"{description}: {elapsed:.2f} seconds")

# Time individual operations
with timer("Citation linking"):
    result = ref_tractor.link_citation("Your citation here")

with timer("NER extraction"):
    entities = ref_tractor.process_ner_entities("Your citation here")

with timer("Ensemble linking"):
    ensemble_result = ref_tractor.link_citation_ensemble("Your citation here")
```

### Memory Usage Monitoring

```python
import psutil
import os

def monitor_memory_usage():
    """Monitor current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print(f"RSS Memory: {memory_info.rss / 1024 / 1024:.1f} MB")
    print(f"VMS Memory: {memory_info.vms / 1024 / 1024:.1f} MB")

# Monitor before and after processing
print("Before initialization:")
monitor_memory_usage()

ref_tractor = ReferencesTractor()

print("After initialization:")
monitor_memory_usage()
```

> **Performance Considerations**: For hardware requirements, optimization tips, and performance benchmarks, see the [README.md Performance Considerations](README.md#performance-considerations) section.

## Troubleshooting

### Model Loading Issues

```python
# Check model availability and device compatibility
def diagnose_model_loading():
    """Diagnose model loading issues."""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
        
        from transformers import pipeline
        print("Transformers library available")
        
        # Test model loading
        ref_tractor = ReferencesTractor(device="cpu")
        print("✓ Models loaded successfully on CPU")
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")

diagnose_model_loading()
```

### API Connectivity Testing

```python
import requests

def test_api_connectivity():
    """Test connectivity to all supported APIs."""
    apis = {
        'OpenAlex': 'https://api.openalex.org/works?filter=title.search:test',
        'CrossRef': 'https://api.crossref.org/works?query=test',
        'PubMed': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=test'
    }
    
    for api_name, test_url in apis.items():
        try:
            response = requests.get(test_url, timeout=10)
            status = "✓" if response.status_code == 200 else f"✗ ({response.status_code})"
            print(f"{api_name}: {status}")
        except Exception as e:
            print(f"{api_name}: ✗ ({str(e)})")

test_api_connectivity()
```

### Debug Mode

```python
import logging

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test with debug output
ref_tractor = ReferencesTractor()
result = ref_tractor.link_citation(
    "Your problematic citation",
    output="advanced"
)
```

> **Additional Help**: For comprehensive troubleshooting and development guidance, see the [Contributing Guide](contribute.md) and [Evaluation Documentation](evaluation.md).