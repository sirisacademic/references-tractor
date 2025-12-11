# Evaluation System Documentation

The References Tractor evaluation system provides comprehensive assessment of citation linking performance across multiple academic databases using classification metrics and detailed error analysis.

## Overview

The evaluation framework tests the system's ability to correctly link raw citation text to scholarly publications using a gold standard dataset of manually validated citation-to-publication mappings.

### Supported APIs

- **OpenAlex**: Open academic search engine with broad coverage
- **OpenAIRE**: European open science infrastructure  
- **PubMed**: Biomedical literature database
- **CrossRef**: Academic metadata registry
- **HAL**: French open archive system
- **Ensemble**: Multi-API consensus method using DOI voting

## Quick Start

### Basic Evaluation

```bash
# Run full evaluation
python evaluation/evaluate_citations.py

# Test with limited citations
python evaluation/evaluate_citations.py --limit 5

# Use loose evaluation criteria
python evaluation/evaluate_citations.py --evaluation-mode loose
```

### Custom Configuration

```bash
python evaluation/evaluate_citations.py \
    --gold-standard custom_test.json \
    --output-dir results/ \
    --device auto \
    --evaluation-mode strict \
    --show-preview
```

### Regenerating Results

```bash
# Regenerate summary from existing results
python evaluation/generate_summary_evaluation.py \
    results/05_results_20250623_170050.json \
    strict \
    new_output/
```

## Evaluation Methodology

### Gold Standard Dataset

The evaluation uses manually validated citation mappings in JSON format:

```json
{
    "Citation text goes here": {
        "doi": "10.1038/s41591-019-0123-4",
        "openalex": "W2963456789",
        "openaire": "doi_dedup___::abc123",
        "pubmed": "31234567",
        "crossref": "10.1038/s41591-019-0123-4",
        "hal": "hal-12345678"
    }
}
```

### Evaluation Modes

#### Strict Mode (Default)
- Only exact DOI/ID matches count as correct
- Highest precision, most conservative evaluation
- Recommended for production system assessment

#### Loose Mode
- Exact matches AND likely matches count as correct
- Includes metadata-based similarity matches
- Useful for development and system tuning

### Classification Metrics

The system categorizes results into specific types for detailed analysis:

#### Correct Results
- **Correct Matches (C_Match)**: Citations correctly linked to expected results
- **Correct No Result (C_NoRes)**: Citations that correctly returned no match when none expected

#### Incorrect Results
- **Incorrect Matches (I_Match)**: Wrong links when correct link was expected
- **Incorrect Missing (I_Miss)**: No result when correct link was expected  
- **Incorrect Spurious (I_Spur)**: Unexpected result when no link was expected

#### Technical Issues
- **ERROR**: Technical failures during processing (excluded from accuracy calculations)

### Matching Logic

#### Individual API Evaluation
1. **DOI Matching**: Compare retrieved DOI(s) against expected DOI
2. **ID Matching**: Compare API-specific IDs (OpenAlex ID, PubMed ID, etc.)
3. **Multiple DOI Support**: Check both main and alternative DOIs
4. **Empty Result Handling**: Validate appropriate no-result responses

#### Ensemble Evaluation
1. **DOI Consensus**: Majority voting among API results
2. **Quality Filtering**: Only include high-confidence API results
3. **External ID Backfill**: Search for additional IDs using consensus DOI

## Output Files

### 1. Summary Dashboard (`01_summary_dashboard_*.txt`)

Executive summary with comprehensive metrics table:

| Column | Description |
|--------|-------------|
| **API** | Database/search engine name |
| **Accuracy** | Percentage of correct predictions |
| **Total** | Citations processed (excluding errors) |
| **T_Corr** | Total Correct - All correct classifications |
| **C_Match** | Correct Matches - Properly linked citations |
| **C_NoRes** | Correct No Result - Appropriate empty results |
| **T_Incor** | Total Incorrect - All incorrect classifications |
| **I_Match** | Incorrect Matches - Wrong linkages |
| **I_Miss** | Incorrect Missing - Missed expected results |
| **I_Spur** | Incorrect Spurious - Unexpected false positives |

**Example Output:**
```
API          Accuracy  Total  T_Corr  C_Match  C_NoRes  T_Incor  I_Match  I_Miss  I_Spur
Openalex     0.845     100    84      78       6        16       8        6       2
Openaire     0.782     100    78      72       6        22       10       8       4
Pubmed       0.892     100    89      83       6        11       5        4       2
Ensemble     0.924     100    92      88       4        8        3        3       2
```

### 2. Accuracy Metrics (`02_accuracy_metrics_*.tsv`)

Detailed breakdown in TSV format for analysis tools. Contains same metrics as summary dashboard in spreadsheet-compatible format.

### 3. Comparison Table (`03_comparison_table_*.tsv`)

Citation-by-citation comparison showing:
- Original citation text
- Expected results from gold standard
- Retrieved results from each API
- Final evaluation status for each API

### 4. Detailed API Results (`04_*_detailed_*.tsv`)

Individual API performance with complete metadata:
- Citation processing details
- Retrieved publication information
- Confidence scores and status codes
- Formatted citations and DOI information

### 5. Raw Results (`05_results_*.json`)

Complete evaluation data in JSON format for custom analysis and debugging.

## Performance Analysis

### Interpreting Accuracy Scores

#### High Performance (>0.8)
- **Characteristics**: Low error rates across all categories
- **Indicators**: High C_Match, low I_Miss/I_Spur/I_Match
- **Meaning**: Reliable for production use

#### Medium Performance (0.5-0.8)
- **Characteristics**: Some errors but generally functional
- **Indicators**: Moderate error rates in specific categories
- **Meaning**: May need threshold tuning or domain-specific optimization

#### Low Performance (<0.5)
- **Characteristics**: High error rates, unreliable results
- **Indicators**: High values in multiple incorrect categories
- **Meaning**: Requires investigation and system improvements

### Error Pattern Analysis

#### High I_Miss (Missing Results)
**Symptoms**: API fails to find papers that should be discoverable
**Possible Causes**:
- Limited database coverage
- Overly restrictive search criteria
- Citation format not recognized
- API-specific indexing gaps

**Solutions**:
- Broaden search strategy
- Improve NER entity extraction
- Add fallback search methods

#### High I_Spur (Spurious Results)
**Symptoms**: API returns matches when none expected
**Possible Causes**:
- Overly broad search criteria
- Low confidence thresholds
- Common title/author name confusion

**Solutions**:
- Increase confidence thresholds
- Improve result filtering
- Add negative examples to training

#### High I_Match (Incorrect Matches)
**Symptoms**: API finds papers but links to wrong publications
**Possible Causes**:
- Similar titles/authors causing confusion
- Metadata quality issues
- Disambiguation problems

**Solutions**:
- Improve SELECT model training
- Add more discriminative features
- Implement better ranking strategies

### API-Specific Performance Patterns

#### OpenAlex
- **Strengths**: Broad coverage, good general performance
- **Weaknesses**: May have gaps in recent publications
- **Optimization**: Focus on publication year handling

#### PubMed  
- **Strengths**: High precision in biomedical domain
- **Weaknesses**: Limited coverage outside life sciences
- **Optimization**: Domain-specific citation processing

#### OpenAIRE
- **Strengths**: Strong European research coverage
- **Weaknesses**: May miss non-European publications
- **Optimization**: Geographic and institutional handling

#### CrossRef
- **Strengths**: Comprehensive DOI coverage
- **Weaknesses**: Limited full-text search capabilities
- **Optimization**: DOI-based search strategies

#### Ensemble
- **Strengths**: Combines API advantages, improves accuracy
- **Weaknesses**: Requires multiple API calls, slower processing
- **Optimization**: API selection and weighting strategies

## Configuration Options

### Command Line Parameters

#### Basic Options
```bash
--gold-standard PATH     # Gold standard JSON file (default: evaluation/api_linking_test.json)
--limit N                # Process only first N citations (for testing)
--output-dir DIR         # Output directory (default: evaluation_results)
--evaluation-mode MODE   # strict or loose (default: strict)
```

#### Device Control
```bash
--device DEVICE          # auto, cpu, cuda, mps (default: auto)
```

#### Output Control
```bash
--verbose                # Enable detailed processing output
--show-preview           # Display metrics preview after completion
```

### Environment Variables

```bash
# Cache configuration
export REFERENCES_TRACTOR_CACHE_SIZE=1000
export REFERENCES_TRACTOR_ENABLE_CACHE=true

# Model paths
export REFERENCES_TRACTOR_NER_MODEL=path/to/ner/model
export REFERENCES_TRACTOR_SELECT_MODEL=path/to/select/model
```

## Processing Pipeline Details

> **System Architecture**: For the complete system architecture diagram and processing flow, see the [README.md System Architecture](README.md#system-architecture) section.

### Quality Thresholds

#### SELECT Model Confidence
- **High Confidence**: ≥0.80 (accept immediately)
- **Low Confidence**: <0.80 (require NER validation)

#### NER Similarity Fallback
- **Acceptance Threshold**: ≥0.70
- **Below Threshold**: Reject result

### Caching System

The evaluation system includes intelligent caching to improve performance:

#### Cache Behavior
- **Key Generation**: MD5 hash of citation + API + output mode
- **Size Management**: LRU eviction when limit exceeded
- **Statistics Tracking**: Hit rate, miss count, total requests

#### Cache Benefits
- Avoid duplicate API calls during evaluation
- Faster iteration during development
- Reduced API rate limiting impact

## Troubleshooting

### Common Issues

#### Low Overall Accuracy
**Symptoms**: All APIs show poor performance
**Potential Causes**:
- Gold standard quality issues
- Domain mismatch between training and test data
- System threshold misconfiguration

**Solutions**:
1. Validate gold standard annotations
2. Check domain coverage in test set
3. Experiment with loose evaluation mode
4. Review confidence thresholds

#### API-Specific Failures
**Symptoms**: Single API shows ERROR status frequently
**Potential Causes**:
- Network connectivity issues
- API rate limiting
- Authentication problems
- API endpoint changes

**Solutions**:
1. Check internet connectivity
2. Verify API access credentials
3. Review API documentation for changes
4. Implement retry logic

#### Performance Degradation
**Symptoms**: Slow evaluation processing
**Potential Causes**:
- Large evaluation dataset
- CPU-only processing
- Network latency
- Inefficient caching

**Solutions**:
1. Use `--limit` for testing
2. Enable GPU acceleration
3. Check network connectivity
4. Monitor cache hit rates

#### Memory Issues
**Symptoms**: Out of memory errors during evaluation
**Potential Causes**:
- Large model sizes
- Insufficient hardware
- Memory leaks in processing

**Solutions**:
1. Use CPU-only mode (`--device cpu`)
2. Reduce batch sizes in models
3. Clear cache frequently
4. Monitor memory usage

### Debugging Tips

#### Enable Verbose Output
```bash
python evaluation/evaluate_citations.py --verbose --limit 5
```

#### Check Individual Citations
```python
from references_tractor import ReferencesTractor

ref_tractor = ReferencesTractor()
citation = "Your problematic citation here"

# Test NER extraction
entities = ref_tractor.process_ner_entities(citation)
print("NER entities:", entities)

# Test individual API
result = ref_tractor.link_citation(citation, api_target="openalex", output="advanced")
print("API result:", result)
```

#### Analyze Cache Performance
```python
# Check cache statistics
stats = ref_tractor.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']}")
print(f"Total requests: {stats['total_requests']}")

# Clear cache if needed
ref_tractor.clear_cache()
```

## Advanced Usage

### Custom Gold Standards

Create custom test datasets for domain-specific evaluation:

```json
{
    "Your citation text": {
        "doi": "expected_doi",
        "openalex": "expected_openalex_id",
        "pubmed": "expected_pubmed_id"
    }
}
```

### Batch Evaluation

Process multiple evaluation sets:

```bash
# Evaluate different domains
python evaluation/evaluate_citations.py --gold-standard biomedical_test.json --output-dir bio_results/
python evaluation/evaluate_citations.py --gold-standard computer_science_test.json --output-dir cs_results/
```

### Performance Benchmarking

Compare system versions:

```bash
# Baseline evaluation
python evaluation/evaluate_citations.py --output-dir baseline_results/

# Modified system evaluation  
python evaluation/evaluate_citations.py --output-dir modified_results/

# Compare results
python evaluation/compare_evaluations.py baseline_results/ modified_results/
```

### Custom Evaluation Metrics

Add domain-specific evaluation criteria:

```python
# evaluation/custom_evaluator.py
from utils.citation_evaluator import CitationEvaluator

class DomainSpecificEvaluator(CitationEvaluator):
    def evaluate_biomedical_performance(self):
        """Evaluate performance on biomedical citations"""
        biomedical_keywords = ['pubmed', 'medline', 'doi', 'pmid']
        
        biomedical_results = []
        for result in self.results:
            citation = result['citation'].lower()
            if any(keyword in citation for keyword in biomedical_keywords):
                biomedical_results.append(result)
        
        # Calculate biomedical-specific metrics
        return self._calculate_metrics_for_subset(biomedical_results)
    
    def evaluate_recent_publications(self, year_threshold: int = 2020):
        """Evaluate performance on recent publications"""
        recent_results = []
        for result in self.results:
            # Extract year from citation or gold standard
            year = self._extract_year(result)
            if year and year >= year_threshold:
                recent_results.append(result)
        
        return self._calculate_metrics_for_subset(recent_results)
```

## Integration

### CI/CD Integration

Add evaluation to continuous integration:

```yaml
# .github/workflows/evaluation.yml
name: Citation Linking Evaluation
on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -e .
      - name: Run evaluation
        run: python evaluation/evaluate_citations.py --limit 10
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: evaluation-results
          path: evaluation_results/
```

### Monitoring Integration

Track evaluation metrics over time:

```python
import json
from datetime import datetime

# Load evaluation results
with open('results/05_results_timestamp.json') as f:
    results = json.load(f)

# Extract metrics for monitoring
metrics = {
    'timestamp': datetime.now().isoformat(),
    'total_citations': len(results),
    'apis': ['openalex', 'openaire', 'pubmed', 'crossref', 'hal'],
    'accuracy': {}
}

# Calculate accuracy per API
for api in metrics['apis']:
    correct = sum(1 for r in results 
                  if r.get('api_results', {}).get(api, {}).get('final_evaluation') == 'CORRECT')
    total = sum(1 for r in results 
                if r.get('api_results', {}).get(api, {}).get('final_evaluation') != 'ERROR')
    metrics['accuracy'][api] = correct / total if total > 0 else 0

# Send to monitoring system (Prometheus, DataDog, etc.)
print(json.dumps(metrics, indent=2))
```

### A/B Testing Framework

Compare different system configurations:

```python
# evaluation/ab_testing.py
import json
import numpy as np
from scipy import stats

class ABTestingFramework:
    def __init__(self, baseline_results: str, variant_results: str):
        self.baseline = self._load_results(baseline_results)
        self.variant = self._load_results(variant_results)
    
    def _load_results(self, file_path: str) -> dict:
        with open(file_path) as f:
            return json.load(f)
    
    def compare_accuracy(self, api: str = 'ensemble') -> dict:
        """Statistical comparison of accuracy between baseline and variant"""
        
        baseline_correct = [
            1 if r.get('api_results', {}).get(api, {}).get('final_evaluation') == 'CORRECT' else 0
            for r in self.baseline
        ]
        
        variant_correct = [
            1 if r.get('api_results', {}).get(api, {}).get('final_evaluation') == 'CORRECT' else 0
            for r in self.variant
        ]
        
        # Perform statistical test
        statistic, p_value = stats.ttest_ind(baseline_correct, variant_correct)
        
        baseline_accuracy = np.mean(baseline_correct)
        variant_accuracy = np.mean(variant_correct)
        
        return {
            'baseline_accuracy': baseline_accuracy,
            'variant_accuracy': variant_accuracy,
            'improvement': variant_accuracy - baseline_accuracy,
            'improvement_percentage': (variant_accuracy - baseline_accuracy) / baseline_accuracy * 100,
            'p_value': p_value,
            'statistically_significant': p_value < 0.05,
            'sample_size': len(baseline_correct)
        }

# Usage
ab_test = ABTestingFramework('baseline_results.json', 'variant_results.json')
comparison = ab_test.compare_accuracy('ensemble')
print(f"Improvement: {comparison['improvement_percentage']:.2f}% (p={comparison['p_value']:.4f})")
```

## Best Practices

### Evaluation Design
1. **Diverse Test Set**: Include citations from multiple domains and formats
2. **Representative Sampling**: Ensure test set matches production distribution
3. **Regular Updates**: Refresh gold standard as APIs evolve
4. **Version Control**: Track evaluation datasets and results

### Performance Optimization
1. **Hardware Selection**: Use GPU acceleration when available
2. **Batch Processing**: Process multiple citations efficiently
3. **Caching Strategy**: Enable caching for repeated evaluations
4. **Resource Monitoring**: Track memory and API usage

### Result Interpretation
1. **Context Matters**: Consider domain-specific performance expectations
2. **Error Analysis**: Focus on specific error types for improvement
3. **Comparative Analysis**: Compare across APIs and configurations
4. **Trend Monitoring**: Track performance changes over time

### Quality Assurance
1. **Gold Standard Validation**: Regularly review and update test annotations
2. **Inter-annotator Agreement**: Measure consistency in manual annotations
3. **Edge Case Testing**: Include challenging citations in evaluation sets
4. **Regression Testing**: Ensure changes don't degrade performance

### Documentation
1. **Evaluation Reports**: Document methodology and findings
2. **Performance Baselines**: Establish acceptable performance levels
3. **Improvement Tracking**: Record system changes and impact
4. **Sharing Results**: Make evaluation results accessible to team