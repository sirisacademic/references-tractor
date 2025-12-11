# Contributing to References Tractor

Thank you for your interest in contributing to References Tractor! This guide will help you get started with contributing to the project.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment tool (venv, conda, or virtualenv)

### Setting Up Development Environment

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/your-username/references-tractor.git
   cd references-tractor
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e .[dev]
   ```

4. **Verify Installation**
   ```bash
   python -c "from references_tractor import ReferencesTractor; print('Installation successful')"
   ```

> **Note**: For basic installation and usage instructions, see the [README.md Quick Start](README.md#quick-start) guide.

## Development Workflow

### Code Style and Standards

We follow PEP 8 and use automated tools to maintain code quality:

#### Code Formatting
```bash
# Format code with black
black references_tractor/ evaluation/ tests/

# Check formatting
black --check references_tractor/
```

#### Linting
```bash
# Run flake8 for style checking
flake8 references_tractor/ --max-line-length=88

# Run mypy for type checking
mypy references_tractor/
```

#### Import Sorting
```bash
# Sort imports with isort
isort references_tractor/ evaluation/ tests/
```

### Pre-commit Hooks

Set up pre-commit hooks to automatically check code quality:

```bash
pip install pre-commit
pre-commit install
```

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88]
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_core.py

# Run with coverage
pytest --cov=references_tractor tests/

# Run only unit tests (skip integration tests)
pytest tests/ -m "not integration"
```

### Writing Tests

#### Unit Tests
Place unit tests in `tests/unit/`:

```python
# tests/unit/test_ner_extraction.py
import pytest
from references_tractor import ReferencesTractor

def test_ner_entity_extraction():
    ref_tractor = ReferencesTractor()
    citation = "Smith, J. et al. Test Paper. Nature 123:456-789, 2020."
    
    entities = ref_tractor.process_ner_entities(citation)
    
    assert "TITLE" in entities
    assert "AUTHORS" in entities
    assert "PUBLICATION_YEAR" in entities
    assert entities["PUBLICATION_YEAR"][0] == "2020"

def test_citation_linking_flow():
    """Test the complete citation linking workflow"""
    ref_tractor = ReferencesTractor()
    citation = "Famous paper citation"
    
    # Test that processing doesn't crash
    result = ref_tractor.link_citation(citation, api_target="openalex")
    
    # Result should be either a dict with results or empty dict
    assert isinstance(result, dict)
    
    # If result found, should have expected structure
    if result:
        assert 'result' in result or 'doi' in result

def test_error_handling():
    """Test error handling for invalid inputs"""
    ref_tractor = ReferencesTractor()
    
    # Test empty citation
    result = ref_tractor.link_citation("", api_target="openalex")
    assert result == {} or 'error' in result
    
    # Test invalid API
    with pytest.raises(ValueError):
        ref_tractor.link_citation("test citation", api_target="invalid_api")
```

#### Integration Tests
Place integration tests in `tests/integration/`:

```python
# tests/integration/test_api_linking.py
import pytest
from references_tractor import ReferencesTractor

@pytest.mark.integration
def test_openalex_linking():
    """Test actual API linking with OpenAlex"""
    ref_tractor = ReferencesTractor()
    citation = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    
    result = ref_tractor.link_citation(citation, api_target="openalex")
    
    # Should return either valid result or empty dict (not error)
    assert isinstance(result, dict)
    assert result.get('status') != 'ERROR'

@pytest.mark.integration  
def test_ensemble_linking():
    """Test ensemble linking with multiple APIs"""
    ref_tractor = ReferencesTractor()
    citation = "Attention Is All You Need"
    
    result = ref_tractor.link_citation_ensemble(citation)
    
    # Should return valid ensemble result structure
    assert isinstance(result, dict)
    if result:
        assert 'ensemble_metadata' in result or 'doi' in result

@pytest.mark.integration
@pytest.mark.slow
def test_text_extraction():
    """Test citation extraction from text"""
    ref_tractor = ReferencesTractor()
    text = """
    Recent work in NLP (Devlin et al., BERT, 2018) has shown significant improvements.
    The transformer architecture (Vaswani et al., 2017) enabled these advances.
    """
    
    results = ref_tractor.extract_and_link_from_text(text)
    
    # Should extract citations and attempt linking
    assert isinstance(results, dict)
    assert len(results) >= 0  # May or may not find citations
```

#### Test Configuration
Create `pytest.ini`:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    integration: marks tests as integration tests (deselect with '-m "not integration"')
    slow: marks tests as slow (deselect with '-m "not slow"')
addopts = --strict-markers
```

### Test Coverage

Aim for >80% test coverage:

```bash
# Generate coverage report
pytest --cov=references_tractor --cov-report=html tests/

# View coverage
open htmlcov/index.html

# Check coverage thresholds
pytest --cov=references_tractor --cov-fail-under=80 tests/
```

### Performance Testing

Create performance benchmarks:

```python
# tests/performance/test_benchmarks.py
import time
import pytest
from references_tractor import ReferencesTractor

@pytest.mark.slow
def test_citation_linking_performance():
    """Benchmark citation linking performance"""
    ref_tractor = ReferencesTractor()
    citations = [
        "Smith et al. Nature 2020",
        "Johnson Science 2019", 
        "Brown Cell 2021"
    ]
    
    start_time = time.time()
    
    for citation in citations:
        ref_tractor.link_citation(citation)
    
    elapsed = time.time() - start_time
    avg_time = elapsed / len(citations)
    
    # Performance assertion (adjust threshold as needed)
    assert avg_time < 5.0, f"Average processing time {avg_time:.2f}s exceeds threshold"

@pytest.mark.slow
def test_memory_usage():
    """Monitor memory usage during processing"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    ref_tractor = ReferencesTractor()
    
    # Process several citations
    for i in range(10):
        ref_tractor.link_citation(f"Test citation {i}")
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Memory should not increase dramatically
    assert memory_increase < 500, f"Memory increased by {memory_increase:.1f}MB"
```

## Documentation

### Code Documentation

#### Docstrings
Use Google-style docstrings:

```python
def link_citation(self, citation: str, api_target: str = 'openalex', output: str = 'simple') -> Dict[str, Any]:
    """Link a citation to a publication using the specified API.
    
    Args:
        citation: Raw citation text to process
        api_target: Target API for searching ('openalex', 'pubmed', etc.)
        output: Output detail level ('simple' or 'advanced')
        
    Returns:
        Dictionary containing linking results with DOI, confidence score, and metadata
        
    Raises:
        ValueError: If api_target is not supported
        ConnectionError: If API is unreachable
        
    Example:
        >>> ref_tractor = ReferencesTractor()
        >>> result = ref_tractor.link_citation("Smith et al. Nature 2020")
        >>> print(result.get('doi'))
    """
```

#### Type Hints
Use type hints for all public functions:

```python
from typing import Dict, List, Optional, Union, Any

def process_ner_entities(self, citation: str) -> Dict[str, List[str]]:
    """Extract entities with proper type annotations."""
    pass

def search_api(
    self, 
    ner_entities: Dict[str, List[str]], 
    api: str = "openalex",
    target_count: int = 10
) -> List[Dict[str, Any]]:
    """Search API with type-safe parameters."""
    pass
```

### Documentation Updates

Update relevant documentation files when making changes:
- `README.md` for main features and usage
- `docs/api.md` for API changes  
- `docs/evaluation.md` for evaluation system changes
- `docs/examples.md` for implementation examples

## Feature Development Process

### Creating New Features

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Design Discussion**
   - Open an issue to discuss the feature
   - Get feedback from maintainers
   - Agree on design approach

3. **Implementation**
   - Write code following style guidelines
   - Add comprehensive tests
   - Update documentation

4. **Testing**
   - Run full test suite
   - Test with different device configurations
   - Run evaluation tests if relevant

5. **Documentation**
   - Update API documentation
   - Add usage examples
   - Update README if needed

### Adding New Components

For detailed examples of adding new APIs, models, or evaluation metrics, see the [Development Examples](examples.md) documentation.

#### Quick Reference

**Adding New APIs:**
1. Add configuration to `search/api_capabilities.py`
2. Implement strategy class in `search/search_api.py`
3. Add formatter in `search/citation_formatter.py`
4. Register in main classes
5. Add comprehensive tests

**Adding New Models:**
1. Extend model initialization in `core.py`
2. Add processing methods
3. Update configuration options
4. Test with different devices

**Adding New Evaluation Metrics:**
1. Extend evaluator in `utils/citation_evaluator.py`
2. Update dashboard generation
3. Add metric documentation

## Pull Request Process

### Before Submitting

1. **Code Quality Checklist**
   - [ ] Code follows PEP 8 style guidelines
   - [ ] All tests pass
   - [ ] Test coverage is maintained or improved
   - [ ] Documentation is updated
   - [ ] Type hints are included
   - [ ] No debug print statements

2. **Testing Checklist**
   - [ ] Unit tests cover new functionality
   - [ ] Integration tests pass
   - [ ] Evaluation tests pass (if applicable)
   - [ ] Tests work on different devices (CPU/GPU)

3. **Documentation Checklist**
   - [ ] API documentation updated
   - [ ] Usage examples included
   - [ ] README updated if needed
   - [ ] Docstrings are complete

### Submitting Pull Request

1. **Create Pull Request**
   - Use descriptive title
   - Reference related issues
   - Provide detailed description

2. **Pull Request Template**
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement

   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests pass
   - [ ] Manual testing completed

   ## Documentation
   - [ ] Code comments updated
   - [ ] API documentation updated
   - [ ] User documentation updated

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Tests pass
   - [ ] Documentation updated
   ```

### Review Process

1. **Automated Checks**
   - CI/CD pipeline runs
   - Code quality checks
   - Test suite execution

2. **Manual Review**
   - Code review by maintainers
   - Architecture feedback
   - Documentation review

3. **Approval and Merge**
   - Address review feedback
   - Get approval from maintainers
   - Merge into main branch

## Development Best Practices

### Code Organization

Follow the established file structure:
```
references_tractor/
├── __init__.py                 # Main package exports
├── core.py                     # Main ReferencesTractor class
├── search/                     # Search and API modules
│   ├── search_api.py          # API strategy implementations
│   ├── citation_formatter.py  # Citation formatting
│   └── api_capabilities.py    # API configurations
├── utils/                      # Utility modules
│   ├── citation_evaluator.py  # Evaluation framework
│   └── entity_validation.py   # Entity processing
tests/
├── unit/                       # Unit tests
├── integration/               # Integration tests
└── performance/               # Performance tests
evaluation/                     # Evaluation framework
docs/                          # Documentation
```

### Error Handling

Implement graceful degradation:
```python
def robust_api_call(self, url: str, timeout: int = 30) -> Optional[Dict]:
    """Make API call with proper error handling."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logging.warning(f"API timeout for {url}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed for {url}: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON response from {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error for {url}: {e}")
        return None
```

### Performance Considerations

- Use appropriate caching strategies
- Implement batch processing for large datasets
- Consider memory usage with large models
- Test performance across different devices

### Logging Best Practices

```python
import logging

# Set up structured logging
logger = logging.getLogger(__name__)

def process_citation(self, citation: str) -> Dict:
    """Process citation with proper logging."""
    logger.info(f"Processing citation: {citation[:50]}...")
    
    try:
        # Processing logic
        result = self._do_processing(citation)
        
        logger.info(f"Successfully processed citation: {bool(result)}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to process citation: {str(e)}", exc_info=True)
        raise

# Use structured logging for debugging
logger.debug("Citation entities extracted", extra={
    'citation_length': len(citation),
    'entities_found': len(entities),
    'entity_types': list(entities.keys())
})
```

### Security Considerations

```python
# Sanitize user inputs
def sanitize_citation(citation: str) -> str:
    """Sanitize citation text for safe processing."""
    if not isinstance(citation, str):
        raise ValueError("Citation must be a string")
    
    # Remove potential script injections
    citation = citation.strip()
    if len(citation) > 10000:  # Reasonable length limit
        raise ValueError("Citation text too long")
    
    return citation

# Handle sensitive configuration
import os
from pathlib import Path

def load_api_config() -> Dict:
    """Load API configuration securely."""
    config_file = Path("config.json")
    
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override with environment variables (more secure)
    config.update({
        'api_timeout': int(os.getenv('API_TIMEOUT', '30')),
        'max_retries': int(os.getenv('MAX_RETRIES', '3')),
        'cache_size': int(os.getenv('CACHE_SIZE', '1000'))
    })
    
    return config
```

## Communication

### Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, general discussion
- **Code Reviews**: Pull request discussions

### Issue Guidelines

#### Bug Reports
```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Initialize with '...'
2. Call method '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.9.7]
- References Tractor version: [e.g. 1.0.0]
- Device: [CPU/CUDA/MPS]

**Additional Context**
Any other context about the problem.
```

#### Feature Requests
```markdown
**Feature Description**
A clear description of what you want to happen.

**Use Case**
Describe the use case and why this feature would be valuable.

**Proposed Implementation**
If you have ideas about how this could be implemented.

**Alternatives Considered**
Any alternative solutions you've considered.
```

### Contribution Guidelines

#### Code Contributions
1. **Start Small**: Begin with documentation fixes or small bug fixes
2. **Discuss First**: Open an issue before working on major features
3. **Follow Standards**: Use established code style and patterns
4. **Add Tests**: Include comprehensive tests for new functionality
5. **Update Docs**: Keep documentation in sync with code changes

#### Documentation Contributions
1. **Clarity**: Write clear, concise documentation
2. **Examples**: Include practical, working examples
3. **Completeness**: Cover edge cases and error scenarios
4. **Consistency**: Follow existing documentation style

### Community Standards

#### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn and contribute
- Report inappropriate behavior

#### Review Standards
- Provide specific, actionable feedback
- Explain the reasoning behind suggestions
- Acknowledge good practices
- Be patient with learning contributors

## Advanced Development Topics

### Custom Model Integration

```python
# Example: Adding a custom preprocessing pipeline
from transformers import AutoTokenizer, AutoModel

class CustomPreprocessor:
    """Custom preprocessing for domain-specific citations"""
    
    def __init__(self, model_name: str = "custom/citation-preprocessor"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def preprocess_citation(self, citation: str) -> str:
        """Apply domain-specific preprocessing"""
        # Custom preprocessing logic
        processed = self._clean_citation(citation)
        processed = self._normalize_entities(processed)
        return processed
    
    def _clean_citation(self, citation: str) -> str:
        """Clean citation text"""
        # Remove extra whitespace
        citation = ' '.join(citation.split())
        
        # Normalize unicode characters
        import unicodedata
        citation = unicodedata.normalize('NFKD', citation)
        
        return citation
    
    def _normalize_entities(self, citation: str) -> str:
        """Normalize specific entities"""
        # Example: standardize journal names
        journal_mappings = {
            'Nat Med': 'Nature Medicine',
            'Nat. Med.': 'Nature Medicine',
            # Add more mappings
        }
        
        for abbrev, full_name in journal_mappings.items():
            citation = citation.replace(abbrev, full_name)
        
        return citation
```

### Performance Profiling

```python
# Example: Memory and performance profiling
import cProfile
import pstats
import tracemalloc
from memory_profiler import profile

def profile_citation_processing():
    """Profile citation processing performance"""
    
    # Start memory tracing
    tracemalloc.start()
    
    # CPU profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        ref_tractor = ReferencesTractor()
        
        # Test citations
        citations = [
            "Smith et al. Nature 2020",
            "Johnson Science 2019",
            "Brown Cell 2021"
        ] * 10  # Process multiple times
        
        for citation in citations:
            result = ref_tractor.link_citation(citation)
    
    finally:
        # Stop profiling
        profiler.disable()
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Print results
        print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
        print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
        
        # Save CPU profile
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions

@profile  # memory_profiler decorator
def memory_profile_example():
    """Example function with memory profiling"""
    ref_tractor = ReferencesTractor()
    
    citations = ["Test citation"] * 100
    results = []
    
    for citation in citations:
        result = ref_tractor.link_citation(citation)
        results.append(result)
    
    return results
```

### Automated Testing Pipeline

```python
# tests/automated/test_pipeline.py
import pytest
import json
from pathlib import Path

class TestAutomatedPipeline:
    """Automated testing for CI/CD pipeline"""
    
    @pytest.fixture
    def sample_citations(self):
        """Load sample citations for testing"""
        return [
            "BERT: Pre-training of Deep Bidirectional Transformers",
            "Attention Is All You Need",
            "Deep Residual Learning for Image Recognition"
        ]
    
    def test_system_integration(self, sample_citations):
        """Test complete system integration"""
        ref_tractor = ReferencesTractor()
        
        results = []
        for citation in sample_citations:
            try:
                result = ref_tractor.link_citation(citation)
                results.append({
                    'citation': citation,
                    'success': bool(result),
                    'has_doi': bool(result.get('doi')) if result else False
                })
            except Exception as e:
                results.append({
                    'citation': citation,
                    'success': False,
                    'error': str(e)
                })
        
        # Save results for CI analysis
        results_file = Path("test_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Assert basic functionality
        success_rate = sum(1 for r in results if r['success']) / len(results)
        assert success_rate > 0.5, f"Success rate {success_rate} below threshold"
    
    @pytest.mark.slow
    def test_performance_regression(self, sample_citations):
        """Test for performance regressions"""
        import time
        
        ref_tractor = ReferencesTractor()
        
        times = []
        for citation in sample_citations:
            start = time.time()
            ref_tractor.link_citation(citation)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Performance assertions
        assert avg_time < 10.0, f"Average time {avg_time:.2f}s exceeds limit"
        assert max_time < 30.0, f"Maximum time {max_time:.2f}s exceeds limit"
        
        # Save performance data
        perf_data = {
            'average_time': avg_time,
            'max_time': max_time,
            'individual_times': times
        }
        
        with open("performance_results.json", 'w') as f:
            json.dump(perf_data, f, indent=2)
```

## Release Process

### Version Management

```bash
# Update version in setup.py and __init__.py
# Follow semantic versioning (MAJOR.MINOR.PATCH)

# Create release branch
git checkout -b release/v1.2.0

# Update CHANGELOG.md
# Update version numbers
# Run full test suite
pytest tests/

# Create release commit
git add .
git commit -m "Release v1.2.0"

# Create tag
git tag -a v1.2.0 -m "Release version 1.2.0"

# Push to GitHub
git push origin release/v1.2.0
git push origin v1.2.0
```

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version numbers updated
- [ ] Performance benchmarks run
- [ ] Security scan completed
- [ ] Release notes prepared

## Getting Help

### Resources

- [API Documentation](api.md)
- [Evaluation Guide](evaluation.md)
- [Development Examples](examples.md)
- [GitHub Discussions](https://github.com/sirisacademic/references-tractor/discussions)

### Common Development Issues

#### Model Loading Problems
```python
# Debug model loading
import torch
from transformers import AutoModel

def debug_model_loading():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        model = AutoModel.from_pretrained("SIRIS-Lab/citation-parser-ENTITY")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
```

#### API Connectivity Issues
```python
# Test API connectivity
import requests

def test_api_connectivity():
    apis = {
        'OpenAlex': 'https://api.openalex.org/works?filter=title.search:test',
        'CrossRef': 'https://api.crossref.org/works?query=test'
    }
    
    for name, url in apis.items():
        try:
            response = requests.get(url, timeout=10)
            print(f"{name}: {'✓' if response.status_code == 200 else '✗'}")
        except Exception as e:
            print(f"{name}: ✗ ({e})")
```

#### Development Environment Issues
```bash
# Clean development environment
pip uninstall references-tractor
pip cache purge
pip install -e .[dev]

# Reset git hooks
pre-commit uninstall
pre-commit install

# Clear pytest cache
pytest --cache-clear
```

Thank you for contributing to References Tractor! Your contributions help make academic citation processing more accessible and reliable for researchers worldwide.
