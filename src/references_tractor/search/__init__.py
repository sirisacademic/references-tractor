# Import all the main classes and functions from search modules
from .api_capabilities import APICapabilities
from .field_mapper import FieldMapper, DOIResult
from .search_api import SearchAPI
from .progressive_search import SearchOrchestrator, ProgressiveSearchStrategy, ResultDeduplicator
from .citation_formatter import CitationFormatterFactory

# Export everything that other modules might need
__all__ = [
    'APICapabilities',
    'FieldMapper', 
    'DOIResult',
    'SearchAPI',
    'SearchOrchestrator',
    'ProgressiveSearchStrategy', 
    'ResultDeduplicator',
    'CitationFormatterFactory'
]

