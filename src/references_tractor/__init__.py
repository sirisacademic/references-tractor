# references_tractor/__init__.py
"""
References Tractor - Citation Processing and Linking Tools
"""

__version__ = "1.0.0"
__author__ = "SIRIS Lab"
__email__ = "info@sirisacademic.com"

# Robust import handling for different installation methods
def _get_main_class():
    """Import main class with fallback handling"""
    try:
        # Try absolute import first
        from references_tractor.core import ReferencesTractor
        return ReferencesTractor
    except ImportError:
        # Fallback to relative import
        try:
            from .core import ReferencesTractor
            return ReferencesTractor
        except ImportError as e:
            raise ImportError(f"Could not import ReferencesTractor. Please ensure the package is properly installed. Error: {e}")

# Make ReferencesTractor available at package level
ReferencesTractor = _get_main_class()

# Also import key utilities for convenience
def _get_utilities():
    """Import utilities with fallback handling"""
    try:
        from references_tractor.utils import CitationEvaluator, EntityValidator
        return CitationEvaluator, EntityValidator
    except ImportError:
        try:
            from .utils import CitationEvaluator, EntityValidator
            return CitationEvaluator, EntityValidator
        except ImportError:
            # Don't fail if utilities can't be imported
            return None, None

CitationEvaluator, EntityValidator = _get_utilities()

# Define what gets imported with "from references_tractor import *"
__all__ = ['ReferencesTractor']
if CitationEvaluator:
    __all__.extend(['CitationEvaluator', 'EntityValidator'])

# Package metadata
__title__ = "references-tractor"
__description__ = "Tools for processing raw citations and linking them to scholarly knowledge graphs"
__url__ = "https://github.com/sirisacademic/references-tractor"
__license__ = "Apache-2.0"
