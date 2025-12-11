# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.1] - 2025-02-12
### Added
- Implemented citation–mention linking system from fulltext.
- Added NER-based extraction of AUTHORS and PUBLICATION_YEAR for references.
- Implemented regex-based mention matching using surname + year.
- Added support for citation ID–based linking when `CITATION_ID` is present.
- Added normalized-author matching (`Muench et al.` → `muench`).

### Changed
- Updated `extract_and_link_from_text()` to also compute NER for references.
- Each reference now contains `claims_in_context` populated via mention–reference matching.
- Improved NER entity grouping logic (merged split tokens like `"20" + "23"` → `"2023"`).
- Improved prescreening & linking decision pipeline.

### Fixed
- Removed obsolete fields before matching when needed.

### Notes
- Matching logic prioritizes CITATION_ID when available.
- Fallback uses (AUTHOR, YEAR) mention-based matching with normalization and fuzzy rules.

## [2.0.0] - 2025-06-23

### Architecture & API Support
- **Enhanced DOI Handling**: Added support for multiple DOIs per paper (conference vs journal versions)
- **Progressive Search Strategy**: Implemented sophisticated field combination strategies and API-specific retry configurations

### Model Architecture  
- **Improved Result Ranking**: Enhanced ranking system that combines SELECT model confidence with NER similarity scoring for better accuracy

### Performance & Scalability
- **Caching System**: Implemented pipeline-level caching to avoid duplicate API calls
- **Rate Limiting**: Added API-specific rate limiting and retry configurations
- **Device Auto-detection**: Automatic detection of best available device (CPU/CUDA/MPS)
- **Enhanced Error Handling**: Robust fallback strategies and comprehensive error handling

### Evaluation & Testing
- **Gold Standard Testing**: Added detailed evaluation against human-validated datasets
- **Performance Metrics**: Comprehensive accuracy metrics, comparison tables, and individual API analysis

### API Integration & Data Processing
- **API-Specific Optimizations**: Specialized handling for each API's response format and requirements
- **Enhanced Field Mapping**: Sophisticated field preprocessing and API-specific transformations
- **Deduplication Logic**: Multiple DOI-aware deduplication across search strategies
- **Result Enhancement**: Enhanced metadata and alternative DOI tracking

### User Experience
- **Better Documentation**: Comprehensive evaluation reports and metrics dashboards
- **Flexible Configuration**: Customizable device selection, cache settings, and API targeting

## [1.0.0] - 2024-11-21

### Added
- Basic structure of the library
- NER model for parsing raw affiliations
- Search via OpenAlex API
- Pairwise model for ranking multiple candidates
- Screening model to ignore strings that are not citations of citable research outputs

## Unreleased

### Added
- Features and improvements for future releases will be listed here
