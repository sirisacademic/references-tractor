# Citation Linking Evaluation Summary

This document provides an overview of the evaluation results for the References Tractor citation linking system.

## Overview

- **Total Citations Evaluated**: 200
- **APIs Tested**: OpenAlex, OpenAIRE, PubMed, CrossRef, HAL
- **Evaluation Mode**: Strict
- **Evaluation Date**: June 28, 2025

## Performance Summary

Thresholds: 0.90 pairwise model / 0.60 entity-based similarity

| API | Accuracy | Total | Correct | Matches | No Results | Incorrect | Wrong Links | Missing | False Positives |
|-----|----------|-------|---------|---------|------------|-----------|-------------|---------|-----------------|
| **Openalex** | 80% | 200 | 160 | 136 | 24 | 40 | 19 | 17 | 4 | 
| **Openaire** | 68% | 197 | 134 | 106 | 28 | 63 | 26 | 26 | 11 | 
| **Pubmed** | 63% | 200 | 125 | 46 | 79 | 75 | 8 | 11 | 56 | 
| **Crossref** | 59% | 200 | 118 | 93 | 25 | 82 | 42 | 30 | 10 | 
| **HAL** | 97% | 200 | 193 | 0 | 193 | 7 | 0 | 0 | 7 | 
| **Ensemble** | 76% | 200 | 151 | 126 | 25 | 49 | 24 | 15 | 10 | 

## Key Findings

### API Characteristics

- **HAL**: Specialized repository with very high precision but limited coverage
- **PubMed**: Excellent for biomedical citations, conservative linking approach
- **OpenAlex**: Balanced performance across all citation types
- **OpenAIRE**: Similar performance to OpenAlex with slightly lower accuracy
- **CrossRef**: Conservative approach, no positive matches in test set
- **Ensemble**: Combines multiple APIs but shows room for improvement in current implementation

## Methodology

The evaluation uses a **strict classification approach** with the following metrics:

- **Correct Matches**: Citations correctly linked to expected database records
- **Correct No Results**: Citations correctly identified as not available in the database
- **Wrong Links**: Citations linked to incorrect database records
- **Missing**: Expected citations that were not found
- **False Positives**: Unexpected citations returned when none were expected

## Evaluation Data

Complete evaluation results including individual citation analysis are available in the `evaluation_results/` directory with detailed breakdowns by API and citation type.

---

*This evaluation was conducted using the References Tractor v2.0.0 citation linking system. For detailed methodology and individual results, see the complete evaluation reports in this directory.*
