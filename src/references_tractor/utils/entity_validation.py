# references_tractor/utils/entity_validation.py
import re
from typing import Dict, List, Optional

class EntityValidator:
    """Utility class for cleaning and validating NER entities"""

    @staticmethod
    def reconstruct_page_range(first_page: str, last_page: str) -> tuple[str, str]:
        """
        Reconstruct abbreviated page ranges by expanding the last page.
        
        Examples:
            4307, 12 → 4307, 4312
            4307, 92 → 4307, 4392
            4307, 123 → 4307, 4123 (invalid - rejected)
            1999, 03 → 1999, 1903 (invalid - rejected)
        
        Args:
            first_page: The starting page number as string
            last_page: The ending page (possibly abbreviated) as string
            
        Returns:
            Tuple of (first_page, expanded_last_page) or original values if no expansion needed/valid
        """
        if not first_page or not last_page:
            return first_page, last_page
        
        # Clean inputs - extract numeric parts only
        first_num = re.search(r'\d+', first_page)
        last_num = re.search(r'\d+', last_page)
        
        if not first_num or not last_num:
            return first_page, last_page
        
        first_digits = first_num.group()
        last_digits = last_num.group()
        
        # Check if expansion is potentially needed
        if len(last_digits) >= len(first_digits):
            # Last page is same length or longer - probably not abbreviated
            return first_page, last_page
        
        # Check if last page is already larger than first (not abbreviated)
        try:
            if int(last_digits) > int(first_digits):
                # Already valid without expansion
                return first_page, last_page
        except ValueError:
            return first_page, last_page
        
        # Attempt expansion by replacing last N digits
        abbreviation_length = len(last_digits)
        
        # Safety check: abbreviation can't be as long as the full first page
        if abbreviation_length >= len(first_digits):
            return first_page, last_page
        
        # Replace the last N digits of first_page with last_digits
        prefix_length = len(first_digits) - abbreviation_length
        prefix = first_digits[:prefix_length]
        expanded_last = prefix + last_digits
        
        # Validate that expanded last page > first page
        try:
            if int(expanded_last) > int(first_digits):
                # Valid expansion
                return first_page, expanded_last
            else:
                # Invalid expansion (would go backwards)
                return first_page, last_page
        except ValueError:
            return first_page, last_page

    @staticmethod
    def validate_and_clean_entities(ner_entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Validate and select the best entity from each NER list.
        First applies general tokenizer cleanup, then field-specific validation.
        Enhanced with page range reconstruction.
        """
        # Step 1: Clean tokenizer artifacts from all fields
        preprocessed_entities = EntityValidator._clean_tokenizer_artifacts(ner_entities)

        # Step 2: Apply field-specific validation to cleaned data
        cleaned_entities = {}

        if 'DOI' in preprocessed_entities and preprocessed_entities['DOI']:
            best_doi = EntityValidator._select_best_doi(preprocessed_entities['DOI'])
            if best_doi:
                cleaned_entities['DOI'] = [best_doi]

        if 'TITLE' in preprocessed_entities and preprocessed_entities['TITLE']:
            best_title = EntityValidator._select_best_title(preprocessed_entities['TITLE'])
            if best_title:
                cleaned_entities['TITLE'] = [best_title]

        if 'AUTHORS' in preprocessed_entities and preprocessed_entities['AUTHORS']:
            best_authors = EntityValidator._select_best_authors(preprocessed_entities['AUTHORS'])
            if best_authors:
                cleaned_entities['AUTHORS'] = [best_authors]

        if 'JOURNAL' in preprocessed_entities and preprocessed_entities['JOURNAL']:
            best_journal = EntityValidator._select_best_journal(preprocessed_entities['JOURNAL'])
            if best_journal:
                cleaned_entities['JOURNAL'] = [best_journal]

        if 'PUBLICATION_YEAR' in preprocessed_entities and preprocessed_entities['PUBLICATION_YEAR']:
            best_year = EntityValidator._select_best_year(preprocessed_entities['PUBLICATION_YEAR'])
            if best_year:
                cleaned_entities['PUBLICATION_YEAR'] = [best_year]

        if 'VOLUME' in preprocessed_entities and preprocessed_entities['VOLUME']:
            best_volume = EntityValidator._select_best_volume(preprocessed_entities['VOLUME'])
            if best_volume:
                cleaned_entities['VOLUME'] = [best_volume]

        if 'ISSUE' in preprocessed_entities and preprocessed_entities['ISSUE']:
            best_issue = EntityValidator._select_best_issue(preprocessed_entities['ISSUE'])
            if best_issue:
                cleaned_entities['ISSUE'] = [best_issue]

        # Handle page range reconstruction
        page_first = None
        page_last = None
        
        if 'PAGE_FIRST' in preprocessed_entities and preprocessed_entities['PAGE_FIRST']:
            page_first = EntityValidator._select_best_page(preprocessed_entities['PAGE_FIRST'])
        
        if 'PAGE_LAST' in preprocessed_entities and preprocessed_entities['PAGE_LAST']:
            page_last = EntityValidator._select_best_page(preprocessed_entities['PAGE_LAST'])
        
        # Apply page range reconstruction if both pages exist
        if page_first and page_last:
            reconstructed_first, reconstructed_last = EntityValidator.reconstruct_page_range(page_first, page_last)
            
            cleaned_entities['PAGE_FIRST'] = [reconstructed_first]
            cleaned_entities['PAGE_LAST'] = [reconstructed_last]
        else:
            # Add individual pages if only one exists
            if page_first:
                cleaned_entities['PAGE_FIRST'] = [page_first]
            if page_last:
                cleaned_entities['PAGE_LAST'] = [page_last]

        return cleaned_entities
    
    @staticmethod
    def  _clean_tokenizer_artifacts(ner_entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Clean tokenizer artifacts (##tokens) from all NER entities.
        Attempts to reconstruct original text from subword tokens.
        """
        cleaned_entities = {}

        for entity_type, entity_list in ner_entities.items():
            if not entity_list:
                continue

            cleaned_list = []

            for entity_text in entity_list:
                # Step 1: Reconstruct from subword tokens
                reconstructed = EntityValidator._reconstruct_from_subwords(entity_text)

                # Step 2: Basic cleanup
                cleaned = EntityValidator._basic_cleanup(reconstructed)

                if cleaned and len(cleaned.strip()) > 0:
                    cleaned_list.append(cleaned)

            if cleaned_list:
                cleaned_entities[entity_type] = cleaned_list

        return cleaned_entities
    
    @staticmethod
    def  _reconstruct_from_subwords(text: str) -> str:
        """
        Reconstruct original text from subword tokenizer artifacts.
        Handles patterns like: "jsr . ##12 ##847" -> "jsr.12847"
        """
        if not text:
            return text

        # Handle ##token patterns (BERT-style subword tokens)
        # ##12 ##847 should become 12847
        reconstructed = re.sub(r'\s*##(\w+)', r'\1', text)

        # Handle spaces around punctuation that got tokenized
        # "jsr . 12847" -> "jsr.12847"
        reconstructed = re.sub(r'\s*\.\s*', '.', reconstructed)
        reconstructed = re.sub(r'\s*/\s*', '/', reconstructed)
        reconstructed = re.sub(r'\s*:\s*', ':', reconstructed)
        reconstructed = re.sub(r'\s*-\s*', '-', reconstructed)

        # Clean up multiple spaces
        reconstructed = re.sub(r'\s+', ' ', reconstructed)

        return reconstructed.strip()

    @staticmethod
    def _basic_cleanup(text: str) -> str:
        """
        Apply basic cleanup common to all fields.
        """
        if not text:
            return text

        # Remove any remaining standalone ## tokens (comprehensive)
        # Matches: ##123, ##abc, ##-ing, ##_test, etc.
        text = re.sub(r'\b##[^\s]*\b\s*', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove leading/trailing punctuation artifacts
        text = re.sub(r'^[.,;:\-\s]+|[.,;:\-\s]+$', '', text)

        return text
    
    @staticmethod
    def   _select_best_doi(doi_list: List[str]) -> Optional[str]:
        """Select the best DOI from a list of candidates"""
        if not doi_list:
            return None

        best_doi = None
        best_score = -1

        for doi_candidate in doi_list:
            score = 0
            cleaned_doi = doi_candidate.strip()

            # Skip obvious noise (remaining tokenizer artifacts)
            if re.match(r'^[#\d]+$', cleaned_doi):
                continue

            # Additional cleanup for any remaining artifacts
            cleaned_doi = re.sub(r'[#\s]+', '', cleaned_doi)

            # Score based on DOI pattern matching
            if re.match(r'^10\.\d{4,}/[^\s]+$', cleaned_doi):
                score = 100  # Perfect DOI format
            elif '10.' in cleaned_doi and '/' in cleaned_doi:
                score = 50   # Looks like a DOI but needs cleaning
                # Try to extract clean DOI
                doi_match = re.search(r'10\.\d{4,}/[^\s]+', cleaned_doi)
                if doi_match:
                    cleaned_doi = doi_match.group()
                    score = 80

            if score > best_score:
                best_score = score
                best_doi = cleaned_doi

        return best_doi if best_score > 0 else None
    
    @staticmethod
    def  _select_best_title(title_list: List[str]) -> Optional[str]:
        """Select the best title from a list of candidates"""
        if not title_list:
            return None

        best_title = None
        best_length = 0

        for title_candidate in title_list:
            cleaned_title = title_candidate.strip()
            cleaned_title = re.sub(r'[\'\"]+', '', cleaned_title)
            cleaned_title = re.sub(r'\s+', ' ', cleaned_title)

            if len(cleaned_title) > best_length and len(cleaned_title) >= 10:
                best_title = cleaned_title
                best_length = len(cleaned_title)

        return best_title

    @staticmethod
    def _select_best_authors(authors_list: List[str]) -> Optional[str]:
        """Select the best authors string from a list of candidates"""
        if not authors_list:
            return None

        best_authors = None
        best_score = 0

        for authors_candidate in authors_list:
            cleaned_authors = authors_candidate.strip()
            cleaned_authors = re.sub(r'\s+', ' ', cleaned_authors)

            score = len(cleaned_authors)

            # Bonus for common author patterns
            if ',' in cleaned_authors:  # Multiple authors
                score += 20
            if re.search(r'[A-Z]{1,3}[,\s]', cleaned_authors):  # Initials pattern
                score += 10

            if score > best_score and len(cleaned_authors) >= 3:
                best_score = score
                best_authors = cleaned_authors

        return best_authors
    
    @staticmethod
    def  _select_best_journal(journal_list: List[str]) -> Optional[str]:
        """Select the best journal name from a list of candidates"""
        if not journal_list:
            return None

        best_journal = None
        best_length = 0

        for journal_candidate in journal_list:
            cleaned_journal = journal_candidate.strip()
            cleaned_journal = re.sub(r'\s+', ' ', cleaned_journal)

            if len(cleaned_journal) > best_length and len(cleaned_journal) >= 3:
                best_journal = cleaned_journal
                best_length = len(cleaned_journal)

        return best_journal

    @staticmethod
    def _select_best_year(year_list: List[str]) -> Optional[str]:
        """Select the best year from a list of candidates"""
        if not year_list:
            return None

        for year_candidate in year_list:
            year_text = year_candidate.strip()

            # Extract 4-digit year
            year_match = re.search(r'(19|20)\d{2}', year_text)
            if year_match:
                year = int(year_match.group())
                # Reasonable year range
                if 1800 <= year <= 2050:
                    return str(year)

        return None
    
    @staticmethod
    def  _select_best_volume(volume_list: List[str]) -> Optional[str]:
        """Select the best volume from a list of candidates"""
        if not volume_list:
            return None

        for volume_candidate in volume_list:
            volume_text = volume_candidate.strip()

            # Extract numeric volume
            volume_match = re.search(r'\d+', volume_text)
            if volume_match:
                return volume_match.group()

        return None

    @staticmethod
    def _select_best_issue(issue_list: List[str]) -> Optional[str]:
        """Select the best issue from a list of candidates"""
        if not issue_list:
            return None

        for issue_candidate in issue_list:
            issue_text = issue_candidate.strip()

            # Extract numeric issue
            issue_match = re.search(r'\d+', issue_text)
            if issue_match:
                return issue_match.group()

        return None
    
    @staticmethod
    def  _select_best_page(page_list: List[str]) -> Optional[str]:
        """Select the best page number from a list of candidates"""
        if not page_list:
            return None

        best_page = None
        best_score = 0

        for page_candidate in page_list:
            page_text = page_candidate.strip()
            score = 0
            result = None

            # Electronic article identifiers get high priority
            e_article_match = re.search(r'e\d+', page_text, re.IGNORECASE)
            if e_article_match:
                result = e_article_match.group()
                score = 100
            else:
                # Regular page numbers
                page_match = re.search(r'\d+', page_text)
                if page_match:
                    result = page_match.group()
                    score = 50

            if score > best_score and result:
                best_score = score
                best_page = result

        return best_page

