# citation_formatter.py
import xml.etree.ElementTree as ET

# CitationFormatter Base Class
class CitationFormatter:
    def generate_apa_citation(self, data):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _format_doi_section(self, main_doi, alternative_dois=None):
        """
        Format DOI section with main DOI and alternative DOIs
        Format: "DOI: 10.1109/20.179443. Alternative DOIs: xxx, yyy, zzz"
        """
        if not main_doi:
            return ""
        
        # Clean main DOI (remove prefixes)
        clean_main_doi = main_doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
        doi_section = f"DOI: {clean_main_doi}"
        
        # Add alternative DOIs if they exist
        if alternative_dois and len(alternative_dois) > 0:
            # Clean alternative DOIs
            clean_alt_dois = []
            for alt_doi in alternative_dois:
                if alt_doi and alt_doi != main_doi:  # Avoid duplicates
                    clean_alt = alt_doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
                    if clean_alt != clean_main_doi:  # Double-check for duplicates after cleaning
                        clean_alt_dois.append(clean_alt)
            
            if clean_alt_dois:
                alt_dois_str = ", ".join(clean_alt_dois)
                doi_section += f". Alternative DOIs: {alt_dois_str}"
        
        return doi_section
    
    def _get_doi_info(self, data):
        """
        Extract DOI information from enhanced result structure
        Returns: (main_doi, alternative_dois)
        """
        # Try enhanced structure first
        main_doi = data.get('main_doi')
        alternative_dois = data.get('alternative_dois', [])
        
        # Fallback to legacy structure
        if not main_doi:
            main_doi = data.get('doi')
            alternative_dois = []
        
        return main_doi, alternative_dois
    
class OpenAlexFormatter(CitationFormatter):
    def generate_apa_citation(self, data):
        
        authors_list = [auth['raw_author_name'] for auth in data.get('authorships', [])]
        if len(authors_list) > 3:
            authors = ", ".join(authors_list[:3]) + ", et al."
        else:
            authors = ", ".join(authors_list)

        title = data.get('title', "Unknown Title")
        year = data.get('publication_year', "n.d.")

        try:
            journal = data.get('primary_location', {}).get('source', {}).get('display_name', None)
        except AttributeError:
            journal = None

        try:
            volume = data.get('biblio', {}).get('volume', None)
        except AttributeError:
            volume = None

        try:
            issue = data.get('biblio', {}).get('issue', None)
        except AttributeError:
            issue = None

        try:
            first_page = data.get('biblio', {}).get('first_page', '')
            last_page = data.get('biblio', {}).get('last_page', '')
            pages = f"{first_page}-{last_page}".strip("-")
        except AttributeError:
            pages = ""

        # Get DOI information (main + alternatives)
        main_doi, alternative_dois = self._get_doi_info(data)
        doi_section = self._format_doi_section(main_doi, alternative_dois)

        citation_parts = [
            f"{authors}" if authors else "",
            f"({year})." if authors else ".",
            f"{title}." if title else "",
            f"{journal}," if journal else "",
            f"{volume}" if volume else "",
            f"({issue})" if issue else "",
            f"{pages}." if pages else "",
            doi_section
        ]

        return " ".join(part for part in citation_parts if part).strip()

class OpenAIREFormatter(CitationFormatter):
    def generate_apa_citation(self, data):
        """Generate APA-style citation from OpenAIRE Graph API data with NULL SAFE handling"""
        
        # Extract authors - NULL SAFE
        authors_list = []
        authors_data = data.get('authors') or []  # Handle both missing and null
        if isinstance(authors_data, list):  # Safety check
            for author in authors_data:
                if isinstance(author, dict):
                    full_name = author.get('fullName', '')
                    if full_name:
                        authors_list.append(full_name)
        
        # Format authors
        if len(authors_list) > 3:
            authors = ", ".join(authors_list[:3]) + ", et al."
        else:
            authors = ", ".join(authors_list)
        
        # Extract other fields
        title = data.get('mainTitle', "Unknown Title")
        publication_date = data.get('publicationDate', '')
        year = publication_date[:4] if publication_date else "n.d."
        
        # Extract journal info - NULL SAFE
        container = data.get('container')  # Don't use default {}, check for None
        journal = None
        volume = None
        pages = ""
        
        if isinstance(container, dict):  # Only process if container is actually a dict
            journal = container.get('name')
            volume = container.get('vol')
            sp = container.get('sp', '')
            ep = container.get('ep', '')
            pages = f"{sp}-{ep}".strip('-')
        
        # Get DOI information (main + alternatives) - Enhanced for OpenAIRE
        main_doi, alternative_dois = self._get_doi_info(data)
        
        # If no enhanced structure, extract from pids manually - NULL SAFE
        if not main_doi:
            pids = data.get('pids') or []  # Handle both missing and null
            extracted_dois = []
            if isinstance(pids, list):  # Safety check
                for pid in pids:
                    if isinstance(pid, dict) and pid.get('scheme') == 'doi':
                        doi_value = pid.get('value', '')
                        if doi_value:
                            extracted_dois.append(doi_value)
            
            if extracted_dois:
                main_doi = extracted_dois[0]
                alternative_dois = extracted_dois[1:] if len(extracted_dois) > 1 else []
        
        doi_section = self._format_doi_section(main_doi, alternative_dois)
        
        # Construct citation
        citation_parts = [
            f"{authors}" if authors else "",
            f"({year})." if year else ".",
            f"{title}." if title else "",
            f"{journal}," if journal else "",
            f"{volume}" if volume else "",
            f"{pages}." if pages else "",
            doi_section
        ]
        
        return " ".join(part for part in citation_parts if part).strip()
    
class PubMedFormatter(CitationFormatter):
    def generate_apa_citation(self, data):
        """Generate APA citation from PubMed parsed dict structure (or fallback to XML)"""
        
        # Handle both parsed dict structure and XML fallback
        if isinstance(data, dict) and not data.get('xml_content'):
            # Use parsed dict structure (preferred)
            return self._generate_from_dict(data)
        elif isinstance(data, dict) and data.get('xml_content'):
            # Try dict first, fallback to XML
            try:
                return self._generate_from_dict(data)
            except:
                return self._generate_from_xml(data['xml_content'])
        elif isinstance(data, str):
            # Raw XML string (legacy support)
            return self._generate_from_xml(data)
        else:
            return "Unknown PubMed citation format"
    
    def _generate_from_dict(self, data):
        """Generate citation from parsed dict structure"""
        
        # Extract authors from parsed structure
        authors_list = data.get('authors', [])
        authors = []
        for author in authors_list:
            if isinstance(author, dict):
                authors.append(author.get('full_name', ''))
            else:
                authors.append(str(author))
        
        if len(authors) > 3:
            authors_str = ", ".join(authors[:3]) + ", et al."
        else:
            authors_str = ", ".join(authors)
        
        # Extract other fields
        title = data.get('title', 'Unknown Title')
        year = data.get('publication_year', 'n.d.')
        journal = data.get('journal', '')
        volume = data.get('volume', '')
        issue = data.get('issue', '')
        first_page = data.get('first_page', '')
        last_page = data.get('last_page', '')
        
        # Format pages
        pages = f"{first_page}-{last_page}".strip("-")
        
        # Get DOI information (main + alternatives)
        main_doi, alternative_dois = self._get_doi_info(data)
        doi_section = self._format_doi_section(main_doi, alternative_dois)
        
        # Construct citation
        citation_parts = [
            f"{authors_str} ({year})." if authors_str else f"({year}).",
            f"{title}." if title else "",
            f"{journal}," if journal else "",
            f"{volume}({issue})" if volume or issue else "",
            f"{pages}." if pages else "",
            doi_section
        ]
        
        return " ".join(part for part in citation_parts if part).strip()
    
    def _generate_from_xml(self, xml_data):
        """Generate citation from raw XML (fallback method)"""
        
        # Parse the XML data
        root = ET.fromstring(xml_data)
        article = root.find("PubmedArticle/MedlineCitation/Article")
        
        if article is None:
            return "Unable to parse PubMed XML"
        
        # Extract authors
        authors = []
        author_list = article.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                last_name = author.findtext("LastName", "")
                fore_name = author.findtext("ForeName", "")
                initials = author.findtext("Initials", "")
                authors.append(f"{last_name}, {initials or fore_name[0] if fore_name else ''}.")

        if len(authors) > 3:
            authors = ", ".join(authors[:3]) + ", et al."
        else:
            authors = ", ".join(authors)

        # Extract title
        title = article.findtext("ArticleTitle", "Unknown Title")

        # Extract year
        pub_date = article.find("Journal/JournalIssue/PubDate")
        year = pub_date.findtext("Year", "n.d.") if pub_date is not None else "n.d."

        # Extract journal
        journal = article.findtext("Journal/Title", None)

        # Extract volume, issue, and pages
        volume = article.findtext("Journal/JournalIssue/Volume", None)
        issue = article.findtext("Journal/JournalIssue/Issue", None)
        start_page = article.findtext("Pagination/StartPage", "")
        end_page = article.findtext("Pagination/EndPage", "")
        pages = f"{start_page}-{end_page}".strip("-")

        # Extract DOI - Enhanced for multiple DOIs
        main_doi, alternative_dois = self._get_doi_info({'xml_content': xml_data})
        
        # If no enhanced structure, extract from XML
        if not main_doi:
            main_doi = article.findtext("ELocationID[@EIdType='doi']", None)
            alternative_dois = []

        doi_section = self._format_doi_section(main_doi, alternative_dois)

        citation_parts = [
            f"{authors} ({year}).",
            f"{title}.",
            f"{journal}," if journal else "",
            f"{volume}({issue})" if volume or issue else "",
            f"{pages}." if pages else "",
            doi_section
        ]
        return " ".join(part for part in citation_parts if part)

class CrossrefFormatter(CitationFormatter):
    def generate_apa_citation(self, data):
        # Extracting authors
        authors_list = [f"{auth.get('given', '')} {auth.get('family', '')}".strip() for auth in data.get('author', [])]

        if len(authors_list) > 3:
            authors = ", ".join(authors_list[:3]) + ", et al."
        else:
            authors = ", ".join(authors_list)

        # Title of the paper
        title = data.get('title', ["Unknown Title"])[0]

        # Publication year
        year = data.get('issued', {}).get('date-parts', [[None]])[0][0] or "n.d."

        # Publisher info
        publisher = data.get('publisher', "")
        publisher_location = data.get('container-title', "Unknown Location")[0] or ''

        # Pages range (if available)
        pages = data.get('page', "")
        pages_range = f"{pages}" if pages else ""

        # Get DOI information (main + alternatives)
        main_doi, alternative_dois = self._get_doi_info(data)
        
        # If no enhanced structure, extract from CrossRef DOI field
        if not main_doi:
            main_doi = data.get('DOI', None)
            alternative_dois = []

        doi_section = self._format_doi_section(main_doi, alternative_dois)

        # Format the citation in APA style
        citation_parts = [
            f"{authors} ({year}).",
            f"{title}.",
            f"{publisher_location}: {publisher}.",
            f"{pages_range}." if pages_range else "",
            doi_section
        ]
        return " ".join(part for part in citation_parts if part)

class HALFormatter(CitationFormatter):
    def generate_apa_citation(self, data):
        # Extract authors
        author_name = data.get('authFullName_s', "")
        authors_list = author_name
        if len(authors_list) > 3:
            authors = ", ".join(authors_list[:3]) + ", et al."
        else:
            authors = ", ".join(authors_list)

        # Extract title
        title = data.get('title_s', [""])[0]

        # Extract year
        year = data.get('conferenceStartDateY_i', data.get('publicationStartDateY_i', data.get('publicationDateY_i', "")))

        # Extract book/journal title
        book_title = data.get('journalTitle_s', data.get('bookTitle_s', None))

        # Extract pages
        pages = data.get('page_s', None)
        volume = data.get('volume_s', None)
        issue = data.get('issue_s', [None])[0]

        # Get DOI information (main + alternatives)
        main_doi, alternative_dois = self._get_doi_info(data)
        
        # If no enhanced structure, extract from HAL doiId_s field
        if not main_doi:
            main_doi = data.get('doiId_s', None)
            alternative_dois = []

        doi_section = self._format_doi_section(main_doi, alternative_dois)

        # Construct citation
        citation_parts = [
            f"{authors}" if authors else "",
            f"({year})." if year else ".",
            f"{title}." if title else "",
            f"{book_title}," if book_title else "",
            f"{volume}" if volume else "",
            f"({issue})" if issue else "",
            f"pp. {pages}." if pages else "",
            doi_section
        ]

        # Return formatted citation
        return " ".join(part for part in citation_parts if part).strip()


class CitationFormatterFactory:
    formatters = {
        "openalex": OpenAlexFormatter(),
        "openaire": OpenAIREFormatter(),
        "pubmed": PubMedFormatter(),
        "crossref": CrossrefFormatter(),
        "hal": HALFormatter(),
    }

    @staticmethod
    def get_formatter(api_name):
        if api_name not in CitationFormatterFactory.formatters:
            raise ValueError(f"Unsupported API: {api_name}")
        return CitationFormatterFactory.formatters[api_name]
