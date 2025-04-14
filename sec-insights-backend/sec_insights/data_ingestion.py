import os
import re
from typing import List, Dict, Any, Optional, Tuple
from sec_downloader import Downloader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define target sections (keys must match SEC section identifiers)
TARGET_SECTIONS = {
    "1A": "Risk Factors",
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    # "1": "Business", # Example: Add more if needed
}

class SECExtractorAPI:
    """Wrapper around the sec-api.io API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the SEC API extractor with an API key.
        If no key is provided, it will look for SEC_API_KEY in environment variables.
        """
        self.api_key = api_key or os.environ.get("SEC_API_KEY")
        if not self.api_key:
            logging.warning("No SEC API key provided. Set SEC_API_KEY environment variable or pass it to the constructor.")
        self.base_url = "https://api.sec-api.io/extractor"
    
    def get_section(self, filing_url: str, section_id: str, return_type: str = "text") -> str:
        """
        Extract a section from an SEC filing using the sec-api.io API.
        
        Args:
            filing_url: URL to the SEC filing on sec.gov
            section_id: ID of the section to extract (e.g., "1A" for Risk Factors)
            return_type: 'text' for plain text or 'html' for HTML content
            
        Returns:
            String containing the extracted section content
        """
        if not self.api_key:
            logging.error("Cannot extract section: No API key provided")
            return ""
            
        try:
            params = {
                "url": filing_url,
                "item": section_id,
                "type": return_type,
                "token": self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Error extracting section {section_id} from {filing_url}: {e}")
            return ""

class SECDataIngestion:
    """
    Handles fetching, section extraction, and chunking SEC filings.
    """

    def __init__(self, download_path: str = "./sec_filings_download", email_address: str = "your_email@example.com", sec_api_key: Optional[str] = None):
        """
        Initializes the SECDataIngestion component.
        
        Args:
            download_path: Directory to store downloaded filings
            email_address: Email for SEC Edgar access
            sec_api_key: API key for sec-api.io (optional if set as environment variable)
        """
        self.download_path = os.path.abspath(download_path)
        logging.info(f"Using download path: {self.download_path}")
        try:
            os.makedirs(self.download_path, exist_ok=True)
            logging.info(f"Download directory verified: {self.download_path}")
        except Exception as e:
            logging.error(f"Failed to create/verify download directory: {e}")
            raise

        company_name_for_user_agent = "SEC Insights Engine Bot"
        try:
            # Initialize the SEC Edgar downloader
            self.dl = Downloader(company_name_for_user_agent, email_address)
            logging.info(f"Initialized Downloader for '{company_name_for_user_agent}'")
        except Exception as init_e:
             logging.error(f"Failed to initialize Downloader: {init_e}", exc_info=True)
             raise
             
        # Initialize the SEC API extractor
        self.sec_api = SECExtractorAPI(sec_api_key)
        logging.info("Initialized SEC API extractor")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, length_function=len, add_start_index=True
        )
        logging.info("SECDataIngestion initialized.")

    def download_filings(self, ticker: str, filing_type: str = "10-K", limit: int = 1):
        """
        Downloads the specified number of the latest filings for a ticker.
        """
        try:
            logging.info(
                f"Attempting to download {limit} {filing_type} filings for {ticker} "
                f"to path: {os.path.abspath(self.download_path)} using email: {self.dl.user_agent}"
            )

            original_dir = os.getcwd()
            try:
                os.chdir(self.download_path)
                count = self.dl.get(filing_type, ticker, limit=limit, download_details=True)
                logging.info(
                    f"sec-edgar-downloader reported downloading {count} {filing_type} filings for {ticker}."
                )
            finally:
                os.chdir(original_dir)  # Always restore the original directory

            expected_base_dir = os.path.join(self.download_path, "sec-edgar-filings", ticker.upper(), filing_type)
            if os.path.exists(expected_base_dir):
                logging.info(f"Base directory EXISTS: {expected_base_dir}")
                try:
                    dir_contents = os.listdir(expected_base_dir)
                    logging.info(f"Contents of base directory {expected_base_dir}: {dir_contents}")
                except Exception as list_e:
                    logging.warning(f"Could not list contents of {expected_base_dir}: {list_e}")
            else:
                logging.error(f"Base directory DOES NOT exist after download call: {expected_base_dir}")

            try:
                root_contents = os.listdir(self.download_path)
                logging.info(f"Contents of ROOT download directory ({self.download_path}): {root_contents}")
            except Exception as root_list_e:
                logging.error(f"Failed to list contents of ROOT download directory {self.download_path}: {root_list_e}")

        except Exception as e:
            logging.error(f"Error during download_filings for {ticker} ({filing_type}): {e}", exc_info=True)

    def _find_filing_path(self, ticker: str, filing_type: str, accession_number: str) -> Optional[str]:
        """Finds the path to the primary HTML document of a downloaded filing."""
        base_path = os.path.join(self.download_path, "sec-edgar-filings", ticker.upper(), filing_type)
        doc_path = os.path.join(base_path, accession_number) # Use accession number with hyphens

        if not os.path.exists(doc_path):
             logging.warning(f"Directory not found for {accession_number} (path checked: {doc_path})")
             return None

        primary_doc_name = None
        try:
            for filename in os.listdir(doc_path):
                file_path = os.path.join(doc_path, filename)
                if os.path.isfile(file_path) and filename.lower().endswith((".htm", ".html")):
                    primary_doc_name = filename
                    logging.info(f"Found potential primary document: {primary_doc_name} in {doc_path}")
                    break

            if primary_doc_name:
                return os.path.join(doc_path, primary_doc_name)
            else:
                logging.warning(f"Primary HTML document (.htm/.html) not found in {doc_path}")
                txt_fallback = os.path.join(doc_path, "full-submission.txt")
                if os.path.exists(txt_fallback):
                     logging.warning(f"Found full-submission.txt as fallback: {txt_fallback}")
                     return txt_fallback
                else:
                     logging.error(f"No suitable filing document found in {doc_path}")
                     return None
        except Exception as e:
            logging.error(f"Error searching for document in {doc_path}: {e}")
            return None
            
    def _convert_local_path_to_sec_url(self, filing_path: str, accession_number: str, ticker: str) -> str:
        """
        Converts a local file path to an SEC.gov URL.
        
        This is necessary because the SEC API needs URLs, not local file paths.
        We construct a likely SEC.gov URL based on the accession number and ticker.
        """
        # Extract filename from the path
        filename = os.path.basename(filing_path)
        
        # Create CIK with leading zeros (10 digits)
        cik_match = re.match(r'(\d+)-\d+-\d+', accession_number)
        if cik_match:
            cik = cik_match.group(1).zfill(10)
        else:
            cik = "0000000000"
            logging.warning(f"Could not extract CIK from accession number {accession_number}. Using placeholder.")
        
        # Format the accession number without dashes for the URL
        acc_no_dashes = accession_number.replace('-', '')
        
        # Construct the URL
        sec_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dashes}/{filename}"
        logging.info(f"Constructed SEC URL: {sec_url}")
        
        return sec_url

    def process_downloaded_filing(self, ticker: str, filing_type: str, accession_number: str, company_name: Optional[str] = None, filing_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Processes a single downloaded filing using sec-api to extract sections.
        """
        filing_path = self._find_filing_path(ticker, filing_type, accession_number)
        if not filing_path:
            return []

        logging.info(f"Found filing document at: {filing_path}")
        
        # Convert local path to SEC.gov URL for the API
        sec_url = self._convert_local_path_to_sec_url(filing_path, accession_number, ticker)
        
        # Extract sections using SEC API
        extracted_sections = {}
        for section_id, section_name in TARGET_SECTIONS.items():
            logging.info(f"Extracting section {section_id} ({section_name}) from {sec_url}")
            section_text = self.sec_api.get_section(sec_url, section_id)
            
            if section_text:
                logging.info(f"Successfully extracted section {section_id} ({section_name}). Length: {len(section_text)} chars")
                extracted_sections[section_name] = section_text
            else:
                logging.warning(f"Failed to extract section {section_id} ({section_name}) or section is empty")
        
        # Chunk sections
        all_chunks = []
        if not extracted_sections:
             logging.warning(f"No target sections extracted from filing {accession_number}")
        for section_key, section_content in extracted_sections.items():
            logging.info(f"Chunking section '{section_key}' (Length: {len(section_content)})...")
            chunks = self.text_splitter.split_text(section_content)
            logging.info(f"Created {len(chunks)} chunks for section '{section_key}'.")
            for i, chunk_text in enumerate(chunks):
                 chunk_metadata = {
                     "company": company_name or ticker,
                "ticker": ticker,
                "filing_type": filing_type,
                     "date": filing_date or "Unknown",
                     "accession_number": accession_number,
                     "section": section_key, # Use the proper section name
                     "chunk_index": i,
                 }
                 all_chunks.append({
                     "content": chunk_text,
                     "metadata": chunk_metadata
                 })

        logging.info(f"Total chunks created for {accession_number}: {len(all_chunks)}")
        return all_chunks