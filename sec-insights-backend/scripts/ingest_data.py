import asyncio
import os
import sys
import logging
import json
import re
from sec_edgar_downloader import Downloader
from typing import List, Dict

# Ensure the parent directory (sec-insights-backend) is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from sec_insights.data_ingestion import SECDataIngestion
from sec_insights.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Get email from environment variable or use a placeholder
# IMPORTANT: Replace with your actual email for SEC EDGAR
EMAIL_ADDRESS = os.getenv("SEC_EMAIL_ADDRESS", "your_email@example.com")
DOWNLOAD_PATH = os.path.join(parent_dir, "sec_filings_download") # Store downloads outside sec_insights
DB_PATH = os.path.join(parent_dir, "chroma_db") # Store ChromaDB data in backend root

# List of companies (ticker, name) to ingest
# Expand this list based on the requirement of 20-50 companies
COMPANIES_TO_INGEST = [
    {"ticker": "AAPL", "name": "Apple Inc."},
    {"ticker": "MSFT", "name": "Microsoft Corporation"},
    {"ticker": "AMZN", "name": "Amazon.com Inc."},
    {"ticker": "GOOGL", "name": "Alphabet Inc."},
    {"ticker": "META", "name": "Meta Platforms Inc."}
    # Add more companies (e.g., TSLA, NVDA, JPM, WMT, etc.)
]

FILING_TYPES = ["10-K"] # Add "10-Q" if needed
LIMIT_PER_FILING_TYPE = 2 # Number of recent filings to download per type

# Modify function to list accession number directories instead of reading JSON
async def get_filing_metadata(download_dir: str, ticker: str, filing_type: str) -> List[Dict[str, str]]:
    """Gets accession numbers by listing directories within the base filing type folder."""
    metadata_list = []
    # Change the expected base directory to match where files are actually saved
    expected_base_dir = os.path.join("/app", "sec-edgar-filings", ticker.upper(), filing_type)
    logging.info(f"Looking for accession number directories inside: {expected_base_dir}")

    if os.path.exists(expected_base_dir):
        try:
            # List subdirectories, which should be the accession numbers
            accession_dirs = [d for d in os.listdir(expected_base_dir) if os.path.isdir(os.path.join(expected_base_dir, d))]
            logging.info(f"Found accession number directories: {accession_dirs}")
            for acc_num_dir in accession_dirs:
                 # Check if it looks like an accession number (optional sanity check)
                 if re.match(r"\d{10}-\d{2}-\d{6}", acc_num_dir):
                     metadata_list.append({
                         "accession_number": acc_num_dir,
                         "filing_date": "Unknown" # Date is not available from dir name
                     })
                 else:
                      logging.warning(f"Skipping directory that doesn't look like accession number: {acc_num_dir}")
        except Exception as e:
            logging.error(f"Error listing directories in {expected_base_dir}: {e}")
    else:
        logging.warning(f"Base directory not found: {expected_base_dir}")

    # Return based on limit, even if date is unknown
    return metadata_list[:LIMIT_PER_FILING_TYPE]

async def main():
    logging.info("--- Starting SEC Data Ingestion Process ---")

    if EMAIL_ADDRESS == "your_email@example.com":
        logging.warning("Using placeholder email address. Set SEC_EMAIL_ADDRESS environment variable.")

    # Initialize components
    ingestion = SECDataIngestion(download_path=DOWNLOAD_PATH, email_address=EMAIL_ADDRESS)
    vector_db = VectorStore(persist_directory=DB_PATH)
    total_chunks_added = 0

    for company in COMPANIES_TO_INGEST:
        ticker = company["ticker"]
        company_name = company["name"]
        logging.info(f"--- Processing Company: {company_name} ({ticker}) ---")

        for filing_type in FILING_TYPES:
            logging.info(f"Downloading {filing_type} for {ticker}...")
            ingestion.download_filings(ticker, filing_type=filing_type, limit=LIMIT_PER_FILING_TYPE)

            # Get accession numbers by listing directories
            logging.info(f"Getting accession numbers for downloaded {filing_type} of {ticker}...")
            filing_metadatas = await get_filing_metadata(ingestion.download_path, ticker, filing_type)

            if not filing_metadatas:
                 logging.warning(f"No accession numbers found for {ticker} {filing_type}. Skipping processing.")
                 continue

            # Process filings using the found accession numbers
            for metadata in filing_metadatas:
                accession_number = metadata.get("accession_number")
                filing_date = metadata.get("filing_date") # Will be "Unknown"
                if not accession_number:
                    logging.warning(f"Skipping metadata entry with missing accession number for {ticker}")
                    continue

                logging.info(f"Processing {filing_type} {accession_number} (Date: {filing_date})...")
                chunks = ingestion.process_downloaded_filing(
                    ticker=ticker,
                    filing_type=filing_type,
                    accession_number=accession_number,
                    company_name=company_name,
                    filing_date=filing_date
                )

                if chunks:
                    logging.info(f"Adding {len(chunks)} chunks to vector store...")
                    await vector_db.add_documents(chunks)
                    total_chunks_added += len(chunks)
                else:
                    logging.warning(f"No chunks generated for {ticker} {filing_type} {accession_number}.")

    logging.info(f"--- Ingestion Process Complete --- Total Chunks Added: {total_chunks_added} ---")

if __name__ == "__main__":
    # Ensure asyncio event loop is handled correctly
    asyncio.run(main()) 