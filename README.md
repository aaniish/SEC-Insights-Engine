# SEC Insights Engine

An AI-powered answer engine for extracting insights from SEC filings.

## Project Overview

SEC Insights Engine is a sophisticated Retrieval Augmented Generation (RAG) application that allows users to query financial information from SEC filings. This specialized domain-focused approach provides valuable insights for investors, analysts, and anyone interested in public company data.

**Note:** For this demonstration, the application intentionally includes only a limited set of companies (AAPL, MSFT, AMZN, GOOGL, META). This focused approach allows for faster data ingestion and setup while showcasing the core functionality. The system is designed to be easily expanded to include additional companies.

### Key Features

- **Interactive Chat Interface** for natural language queries about SEC filings
- **RAG Architecture** with semantic search and contextual retrieval
- **Company Selection** to focus queries on specific organizations
- **10-K and 10-Q Report Analysis** for comprehensive financial insights
- **Personalization** with dark/light mode themes
- **Multi-turn Conversations** with contextual follow-ups
- **Suggested Queries** based on conversation context
- **Citations** linking back to source documents and sections
- **Responsive Design** with custom black/grey and orange theme

## Architecture

```
                                  ┌───────────────────────────────────────────┐
                                  │                                           │
                                  │         SEC Insights Engine Frontend      │
                                  │         (Next.js Application)             │
                                  │                                           │
                                  └───────────────┬───────────────────────────┘
                                                  │
                                                  │ HTTP/REST API
                                                  │
                                  ┌───────────────▼───────────────────────────┐
                                  │                                           │
                                  │         SEC Insights Backend              │
                                  │         (FastAPI Application)             │
                                  │                                           │
                                  └───┬────────────────┬────────────┬─────────┘
                                      │                │            │
              ┌─────────────────────┐ │                │            │ ┌──────────────────────┐
              │                     ◄─┘                │            └─►                      │
              │  Vector Store       │                  │              │  SEC Agent           │
              │  (ChromaDB)         │                  │              │  (LangChain Agent)   │
              │                     │                  │              │                      │
              └─────────────────────┘                  │              └──────────────────────┘
                                                       │
                                      ┌────────────────▼───────────────┐
                                      │                                │
                                      │  RAG Pipeline                  │
                                      │                                │
                                      └────────────────┬───────────────┘
                                                       │
┌────────────────────────────┐         ┌──────────────▼────────────────┐
│                            │         │                               │
│  SEC Data Ingestion        ├────────►│  SEC Documents                │
│  (ETL Process)             │         │  (10-K & 10-Q Filings)        │
│                            │         │                               │
└────────────────────────────┘         └───────────────────────────────┘
```

The application follows a modern architecture with three main components:

1. **Frontend (Next.js + TypeScript + Tailwind CSS)**
   - React components for UI elements
   - TypeScript for type safety
   - Tailwind CSS for styling
   - Context providers for state management

2. **Backend (Python + FastAPI)**
   - FastAPI server for API endpoints
   - RAG implementation for document retrieval and processing
   - LLM integration for generating responses
   - SEC API integration for data sourcing

3. **Vector Database**
   - ChromaDB for vector storage
   - Document embeddings for semantic search
   - Metadata storage for document context

### Key Components

#### Frontend Layer
- **SEC Insights Engine Frontend**: Single-page application built with Next.js
  - `Chat`: Main interface for user interactions
  - `ChatMessage`: Renders chat messages with citations
  - `CompanySelector`: Allows users to select companies for analysis
  - `Sidebar`: Navigation and layout structure

#### Backend Layer
- **SEC Insights Backend**: REST API service using FastAPI
  - Exposes endpoints for queries and company listings
  - Manages connections to vector database and AI services

- **RAG Pipeline**: Core retrieval-augmented generation pipeline
  - Retrieves relevant document chunks based on queries
  - Generates answers with citations
  - Provides suggested follow-up queries based on context

- **Vector Store**: ChromaDB implementation for semantic search
  - Stores document embeddings and metadata
  - Enables efficient similarity search for document retrieval

- **SEC Agent**: LangChain-based agent for complex queries
  - Implements specialized tools for financial document analysis:
    - SECSectionRetriever: Fetches relevant SEC filing sections
    - LLMSummarizer: Summarizes retrieved text
    - LLMComparer: Compares information between companies
    - FinancialDataExtractor: Extracts financial metrics
    - TrendCalculator: Analyzes trend data

#### Data Layer
- **SEC Data Ingestion**: ETL process to download and process SEC filings
  - Downloads filings via SEC Edgar API
  - Extracts relevant sections (Risk Factors, MD&A)
  - Chunks and processes text for vector storage

- **SEC Documents**: Raw and processed SEC filings
  - Stored in persistent volumes
  - Organized by company, filing type, and accession number

## Setup

### Prerequisites

- Node.js 18+
- Docker and Docker Compose
- Git
- OpenAI API key
- SEC API credentials

### Environment Setup

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd sec-insights-engine
   ```

2. Copy the example environment file and add your API keys:
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file and fill in your:
   - OpenAI API key
   - SEC API credentials (email and API key)
   - Other required API keys

3. Start the application with Docker Compose:
   ```bash
   docker-compose up --build
   ```

4. Open your browser to [http://localhost:3000](http://localhost:3000)

### Data Ingestion

Before you can use the application effectively, you need to ingest SEC filings data into the vector database:

1. Make sure your containers are running:
   ```bash
   docker-compose up -d
   ```

2. Run the ingestion script:
   ```bash
   docker-compose exec backend python scripts/ingest_data.py
   ```

3. This script will:
   - Download 10-K filings for the configured companies
   - Extract relevant sections (Risk Factors, MD&A, etc.)
   - Process and chunk the text
   - Add the chunks to the vector database

4. By default, the script processes a limited set of companies (AAPL, MSFT, AMZN, GOOGL, META) for demonstration purposes. To add more companies:
   - Edit `sec-insights-backend/scripts/ingest_data.py`
   - Modify the `COMPANIES_TO_INGEST` list
   - Adjust the `FILING_TYPES` and `LIMIT_PER_FILING_TYPE` settings if needed

5. Wait for the ingestion process to complete. This may take several minutes depending on the number of filings being processed.

## Demo Video

A demonstration of the SEC Insights Engine is available here:

[![SEC Insights Engine Demo](https://img.youtube.com/vi/KYFof-RJxCQ/0.jpg)](https://youtu.be/KYFof-RJxCQ)

The demo showcases:
- Application overview and UI walkthrough
- Company selection and theme customization
- Example queries and responses
- Multi-turn conversation capability
- How citations and suggested queries work

## Project Structure

- `sec-insights-engine/`: Frontend Next.js application
  - `app/`: Next.js app router files
  - `components/`: React components including chat interface
  - `lib/`: Utility functions and type definitions
  
- `sec-insights-backend/`: Python backend for SEC data processing
  - `main.py`: FastAPI application entry point
  - `scripts/`: Data ingestion scripts
  - `sec_insights/`: Core modules including RAG pipeline
    - `agent.py`: Implementation of SEC-specific agent
    - `data_ingestion.py`: SEC data fetching and processing
    - `rag_pipeline.py`: RAG implementation
    - `vector_store.py`: Vector database interface

- `docker-compose.yml`: Docker configuration for the full stack

## Implementation Notes

### Third-Party Dependencies

- **SEC Edgar API**: Used for retrieving SEC filings
- **OpenAI API**: Powers the LLM for response generation
- **ChromaDB**: Vector database for storing embeddings
- **LangChain**: Tools for RAG implementation

### Limitations and Future Improvements

- Currently supports a limited set of companies by design; the architecture is built to be easily expanded to include more companies
- Could be enhanced with visualizations for financial data
- Potential for more detailed analysis of financial metrics
- Could integrate additional data sources beyond SEC filings
