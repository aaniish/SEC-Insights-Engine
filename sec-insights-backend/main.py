from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from contextlib import asynccontextmanager
import asyncio
import json

from sec_insights.rag_pipeline import RAGPipeline
from sec_insights.models import SecResponse, ChatMessage

# Global variable for the RAG pipeline instance (will be initialized during startup)
rag_pipeline_instance: Optional[RAGPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    global rag_pipeline_instance
    print("Starting up SEC Insights Engine API...")
    # Initialize the RAG Pipeline - potentially load models, connect to DB etc.
    # Consider adding configuration loading here (e.g., from .env)
    rag_pipeline_instance = RAGPipeline()
    print("RAG Pipeline initialized.")
    yield
    # Clean up resources on shutdown
    print("Shutting down SEC Insights Engine API...")
    # Add any cleanup logic here if needed
    rag_pipeline_instance = None # Or call a cleanup method if RAGPipeline has one

app = FastAPI(
    title="SEC Insights Engine API",
    description="API for querying and analyzing SEC filings using RAG and Agentic Workflows.",
    version="0.1.0",
    lifespan=lifespan # Use the lifespan context manager
)

class QueryRequest(BaseModel):
    """Request model for the query endpoint."""
    query: str
    companies: List[str] = []
    chat_history: Optional[List[ChatMessage]] = None # Use ChatMessage model
    stream: bool = False  # Option to stream responses for long-running queries

async def process_with_timeout(request: QueryRequest, timeout: int = 120):
    """Process the query with a specified timeout to prevent hanging."""
    try:
        # Create a task with timeout
        return await asyncio.wait_for(
            rag_pipeline_instance.aprocess_query(
                query=request.query,
                companies=request.companies,
                chat_history=request.chat_history
            ),
            timeout=timeout  # seconds
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,  # Gateway Timeout
            detail="Request processing took too long. Try a simpler query or fewer companies."
        )

@app.post("/api/query", response_model=SecResponse, summary="Process a natural language query")
async def process_query(request: QueryRequest):
    """
    Accepts a natural language query about SEC filings and returns a synthesized answer.
    
    Handles optional filtering by company and maintains conversational context.
    For complex queries, consider using stream=true to receive responses as they're generated.
    """
    if rag_pipeline_instance is None:
        # This should ideally not happen if lifespan management is correct
        raise HTTPException(status_code=503, detail="Service Unavailable: RAG Pipeline not initialized.")

    try:
        # For streaming responses (complex agent queries)
        if request.stream:
            # This would require the RAGPipeline to support streaming, which is not implemented yet
            # For now, just use an extended timeout
            response = await process_with_timeout(request, timeout=180)  # 3 minutes
            return response
        else:
            # Standard processing with a reasonable timeout
            response = await process_with_timeout(request, timeout=60)  # 1 minute
            return response
    except ValueError as ve: # Example: Catch specific, expected errors
        raise HTTPException(status_code=400, detail=f"Invalid input: {ve}")
    except Exception as e:
        # Log the exception details for debugging
        print(f"Error processing query: {e}") # Replace with proper logging
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

class CompanyInfo(BaseModel):
    """Model for company information."""
    ticker: str
    name: str

class CompaniesResponse(BaseModel):
    """Response model for the companies endpoint."""
    companies: List[CompanyInfo]

@app.get("/api/companies", response_model=CompaniesResponse, summary="Get list of tracked companies")
async def get_companies():
    """
    Retrieves the list of companies available for querying.

    (Currently static, should fetch dynamically in a full implementation).
    """
    # TODO: Fetch this list dynamically from the data ingestion source/database
    static_companies = [
        {"ticker": "AAPL", "name": "Apple Inc."},
        {"ticker": "MSFT", "name": "Microsoft Corporation"},
        {"ticker": "AMZN", "name": "Amazon.com Inc."},
        {"ticker": "GOOGL", "name": "Alphabet Inc."},
        {"ticker": "META", "name": "Meta Platforms Inc."}
    ]
    return CompaniesResponse(companies=static_companies)

if __name__ == "__main__":
    # Consider environment variables for host/port/reload
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)