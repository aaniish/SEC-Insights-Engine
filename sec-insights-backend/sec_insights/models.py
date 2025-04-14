from pydantic import BaseModel
from typing import List, Optional, Literal

class Citation(BaseModel):
    company: str
    ticker: str
    filing: str
    section: str
    page: int

class ChatMessage(BaseModel):
    """Represents a single message in the chat history."""
    role: Literal['user', 'assistant']
    content: str

class SecResponse(BaseModel):
    answer: str
    citations: List[Citation]
    suggested_queries: Optional[List[str]] = None