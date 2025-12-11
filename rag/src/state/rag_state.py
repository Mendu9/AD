"""RAG state definition for LangGraph"""

from typing import List, Dict, Any
from pydantic import BaseModel,Field
from dataclasses import dataclass, field
from langchain.schema import Document

class RAGState(BaseModel):
    """State object for RAG workflow"""
    
    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""
    tool_sources: List[Dict[str,Any]] = Field(default_factory=list)