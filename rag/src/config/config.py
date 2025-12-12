"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""
    HF_TOKEN=os.getenv("HF_TOKEN")
    GROQ_API_KEY=os.getenv("GROQ_API_KEY")
    LLM_MODEL = "openai/gpt-oss-120b"

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100

    path_dir = ["rag/data"]
    
    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")
        os.environ["GROQ_API_KEY"] = cls.GROQ_API_KEY
        return init_chat_model(model=cls.LLM_MODEL, model_provider="groq")