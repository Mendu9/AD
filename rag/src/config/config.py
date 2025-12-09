"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from pathlib import Path
from langchain.chat_models import init_chat_model
import streamlit as st

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""
    
    # API Keys
    #OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # HF_TOKEN=os.getenv("HF_TOKEN")
    # GROQ_API_KEY=os.getenv("GROQ_API_KEY")
    GROQ_API_KEY=st.secrets["GROQ_API_KEY"]
    # Model Configuration
    LLM_MODEL = "openai/gpt-oss-120b"
    USER_AGENT = os.getenv("USER_AGENT", "alzheimer-rag/1.0")
    # Document Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    
    # Default URLs
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = BASE_DIR / "data"
    path_dir = str(DATA_DIR)
    print("Path dir": path_dir)
    
    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")
            
        # GROQ_API_KEY=
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

        return init_chat_model(model=cls.LLM_MODEL, model_provider="groq")


