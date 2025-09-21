# config/settings.py - Simple configuration
import os
from dataclasses import dataclass
from typing import List


@dataclass
class AppSettings:
    """Application settings - simplified configuration"""

    # App info
    app_name: str = "DocuChat"
    app_version: str = "1.0.0"

    # File processing
    max_file_size_mb: int = 200
    supported_extensions: List[str] = None
    chunk_size: int = 500
    chunk_overlap: int = 50

    # LLM settings
    llm_provider: str = "ollama"  # ollama, openai
    llm_model: str = "llama2"
    llm_base_url: str = "http://localhost:11434"
    openai_api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 1000

    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector database
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    collection_name: str = "documents"

    def __post_init__(self):
        """Load from environment variables"""
        if self.supported_extensions is None:
            self.supported_extensions = ['.pdf', '.docx', '.txt', '.xlsx', '.csv']

        # Override with environment variables
        self.llm_provider = os.getenv("LLM_PROVIDER", self.llm_provider)
        self.llm_model = os.getenv("LLM_MODEL", self.llm_model)
        self.llm_base_url = os.getenv("LLM_BASE_URL", self.llm_base_url)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        self.qdrant_url = os.getenv("QDRANT_URL", self.qdrant_url)
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", self.qdrant_api_key)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", self.embedding_model)


# Global settings instance
settings = AppSettings()
