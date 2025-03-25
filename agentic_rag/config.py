import logging
import os
from typing import Any, Dict, Optional

import lancedb


class Config:
    """Configuration for the Agentic RAG system."""

    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4o",
        provider: str = "openai",
        embedding_model: str = "text-embedding-3-large",
        embedding_dimensions: int = 3072,
        db_path: str = "./lancedb",
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
        temperature: float = 0.1,
        top_k: int = 5,
    ):

        # LLM settings
        self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
        self.model = model
        self.provider = provider
        self.temperature = temperature

        # Embedding settings
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions

        # Vector DB settings
        self.db_path = db_path
        self.top_k = top_k

        # Text processing settings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize clients
        self._init_clients()

    def _init_clients(self):
        """Initialize API clients based on provider."""
        if self.provider.lower() == "openai":
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)
        elif self.provider.lower() == "anthropic":
            from anthropic import Anthropic

            self.client = Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        # Create db_path directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

        # Initialize LanceDB
        self.db = lancedb.connect(self.db_path)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create a Config instance from a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            Config instance
        """
        return cls(
            api_key=config_dict.get("api_key"),
            model=config_dict.get("model", "gpt-4o"),
            provider=config_dict.get("provider", "openai"),
            embedding_model=config_dict.get(
                "embedding_model", "text-embedding-3-large"
            ),
            embedding_dimensions=config_dict.get("embedding_dimensions", 3072),
            db_path=config_dict.get("db_path", "./lancedb"),
            chunk_size=config_dict.get("chunk_size", 1500),
            chunk_overlap=config_dict.get("chunk_overlap", 150),
            temperature=config_dict.get("temperature", 0.1),
            top_k=config_dict.get("top_k", 5),
        )
