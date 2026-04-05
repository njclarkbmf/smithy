import logging
import os
from typing import Any, Dict, Optional

import lancedb

from .providers.model_router import ModelRouter


class Config:
    """Configuration for the Agentic RAG system."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        provider: str = "openai",
        embedding_model: str = "text-embedding-3-large",
        embedding_dimensions: int = 3072,
        db_path: str = "./lancedb",
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
        temperature: float = 0.1,
        top_k: int = 5,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        # Qwen / DashScope
        dashscope_base_url: Optional[str] = None,
        use_dashscope: bool = False,
        # Separate embedding provider
        embedding_provider: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_use_dashscope: bool = False,
    ):
        # LLM settings
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # Embedding settings
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.embedding_provider = embedding_provider or provider

        # Qwen / DashScope settings
        self.dashscope_base_url = dashscope_base_url
        self.use_dashscope = use_dashscope
        self.embedding_base_url = embedding_base_url
        self.embedding_use_dashscope = embedding_use_dashscope
        self.embedding_api_key = embedding_api_key or api_key

        # Vector DB settings
        self.db_path = db_path
        self.top_k = top_k

        # Text processing settings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize clients
        self._init_clients()

    def _init_clients(self):
        """Initialize API clients based on provider using the ModelRouter."""
        # Resolve API key from environment if not provided
        if not self.api_key:
            env_key_map = {
                "openai": os.environ.get("OPENAI_API_KEY"),
                "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
                "qwen": os.environ.get("DASHSCOPE_API_KEY"),
            }
            self.api_key = env_key_map.get(self.provider.lower())

        # Create the ModelRouter (unified provider abstraction)
        self.client = ModelRouter(
            provider_name=self.provider,
            api_key=self.api_key,
            model=self.model,
            embedding_model=self.embedding_model,
            base_url=self.dashscope_base_url if self.provider == "qwen" else None,
            use_dashscope=self.use_dashscope,
            embedding_api_key=self.embedding_api_key,
            embedding_base_url=self.embedding_base_url,
            embedding_use_dashscope=self.embedding_use_dashscope,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        # Create db_path directory if it doesn't exist
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

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
            top_p=config_dict.get("top_p", 1.0),
            max_tokens=config_dict.get("max_tokens"),
            dashscope_base_url=config_dict.get("dashscope_base_url"),
            use_dashscope=config_dict.get("use_dashscope", False),
            embedding_provider=config_dict.get("embedding_provider"),
            embedding_api_key=config_dict.get("embedding_api_key"),
            embedding_base_url=config_dict.get("embedding_base_url"),
            embedding_use_dashscope=config_dict.get("embedding_use_dashscope", False),
        )
