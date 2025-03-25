import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_environment(env_file: str = ".env") -> Dict[str, Any]:
    """
    Load environment variables from .env file and return config dictionary.

    Args:
        env_file: Path to the .env file

    Returns:
        Dictionary containing configuration parameters
    """
    # Load environment variables from file
    load_dotenv(env_file)

    # Set up logging level
    log_level = os.getenv("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # LLM Configuration
    config = {
        "api_key": None,  # Will be set based on provider
        "provider": os.getenv("LLM_PROVIDER", "openai"),
        "model": os.getenv("LLM_MODEL", "gpt-4o"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        "temperature": float(os.getenv("TEMPERATURE", "0.1")),
        # Vector DB Configuration
        "db_path": os.getenv("DB_PATH", "./data/lancedb"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1500")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "150")),
        "top_k": int(os.getenv("TOP_K", "5")),
    }

    # Set API key based on provider
    if config["provider"].lower() == "openai":
        config["api_key"] = os.getenv("OPENAI_API_KEY")
        if not config["api_key"]:
            logger.warning("OPENAI_API_KEY not found in environment variables")
    elif config["provider"].lower() == "anthropic":
        config["api_key"] = os.getenv("ANTHROPIC_API_KEY")
        if not config["api_key"]:
            logger.warning("ANTHROPIC_API_KEY not found in environment variables")
    else:
        logger.error(f"Unsupported provider: {config['provider']}")

    return config
