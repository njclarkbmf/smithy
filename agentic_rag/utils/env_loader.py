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
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    config = {
        "api_key": None,  # Will be set based on provider
        "provider": provider,
        "model": os.getenv("LLM_MODEL", _default_model_for_provider(provider)),
        "embedding_model": os.getenv(
            "EMBEDDING_MODEL", _default_embedding_for_provider(provider)
        ),
        "temperature": float(os.getenv("TEMPERATURE", "0.1")),
        "top_p": float(os.getenv("TOP_P", "1.0")),
        "max_tokens": _optional_int(os.getenv("MAX_TOKENS")),
        # Qwen / DashScope settings
        "dashscope_base_url": os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        "use_dashscope": os.getenv("USE_DASHSCOPE", "true").lower() == "true",
        # Separate embedding provider settings
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", provider).lower(),
        "embedding_api_key": os.getenv("EMBEDDING_API_KEY"),
        "embedding_base_url": os.getenv("EMBEDDING_BASE_URL"),
        "embedding_use_dashscope": os.getenv(
            "EMBEDDING_USE_DASHSCOPE",
            os.getenv("USE_DASHSCOPE", "true"),
        ).lower() == "true",
        # Vector DB Configuration
        "db_path": os.getenv("DB_PATH", "./data/lancedb"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1500")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "150")),
        "top_k": int(os.getenv("TOP_K", "5")),
    }

    # Set API key based on provider
    config["api_key"] = _api_key_for_provider(provider, config)

    # Fill embedding API key fallback
    if not config["embedding_api_key"]:
        config["embedding_api_key"] = config["api_key"]


def _default_model_for_provider(provider: str) -> str:
    return {
        "openai": "gpt-4o",
        "anthropic": "claude-3-5-sonnet-20241022",
        "qwen": "qwen-plus",
    }.get(provider, "gpt-4o")


def _default_embedding_for_provider(provider: str) -> str:
    return {
        "openai": "text-embedding-3-large",
        "qwen": "text-embedding-v3",
    }.get(provider, "text-embedding-3-large")


def _api_key_for_provider(provider: str, config: Dict[str, Any]) -> str | None:
    key_map = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "qwen": os.getenv("DASHSCOPE_API_KEY"),
    }
    api_key = key_map.get(provider.lower())
    if not api_key:
        logger.warning(
            f"{provider.upper()}_API_KEY / DASHSCOPE_API_KEY "
            f"not found in environment variables"
        )
    return api_key


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
