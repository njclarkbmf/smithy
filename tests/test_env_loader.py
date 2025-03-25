import os
import pytest
from unittest.mock import patch, MagicMock

from agentic_rag.utils.env_loader import load_environment

class TestEnvLoader:
    """Tests for the environment loader."""
    
    def test_load_environment(self):
        """Test loading environment variables."""
        # Create a mock environment
        mock_env = {
            "OPENAI_API_KEY": "test_api_key",
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-3.5-turbo",
            "EMBEDDING_MODEL": "text-embedding-ada-002",
            "TEMPERATURE": "0.5",
            "DB_PATH": "./custom_db",
            "CHUNK_SIZE": "2000",
            "CHUNK_OVERLAP": "200",
            "TOP_K": "10",
            "LOG_LEVEL": "DEBUG"
        }
        
        # Patch dotenv.load_dotenv to do nothing
        with patch("dotenv.load_dotenv"), \
             patch.dict(os.environ, mock_env, clear=True):
            
            config = load_environment()
            
            # Check all configuration values
            assert config["api_key"] == "test_api_key"
            assert config["provider"] == "openai"
            assert config["model"] == "gpt-3.5-turbo"
            assert config["embedding_model"] == "text-embedding-ada-002"
            assert config["temperature"] == 0.5
            assert config["db_path"] == "./custom_db"
            assert config["chunk_size"] == 2000
            assert config["chunk_overlap"] == 200
            assert config["top_k"] == 10
    
    def test_load_environment_anthropic(self):
        """Test loading environment variables with Anthropic provider."""
        # Create a mock environment
        mock_env = {
            "ANTHROPIC_API_KEY": "anthropic_key",
            "LLM_PROVIDER": "anthropic",
            "LLM_MODEL": "claude-3-opus",
        }
        
        # Patch dotenv.load_dotenv to do nothing
        with patch("dotenv.load_dotenv"), \
             patch.dict(os.environ, mock_env, clear=True):
            
            config = load_environment()
            
            # Check provider-specific values
            assert config["api_key"] == "anthropic_key"
            assert config["provider"] == "anthropic"
            assert config["model"] == "claude-3-opus"
    
    def test_load_environment_unsupported_provider(self):
        """Test loading environment variables with unsupported provider."""
        # Create a mock environment
        mock_env = {
            "LLM_PROVIDER": "unsupported",
        }
        
        # Patch dotenv.load_dotenv to do nothing and mock the logger
        with patch("dotenv.load_dotenv"), \
             patch("agentic_rag.utils.env_loader.logger.error") as mock_logging, \
             patch.dict(os.environ, mock_env, clear=True):
            
            config = load_environment()
            
            # Check error logging
            mock_logging.assert_called_once()
            assert "unsupported" in mock_logging.call_args[0][0].lower()
            
            # Check that it still sets the provider
            assert config["provider"] == "unsupported"
    
    def test_load_environment_defaults(self):
        """Test loading environment variables with defaults."""
        # Patch dotenv.load_dotenv to do nothing
        with patch("dotenv.load_dotenv"), \
             patch.dict(os.environ, {}, clear=True):
            
            config = load_environment()
            
            # Check default values - updated to match the actual default in your implementation
            assert config["provider"] == "openai"
            assert config["model"] == "gpt-4o-mini"  # Updated to match your actual default
            assert config["embedding_model"] == "text-embedding-3-large"
            assert config["temperature"] == 0.1
            assert config["db_path"] == "/app/data/lancedb"  # Updated to match your actual default
            assert config["chunk_size"] == 1500
            assert config["chunk_overlap"] == 150
            assert config["top_k"] == 5
