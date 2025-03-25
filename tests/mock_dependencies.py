"""
This module provides mock implementations of dependencies for testing.
It allows tests to run without requiring all external dependencies to be installed.
"""
import pandas as pd
from unittest.mock import MagicMock
from typing import List, Dict, Any

# Mock classes for LanceDB
class MockLanceModel:
    """Mock implementation of LanceModel for testing."""
    @classmethod
    def schema(cls):
        """Return a mock schema."""
        return {
            "fields": [
                {"name": "id", "type": "string"},
                {"name": "content", "type": "string"},
                {"name": "embedding", "type": "list<float>"}
            ]
        }

# Mock OpenAI and Anthropic clients
class MockOpenAIClient:
    """Mock implementation of OpenAI client for testing."""
    def __init__(self, *args, **kwargs):
        self.chat = MagicMock()
        self.embeddings = MagicMock()
        
        # Set up mock responses
        chat_response = MagicMock()
        chat_response.choices = [MagicMock()]
        chat_response.choices[0].message.content = "Mock response from OpenAI"
        self.chat.completions.create.return_value = chat_response
        
        embedding_response = MagicMock()
        embedding_response.data = [MagicMock()]
        embedding_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4]
        self.embeddings.create.return_value = embedding_response

class MockAnthropicClient:
    """Mock implementation of Anthropic client for testing."""
    def __init__(self, *args, **kwargs):
        self.messages = MagicMock()
        
        # Set up mock responses
        messages_response = MagicMock()
        content = MagicMock()
        content.text = "Mock response from Anthropic"
        messages_response.content = [content]
        self.messages.create.return_value = messages_response

# Mock LanceDB
class MockLanceDB:
    """Mock implementation of LanceDB for testing."""
    @staticmethod
    def connect(*args, **kwargs):
        """Return a mock LanceDB connection."""
        mock_db = MagicMock()
        mock_table = MagicMock()
        
        # Set up mock table
        mock_table.add.return_value = None
        mock_search = MagicMock()
        mock_search.limit.return_value = mock_search
        mock_search.to_pandas.return_value = pd.DataFrame([
            {"id": "doc1", "source": "source1", "content": "content1", "similarity": 0.9},
            {"id": "doc2", "source": "source2", "content": "content2", "similarity": 0.8}
        ])
        mock_table.search.return_value = mock_search
        
        # Set up mock database
        mock_db.table_names.return_value = ["document_chunks"]
        mock_db.open_table.return_value = mock_table
        mock_db.create_table.return_value = mock_table
        
        return mock_db

# Mock functions for embeddings
def mock_with_embeddings(data: pd.DataFrame, embedding_function: callable, 
                         text_column: str, embedding_column: str) -> pd.DataFrame:
    """Mock implementation of with_embeddings for testing."""
    if text_column in data.columns:
        data[embedding_column] = [[0.1, 0.2, 0.3, 0.4]] * len(data)
    return data
