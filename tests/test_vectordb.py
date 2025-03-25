import pytest
from unittest.mock import patch, MagicMock, call

import pandas as pd
import numpy as np

from agentic_rag.config import Config
from agentic_rag.vectordb import VectorDBManager

class TestVectorDBManager:
    """Tests for the VectorDBManager class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = MagicMock()
        config.db_path = "./test_db"
        config.top_k = 3
        config.embedding_dimensions = 4
        
        # Add client with embeddings
        config.client = MagicMock()
        mock_embedding_response = MagicMock()
        mock_data = [
            MagicMock(embedding=[0.1, 0.2, 0.3, 0.4]),
            MagicMock(embedding=[0.5, 0.6, 0.7, 0.8])
        ]
        mock_embedding_response.data = mock_data
        config.client.embeddings.create.return_value = mock_embedding_response
        
        return config
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database."""
        mock_db = MagicMock()
        mock_table = MagicMock()
        mock_db.open_table.return_value = mock_table
        mock_db.table_names.return_value = []
        return mock_db
    
    @pytest.fixture
    def manager(self, mock_config, mock_db):
        """Create a VectorDBManager with mocks."""
        mock_config.db = mock_db
        with patch('agentic_rag.vectordb.VectorDBManager._ensure_table_exists'):
            manager = VectorDBManager(mock_config)
            manager._ensure_table_exists = MagicMock()  # Replace the method
            return manager
    
    def test_ensure_table_exists_creates_table(self, manager, mock_db):
        """Test that _ensure_table_exists creates a table if it doesn't exist."""
        mock_db.table_names.return_value = []
        
        # Call the real method (not the mock)
        manager._ensure_table_exists = VectorDBManager._ensure_table_exists
        manager._ensure_table_exists(manager)
        
        mock_db.create_table.assert_called_once()
    
    def test_ensure_table_exists_skips_if_exists(self, manager, mock_db):
        """Test that _ensure_table_exists skips if table exists."""
        mock_db.table_names.return_value = ["document_chunks"]
        
        # Call the real method (not the mock)
        manager._ensure_table_exists = VectorDBManager._ensure_table_exists
        manager._ensure_table_exists(manager)
        
        mock_db.create_table.assert_not_called()
    
    def test_get_table(self, manager, mock_db):
        """Test get_table method."""
        manager.get_table()
        mock_db.open_table.assert_called_once_with("document_chunks")
    
    def test_add_documents(self, manager, mock_db):
        """Test add_documents method."""
        # Set up mocks
        mock_processor = MagicMock()
        mock_processor.process_document.side_effect = lambda doc: [
            {"id": f"{doc.doc_id}_0", "content": "chunk 1"},
            {"id": f"{doc.doc_id}_1", "content": "chunk 2"}
        ]
        
        mock_docs = [
            MagicMock(doc_id="doc1"),
            MagicMock(doc_id="doc2")
        ]
        
        # Call the method
        with patch('pandas.DataFrame.to_dict', return_value=[]):
            result = manager.add_documents(mock_docs, mock_processor)
        
        # Assertions
        assert result == 4  # 4 chunks total
        mock_processor.process_document.assert_has_calls([call(mock_docs[0]), call(mock_docs[1])])
        mock_db.open_table().add.assert_called_once()
    
    def test_get_embeddings(self, manager):
        """Test _get_embeddings method."""
        # The mock is already set up in the fixture
        
        # Call the method
        result = manager._get_embeddings(["text1", "text2"])
        
        # Assertions
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3, 0.4]
        assert result[1] == [0.5, 0.6, 0.7, 0.8]
        manager.config.client.embeddings.create.assert_called_once()
    
    def test_get_embeddings_handles_series(self, manager):
        """Test that _get_embeddings handles pandas Series."""
        # Create a Series
        series = pd.Series(["text1", "text2"])
        
        # Call the method
        result = manager._get_embeddings(series)
        
        # Assertions
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3, 0.4]
        assert result[1] == [0.5, 0.6, 0.7, 0.8]
        manager.config.client.embeddings.create.assert_called_once()
    
    def test_get_embeddings_handles_error(self, manager):
        """Test that _get_embeddings handles errors gracefully."""
        # Mock the config.client.embeddings.create method to raise an exception
        manager.config.client.embeddings.create.side_effect = Exception("API error")
        
        # Call the method
        result = manager._get_embeddings(["text1", "text2"])
        
        # Assertions - should return zero embeddings as fallback
        assert len(result) == 2
        assert len(result[0]) == 4  # config.embedding_dimensions
        assert all(v == 0.0 for v in result[0])
    
    def test_search(self, manager):
        """Test search method."""
        # Mock methods
        manager._get_embeddings = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
        mock_table = manager.get_table()
        mock_search = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_df = pd.DataFrame([{"id": "doc1", "content": "result"}])
        mock_search.to_pandas.return_value = mock_df
        
        # Call the method
        result = manager.search("test query")
        
        # Assertions
        manager._get_embeddings.assert_called_once_with(["test query"])
        mock_table.search.assert_called_once_with([0.1, 0.2, 0.3, 0.4])
        mock_search.limit.assert_called_once_with(3)  # config.top_k
        mock_search.to_pandas.assert_called_once()
        assert len(result) == 1
        assert result[0]["id"] == "doc1"
        assert result[0]["content"] == "result"
    
    def test_search_with_custom_top_k(self, manager):
        """Test search method with custom top_k."""
        # Mock methods
        manager._get_embeddings = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
        mock_table = manager.get_table()
        mock_search = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_df = pd.DataFrame([{"id": "doc1", "content": "result"}])
        mock_search.to_pandas.return_value = mock_df
        
        # Call the method with custom top_k
        result = manager.search("test query", top_k=5)
        
        # Assertions
        mock_search.limit.assert_called_once_with(5)
    
    def test_search_handles_error(self, manager):
        """Test that search handles errors gracefully."""
        # Mock _get_embeddings to raise an exception
        manager._get_embeddings = MagicMock(side_effect=Exception("API error"))
        
        # Call the method
        result = manager.search("test query")
        
        # Assertions - should return empty list on error
        assert result == []
