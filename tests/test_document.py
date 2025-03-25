import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from agentic_rag.config import Config
from agentic_rag.document import Document, DocumentProcessor

class TestDocument:
    """Tests for the Document class."""
    
    def test_init_with_defaults(self):
        """Test initialization with default values."""
        doc = Document(content="Test content", source="test.txt")
        
        assert doc.content == "Test content"
        assert doc.source == "test.txt"
        assert isinstance(doc.doc_id, str)  # Should generate a UUID
        assert isinstance(doc.metadata, dict)
        assert len(doc.metadata) == 0
    
    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        metadata = {"author": "Test Author", "date": "2023-01-01"}
        doc = Document(
            content="Test content",
            source="test.txt",
            doc_id="custom_id",
            metadata=metadata
        )
        
        assert doc.content == "Test content"
        assert doc.source == "test.txt"
        assert doc.doc_id == "custom_id"
        assert doc.metadata == metadata
    
    def test_repr(self):
        """Test string representation."""
        doc = Document(content="Test content", source="test.txt", doc_id="test_id")
        
        assert repr(doc) == "Document(id=test_id, source=test.txt, content_length=12)"

class TestDocumentProcessor:
    """Tests for the DocumentProcessor class."""
    
    @pytest.fixture
    def config(self):
        """Create a test config."""
        with patch("agentic_rag.config.Config._init_clients"):
            return Config(
                chunk_size=10,
                chunk_overlap=2
            )
    
    @pytest.fixture
    def processor(self, config):
        """Create a test document processor."""
        return DocumentProcessor(config)
    
    def test_process_document(self, processor):
        """Test processing a document into chunks."""
        doc = Document(
            content="This is a test document with more than ten words to test chunking.",
            source="test.txt",
            doc_id="test_id"
        )
        
        chunks = processor.process_document(doc)
        
        # Should split into multiple chunks
        assert len(chunks) > 1
        
        # Check first chunk
        assert chunks[0]["id"] == "test_id_0"
        assert chunks[0]["doc_id"] == "test_id"
        assert chunks[0]["source"] == "test.txt"
        assert len(chunks[0]["content"]) <= 10 + 2  # chunk_size + overlap
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["metadata"] == {}
    
    def test_load_text_file(self, processor):
        """Test loading a document from a text file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
            temp.write("This is a test file content.")
            temp_path = temp.name
        
        try:
            # Load the file
            doc = processor.load_text_file(temp_path)
            
            assert doc.content == "This is a test file content."
            assert doc.source == temp_path
            assert doc.doc_id == os.path.basename(temp_path)
            assert doc.metadata == {}
            
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_load_url(self, processor):
        """Test loading content from a URL."""
        # Create mocks for the external dependencies
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Test content</p></body></html>"
        mock_response.raise_for_status = MagicMock()
        
        # Create a mock for BeautifulSoup
        mock_soup = MagicMock()
        mock_soup.get_text.return_value = "Test content"
        
        # Patch requests.get and BeautifulSoup with the correct import paths
        with patch('agentic_rag.document.requests.get', return_value=mock_response) as mock_get, \
             patch('agentic_rag.document.BeautifulSoup', return_value=mock_soup) as mock_bs4:
            
            doc = processor.load_url("https://example.com")
            
            # Verify the calls
            mock_get.assert_called_once_with("https://example.com")
            mock_bs4.assert_called_once_with(mock_response.text, 'html.parser')
            mock_soup.get_text.assert_called_once_with(separator='\n')
            
            # Verify the document
            assert doc.content == "Test content"
            assert doc.source == "https://example.com"
            assert isinstance(doc.doc_id, str)
            assert doc.metadata == {}
