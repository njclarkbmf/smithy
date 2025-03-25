import pytest
from unittest.mock import patch, MagicMock, call

from agentic_rag.config import Config
from agentic_rag.document import Document
from agentic_rag.main import AgenticRAG

class TestAgenticRAG:
    """Tests for the AgenticRAG class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock()
        config.top_k = 3  # Add explicit top_k attribute
        return config
    
    @pytest.fixture
    def mock_doc_processor(self):
        """Create a mock document processor."""
        processor = MagicMock()
        processor.load_text_file.return_value = Document(content="test content", source="test.txt")
        processor.load_url.return_value = Document(content="url content", source="https://example.com")
        return processor
    
    @pytest.fixture
    def mock_vector_db(self):
        """Create a mock vector database manager."""
        vector_db = MagicMock()
        vector_db.add_documents.return_value = 5  # Return number of chunks
        return vector_db
    
    @pytest.fixture
    def mock_query_planner(self):
        """Create a mock query planner."""
        planner = MagicMock()
        planner.plan_query.return_value = ["query1", "query2"]
        return planner
    
    @pytest.fixture
    def mock_info_retriever(self):
        """Create a mock information retriever."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"id": "doc1", "source": "source1", "similarity": 0.9},
            {"id": "doc2", "source": "source2", "similarity": 0.8}
        ]
        return retriever
    
    @pytest.fixture
    def mock_response_generator(self):
        """Create a mock response generator."""
        generator = MagicMock()
        generator.generate.return_value = "Generated response"
        return generator
    
    @pytest.fixture
    def rag(self, mock_config, mock_doc_processor, mock_vector_db, 
             mock_query_planner, mock_info_retriever, mock_response_generator):
        """Create an AgenticRAG instance with mocked components."""
        with patch('agentic_rag.main.DocumentProcessor', return_value=mock_doc_processor), \
             patch('agentic_rag.main.VectorDBManager', return_value=mock_vector_db), \
             patch('agentic_rag.main.QueryPlanner', return_value=mock_query_planner), \
             patch('agentic_rag.main.InfoRetriever', return_value=mock_info_retriever), \
             patch('agentic_rag.main.ResponseGenerator', return_value=mock_response_generator):
            rag = AgenticRAG(mock_config)
        
        # Set the mocked components
        rag.document_processor = mock_doc_processor
        rag.vector_db = mock_vector_db
        rag.query_planner = mock_query_planner
        rag.info_retriever = mock_info_retriever
        rag.response_generator = mock_response_generator
        
        return rag
    
    def test_init(self, mock_config):
        """Test initialization."""
        with patch('agentic_rag.main.DocumentProcessor') as mock_doc_processor_cls, \
             patch('agentic_rag.main.VectorDBManager') as mock_vector_db_cls, \
             patch('agentic_rag.main.QueryPlanner') as mock_query_planner_cls, \
             patch('agentic_rag.main.InfoRetriever') as mock_info_retriever_cls, \
             patch('agentic_rag.main.ResponseGenerator') as mock_response_generator_cls:
            
            rag = AgenticRAG(mock_config)
            
            mock_doc_processor_cls.assert_called_once_with(mock_config)
            mock_vector_db_cls.assert_called_once_with(mock_config)
            mock_query_planner_cls.assert_called_once_with(mock_config)
            mock_info_retriever_cls.assert_called_once()
            mock_response_generator_cls.assert_called_once_with(mock_config)
    
    def test_init_default_config(self):
        """Test initialization with default config."""
        with patch('agentic_rag.main.Config') as mock_config_cls, \
             patch('agentic_rag.main.DocumentProcessor'), \
             patch('agentic_rag.main.VectorDBManager'), \
             patch('agentic_rag.main.QueryPlanner'), \
             patch('agentic_rag.main.InfoRetriever'), \
             patch('agentic_rag.main.ResponseGenerator'):
             
            rag = AgenticRAG()
            
            mock_config_cls.assert_called_once()
    
    def test_add_document(self, rag, mock_vector_db):
        """Test add_document method."""
        doc = Document(content="test", source="source")
        result = rag.add_document(doc)
        
        mock_vector_db.add_documents.assert_called_once_with([doc], rag.document_processor)
        assert result == 5  # Return value from mock_vector_db.add_documents
    
    def test_add_documents(self, rag, mock_vector_db):
        """Test add_documents method."""
        docs = [
            Document(content="test1", source="source1"),
            Document(content="test2", source="source2")
        ]
        result = rag.add_documents(docs)
        
        mock_vector_db.add_documents.assert_called_once_with(docs, rag.document_processor)
        assert result == 5  # Return value from mock_vector_db.add_documents
    
    def test_add_text_file(self, rag, mock_doc_processor, mock_vector_db):
        """Test add_text_file method."""
        result = rag.add_text_file("test.txt")
        
        mock_doc_processor.load_text_file.assert_called_once_with("test.txt")
        mock_vector_db.add_documents.assert_called_once()
        assert result == 5  # Return value from mock_vector_db.add_documents
    
    def test_add_url(self, rag, mock_doc_processor, mock_vector_db):
        """Test add_url method."""
        result = rag.add_url("https://example.com")
        
        mock_doc_processor.load_url.assert_called_once_with("https://example.com")
        mock_vector_db.add_documents.assert_called_once()
        assert result == 5  # Return value from mock_vector_db.add_documents
    
    def test_query(self, rag, mock_query_planner, mock_info_retriever, mock_response_generator):
        """Test query method."""
        response, debug_info = rag.query("test query")
        
        # Check method calls
        mock_query_planner.plan_query.assert_called_once_with("test query")
        mock_info_retriever.retrieve.assert_called_once_with(["query1", "query2"], rag.config.top_k)
        mock_response_generator.generate.assert_called_once_with("test query", [
            {"id": "doc1", "source": "source1", "similarity": 0.9},
            {"id": "doc2", "source": "source2", "similarity": 0.8}
        ])
        
        # Check return values
        assert response == "Generated response"
        assert debug_info["subqueries"] == ["query1", "query2"]
        assert len(debug_info["retrieved_info"]) == 2
        assert debug_info["retrieved_info"][0]["id"] == "doc1"
        assert debug_info["retrieved_info"][0]["source"] == "source1"
        assert debug_info["retrieved_info"][0]["similarity"] == 0.9
