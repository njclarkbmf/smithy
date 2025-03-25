import os
import sys
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

# Add the parent directory to sys.path to import agentic_rag
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set test environment variables
@pytest.fixture(scope="session", autouse=True)
def set_test_env():
    """Set environment variables for tests."""
    test_env = {
        "OPENAI_API_KEY": "test_key",
        "ANTHROPIC_API_KEY": "test_key",
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "gpt-4o",
        "DB_PATH": "./test_lancedb"
    }
    with patch.dict(os.environ, test_env, clear=False):
        yield

# Globally patch the client initializations to prevent real API calls
@pytest.fixture(autouse=True)
def mock_external_apis():
    """Automatically mock all external API clients for all tests."""
    with patch('openai.OpenAI') as mock_openai, \
         patch('anthropic.Anthropic') as mock_anthropic, \
         patch('dotenv.load_dotenv') as mock_load_dotenv:
        
        # Create mock OpenAI client with typical responses
        mock_openai_instance = MagicMock()
        mock_chat_response = MagicMock()
        mock_chat_response.choices = [MagicMock()]
        mock_chat_response.choices[0].message.content = "Mocked response"
        mock_openai_instance.chat.completions.create.return_value = mock_chat_response
        
        # Setup embedding response
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock()]
        mock_embedding_response.data[0].embedding = [0.1] * 1536
        mock_openai_instance.embeddings.create.return_value = mock_embedding_response
        
        # Return the mock instance when OpenAI is initialized
        mock_openai.return_value = mock_openai_instance
        
        # Similar setup for Anthropic
        mock_anthropic_instance = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Mocked anthropic response"
        mock_anthropic_response = MagicMock()
        mock_anthropic_response.content = [mock_content]
        mock_anthropic_instance.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_anthropic_instance
        
        yield

# Set up mock modules before any imports
with patch.dict('sys.modules', {
    'lancedb': MagicMock(),
    'lancedb.pydantic': MagicMock(),
    'lancedb.embeddings': MagicMock()
}):
    # Import modules with mocks in place
    from agentic_rag.config import Config
    from agentic_rag.document import Document, DocumentProcessor
    from agentic_rag.vectordb import VectorDBManager, ChunkModel
    from agentic_rag.agents.query_planner import QueryPlanner
    from agentic_rag.agents.info_retriever import InfoRetriever
    from agentic_rag.agents.response_generator import ResponseGenerator
    from agentic_rag.main import AgenticRAG

@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    Artificial Intelligence (AI) is transforming the world in unprecedented ways.
    From self-driving cars to virtual assistants, AI technologies are becoming increasingly integrated into our daily lives.
    Machine learning, a subset of AI, focuses on developing algorithms that can learn from and make predictions based on data.
    Deep learning, a specialized form of machine learning, uses neural networks with many layers to analyze various factors of data.
    """

@pytest.fixture
def sample_document(sample_text):
    """Sample document for testing."""
    return Document(
        content=sample_text,
        source="ai_overview.txt",
        doc_id="doc123",
        metadata={"author": "Test Author", "topic": "AI"}
    )

@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = MagicMock()
    config.model = "gpt-4o"
    config.provider = "openai"
    config.embedding_model = "text-embedding-3-large"
    config.embedding_dimensions = 1536
    config.db_path = "./test_lancedb"
    config.chunk_size = 1000
    config.chunk_overlap = 100
    config.temperature = 0.1
    config.top_k = 3
    
    # Create mock client
    config.client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Mock response"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    config.client.chat.completions.create.return_value = mock_response
    
    # Mock embeddings
    mock_embedding_response = MagicMock()
    mock_data = [MagicMock()]
    mock_data[0].embedding = [0.1, 0.2, 0.3, 0.4]
    mock_embedding_response.data = mock_data
    config.client.embeddings.create.return_value = mock_embedding_response
    
    # Mock database
    mock_db = MagicMock()
    mock_table = MagicMock()
    mock_search = MagicMock()
    mock_limit = MagicMock()
    
    mock_limit.to_pandas.return_value = pd.DataFrame([
        {"id": "doc1", "source": "source1", "content": "content1", "embedding": [0.1, 0.2, 0.3, 0.4]},
        {"id": "doc2", "source": "source2", "content": "content2", "embedding": [0.1, 0.2, 0.3, 0.4]}
    ])
    
    mock_search.limit.return_value = mock_limit
    mock_table.search.return_value = mock_search
    mock_table.add.return_value = None
    mock_db.open_table.return_value = mock_table
    mock_db.table_names.return_value = ["document_chunks"]
    config.db = mock_db
    
    return config

@pytest.fixture
def document_processor(mock_config):
    """Document processor for testing."""
    return DocumentProcessor(mock_config)

@pytest.fixture
def vector_db_manager(mock_config):
    """Vector database manager for testing."""
    with patch('agentic_rag.vectordb.VectorDBManager._ensure_table_exists'):
        manager = VectorDBManager(mock_config)
        
        # Mock get_embeddings method
        manager._get_embeddings = MagicMock(return_value=[
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4]
        ])
        
        return manager

@pytest.fixture
def query_planner(mock_config):
    """Query planner for testing."""
    planner = QueryPlanner(mock_config)
    
    # Mock the plan_query method
    planner.plan_query = MagicMock(return_value=["subquery1", "subquery2"])
    
    return planner

@pytest.fixture
def info_retriever(vector_db_manager):
    """Information retriever for testing."""
    retriever = InfoRetriever(vector_db_manager)
    
    # Set up the retrieve method to return test data
    def mock_retrieve(queries, top_k=None):
        # Return unique results for each query
        results = []
        for i, query in enumerate(queries):
            for j in range(1, 3):  # Two results per query
                results.append({
                    "id": f"doc{i+1}_{j}",
                    "source": f"source{i+1}_{j}",
                    "content": f"content{i+1}_{j}",
                    "similarity": 0.9 - (j * 0.1),
                    "embedding": [0.1, 0.2, 0.3, 0.4]
                })
        return results
        
    # Replace the actual implementation with our mock
    retriever.retrieve = mock_retrieve
    
    return retriever

@pytest.fixture
def response_generator(mock_config):
    """Response generator for testing."""
    generator = ResponseGenerator(mock_config)
    
    # Mock the generate method
    generator.generate = MagicMock(return_value="This is a generated response.")
    
    return generator

@pytest.fixture
def rag_system(mock_config, document_processor, vector_db_manager, 
               query_planner, info_retriever, response_generator):
    """RAG system for testing."""
    rag = AgenticRAG(mock_config)
    
    # Replace components with mocks
    rag.document_processor = document_processor
    rag.vector_db = vector_db_manager
    rag.query_planner = query_planner
    rag.info_retriever = info_retriever
    rag.response_generator = response_generator
    
    return rag
