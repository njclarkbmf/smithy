import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np

from agentic_rag.agents.query_planner import QueryPlanner
from agentic_rag.agents.info_retriever import InfoRetriever
from agentic_rag.agents.response_generator import ResponseGenerator

class TestQueryPlanner:
    """Tests for the QueryPlanner class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock()
        config.provider = "openai"
        config.model = "gpt-4o"
        
        # Mock the OpenAI client
        mock_openai_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "query1\nquery2\nquery3"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_openai_client.chat.completions.create.return_value = mock_response
        config.client = mock_openai_client
        
        return config
    
    @pytest.fixture
    def query_planner(self, mock_config):
        """Create a QueryPlanner instance with a mock config."""
        return QueryPlanner(mock_config)
    
    def test_plan_query_openai(self, query_planner, mock_config):
        """Test plan_query with OpenAI provider."""
        # Switch to using the real method, not the mock
        query_planner.plan_query = QueryPlanner.plan_query
        
        subqueries = query_planner.plan_query(query_planner, "What is artificial intelligence?")
        
        # Check that the correct API call was made
        mock_config.client.chat.completions.create.assert_called_once()
        args, kwargs = mock_config.client.chat.completions.create.call_args
        
        # Check the arguments
        assert kwargs["model"] == "gpt-4o"
        assert len(kwargs["messages"]) == 2
        assert kwargs["messages"][0]["role"] == "system"
        assert kwargs["messages"][1]["role"] == "user"
        assert kwargs["messages"][1]["content"] == "What is artificial intelligence?"
        assert kwargs["temperature"] == 0.2
        
        # Check the result
        assert subqueries == ["query1", "query2", "query3"]
    
    def test_plan_query_error(self, query_planner, mock_config):
        """Test plan_query error handling."""
        # Make the API call raise an exception
        mock_config.client.chat.completions.create.side_effect = Exception("API error")
        
        # Switch to using the real method, not the mock
        query_planner.plan_query = QueryPlanner.plan_query
        
        # Call the method with patched logging.error
        with patch('agentic_rag.agents.query_planner.logger.error') as mock_logging:
            result = query_planner.plan_query(query_planner, "What is artificial intelligence?")
            
            # Check that the error was logged
            mock_logging.assert_called_once()
            
            # Check that the original query is returned as fallback
            assert result == ["What is artificial intelligence?"]
    
    def test_plan_query_anthropic(self, mock_config):
        """Test plan_query with Anthropic provider."""
        # Change the provider to Anthropic
        mock_config.provider = "anthropic"
        
        # Mock the Anthropic client
        mock_anthropic_client = MagicMock()
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "query1\nquery2\nquery3"
        mock_response.content = [mock_content]
        mock_anthropic_client.messages.create.return_value = mock_response
        mock_config.client = mock_anthropic_client
        
        # Create the query planner
        query_planner = QueryPlanner(mock_config)
        
        # Switch to using the real method, not the mock
        query_planner.plan_query = QueryPlanner.plan_query
        
        # Call the method
        subqueries = query_planner.plan_query(query_planner, "What is artificial intelligence?")
        
        # Check that the correct API call was made
        mock_config.client.messages.create.assert_called_once()
        args, kwargs = mock_config.client.messages.create.call_args
        
        # Check the arguments
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["system"] is not None
        assert len(kwargs["messages"]) == 1
        assert kwargs["messages"][0]["role"] == "user"
        assert kwargs["messages"][0]["content"] == "What is artificial intelligence?"
        assert kwargs["temperature"] == 0.2
        
        # Check the result
        assert subqueries == ["query1", "query2", "query3"]

class TestInfoRetriever:
    """Tests for the InfoRetriever class."""
    
    @pytest.fixture
    def mock_vector_db(self):
        """Create a mock vector database manager."""
        vector_db = MagicMock()
        
        # Mock the search method to return results based on query
        def mock_search(query, top_k):
            # Generate unique results for each query
            return [
                {
                    "id": f"{query.replace(' ', '_')}_doc{i}",
                    "source": f"source{i}",
                    "content": f"content{i}",
                    "embedding": [0.1, 0.2, 0.3, 0.4]
                }
                for i in range(1, top_k + 1)
            ]
        
        vector_db.search = mock_search
        
        # Mock _get_embeddings
        vector_db._get_embeddings = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
        
        return vector_db
    
    @pytest.fixture
    def info_retriever(self, mock_vector_db):
        """Create an InfoRetriever instance with a mock vector database."""
        return InfoRetriever(mock_vector_db)
    
    def test_retrieve_single_query(self, info_retriever, mock_vector_db):
        """Test retrieve method with a single query."""
        # Use the real implementation
        info_retriever.retrieve = InfoRetriever.retrieve
        
        results = info_retriever.retrieve(info_retriever, ["What is AI?"], 2)
        
        # Check the results
        assert len(results) == 2
        assert results[0]["id"] == "What_is_AI?_doc1"
        assert results[1]["id"] == "What_is_AI?_doc2"
    
    def test_retrieve_multiple_queries(self, info_retriever, mock_vector_db):
        """Test retrieve method with multiple queries."""
        # Use the real implementation
        info_retriever.retrieve = InfoRetriever.retrieve
        
        results = info_retriever.retrieve(info_retriever, ["What is AI?", "How does machine learning work?"], 2)
        
        # Check that different results are returned for each query
        assert len(results) == 4  # 2 queries * 2 results each
        
        # Check first query results
        assert results[0]["id"] == "What_is_AI?_doc1"
        assert results[1]["id"] == "What_is_AI?_doc2"
        
        # Check second query results
        assert results[2]["id"] == "How_does_machine_learning_work?_doc1"
        assert results[3]["id"] == "How_does_machine_learning_work?_doc2"
    
    def test_retrieve_deduplication(self, info_retriever, mock_vector_db):
        """Test that retrieve deduplicates results with the same ID."""
        # Override the search method to return the same doc for both queries
        mock_vector_db.search = MagicMock(return_value=[
            {"id": "doc1", "source": "source1", "content": "content1", "embedding": [0.1, 0.2, 0.3, 0.4]}
        ])
        
        # Use the real implementation
        info_retriever.retrieve = InfoRetriever.retrieve
        
        results = info_retriever.retrieve(info_retriever, ["What is AI?", "How does machine learning work?"], 1)
        
        # Check that the search method was called twice
        assert mock_vector_db.search.call_count == 2
        
        # Check the results (should be deduplicated to just one result)
        assert len(results) == 1
        assert results[0]["id"] == "doc1"
    
    def test_retrieve_sort_by_relevance(self, info_retriever, mock_vector_db):
        """Test that retrieve sorts results by relevance to the main query."""
        # Mock the vector_db.search to return docs with different IDs
        mock_vector_db.search.side_effect = [
            [
                {"id": "Main_query_doc1", "source": "source1", "content": "content1", 
                 "embedding": [0.1, 0.2, 0.3, 0.4]}
            ],
            [
                {"id": "Secondary_query_doc2", "source": "source2", "content": "content2", 
                 "embedding": [0.5, 0.6, 0.7, 0.8]}
            ]
        ]
        
        # Use the real implementation but with mocked np functions
        info_retriever.retrieve = InfoRetriever.retrieve
        
        # Mock np.dot to control similarity values
        with patch('agentic_rag.agents.info_retriever.np.dot', side_effect=[0.7, 0.9]), \
             patch('agentic_rag.agents.info_retriever.np.linalg.norm', return_value=1.0):
            
            results = info_retriever.retrieve(info_retriever, ["Main query", "Secondary query"], 1)
        
        # Check that the results are sorted by similarity
        assert len(results) == 2
        # The secondary query document has higher similarity and should be first
        assert "Secondary" in results[0]["id"]
        assert results[0]["similarity"] == 0.9
        assert "Main" in results[1]["id"]
        assert results[1]["similarity"] == 0.7

class TestResponseGenerator:
    """Tests for the ResponseGenerator class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock()
        config.provider = "openai"
        config.model = "gpt-4o"
        config.temperature = 0.1
        
        # Mock the OpenAI client
        mock_openai_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Generated response from OpenAI"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_openai_client.chat.completions.create.return_value = mock_response
        config.client = mock_openai_client
        
        return config
    
    @pytest.fixture
    def response_generator(self, mock_config):
        """Create a ResponseGenerator instance with a mock config."""
        return ResponseGenerator(mock_config)
    
    def test_generate_openai(self, response_generator, mock_config):
        """Test generate method with OpenAI provider."""
        # Use the real implementation
        response_generator.generate = ResponseGenerator.generate
        
        retrieved_info = [
            {"source": "source1", "content": "content1"},
            {"source": "source2", "content": "content2"}
        ]
        
        response = response_generator.generate(response_generator, "What is artificial intelligence?", retrieved_info)
        
        # Check that the correct API call was made
        mock_config.client.chat.completions.create.assert_called_once()
        args, kwargs = mock_config.client.chat.completions.create.call_args
        
        # Check the arguments
        assert kwargs["model"] == "gpt-4o"
        assert len(kwargs["messages"]) == 2
        assert kwargs["messages"][0]["role"] == "system"
        assert kwargs["messages"][1]["role"] == "user"
        assert "What is artificial intelligence?" in kwargs["messages"][1]["content"]
        assert "Source: source1" in kwargs["messages"][1]["content"]
        assert "content1" in kwargs["messages"][1]["content"]
        assert "Source: source2" in kwargs["messages"][1]["content"]
        assert "content2" in kwargs["messages"][1]["content"]
        assert kwargs["temperature"] == 0.1
        
        # Check the result
        assert response == "Generated response from OpenAI"
    
    def test_generate_anthropic(self, mock_config):
        """Test generate method with Anthropic provider."""
        # Change the provider to Anthropic
        mock_config.provider = "anthropic"
        
        # Mock the Anthropic client
        mock_anthropic_client = MagicMock()
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Generated response from Anthropic"
        mock_response.content = [mock_content]
        mock_anthropic_client.messages.create.return_value = mock_response
        mock_config.client = mock_anthropic_client
        
        # Create the response generator
        response_generator = ResponseGenerator(mock_config)
        
        # Use the real implementation
        response_generator.generate = ResponseGenerator.generate
        
        retrieved_info = [
            {"source": "source1", "content": "content1"},
            {"source": "source2", "content": "content2"}
        ]
        
        response = response_generator.generate(response_generator, "What is artificial intelligence?", retrieved_info)
        
        # Check that the correct API call was made
        mock_anthropic_client.messages.create.assert_called_once()
        args, kwargs = mock_anthropic_client.messages.create.call_args
        
        # Check the arguments
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["system"] is not None
        assert len(kwargs["messages"]) == 1
        assert kwargs["messages"][0]["role"] == "user"
        assert "What is artificial intelligence?" in kwargs["messages"][0]["content"]
        assert "Source: source1" in kwargs["messages"][0]["content"]
        assert "content1" in kwargs["messages"][0]["content"]
        assert kwargs["temperature"] == 0.1
        
        # Check the result
        assert response == "Generated response from Anthropic"
    
    def test_generate_error(self, response_generator, mock_config):
        """Test generate error handling."""
        # Make the API call raise an exception
        mock_config.client.chat.completions.create.side_effect = Exception("API error")
        
        # Use the real implementation
        response_generator.generate = ResponseGenerator.generate
        
        # Call the method with patched logging.error
        with patch('agentic_rag.agents.response_generator.logger.error') as mock_logging:
            result = response_generator.generate(
                response_generator, "What is AI?", [{"source": "s1", "content": "c1"}]
            )
            
            # Check that the error was logged
            mock_logging.assert_called_once()
            
            # Check that an error message is returned
            assert "error" in result.lower()
