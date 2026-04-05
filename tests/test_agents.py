import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np
from agentic_rag.agents.query_planner import QueryPlanner
from agentic_rag.agents.info_retriever import InfoRetriever
from agentic_rag.agents.response_generator import ResponseGenerator

class TestQueryPlanner:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.provider = "openai"
        config.model = "gpt-4o"

        mock_router = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "query1\nquery2\nquery3"
        mock_router.chat.return_value = mock_resp
        config.client = mock_router
        return config

    @pytest.fixture
    def query_planner(self, mock_config):
        return QueryPlanner(mock_config)

    def test_plan_query(self, query_planner, mock_config):
        subqueries = query_planner.plan_query("What is artificial intelligence?")
        mock_config.client.chat.assert_called_once()
        call_kwargs = mock_config.client.chat.call_args[1]
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0].role == "system"
        assert call_kwargs["messages"][1].content == "What is artificial intelligence?"
        assert call_kwargs["temperature"] == 0.2
        assert subqueries == ["query1", "query2", "query3"]

    def test_plan_query_error(self, query_planner, mock_config):
        mock_config.client.chat.side_effect = Exception("API error")
        with patch('agentic_rag.agents.query_planner.logger.error') as mock_logging:
            result = query_planner.plan_query("What is artificial intelligence?")
            mock_logging.assert_called_once()
            assert result == ["What is artificial intelligence?"]


class TestInfoRetriever:
    @pytest.fixture
    def mock_vector_db(self):
        vector_db = MagicMock()
        def mock_search(query, top_k):
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
        vector_db._get_embeddings = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
        return vector_db

    @pytest.fixture
    def info_retriever(self, mock_vector_db):
        return InfoRetriever(mock_vector_db)

    def test_retrieve_single_query(self, info_retriever, mock_vector_db):
        info_retriever.retrieve = InfoRetriever.retrieve
        results = info_retriever.retrieve(info_retriever, ["What is AI?"], 2)
        assert len(results) == 2
        assert results[0]["id"] == "What_is_AI?_doc1"
        assert results[1]["id"] == "What_is_AI?_doc2"

    def test_retrieve_multiple_queries(self, info_retriever, mock_vector_db):
        info_retriever.retrieve = InfoRetriever.retrieve
        results = info_retriever.retrieve(info_retriever, [
            "What is AI?", "How does machine learning work?"
        ], 2)
        assert len(results) == 4

    def test_retrieve_deduplication(self, info_retriever, mock_vector_db):
        mock_vector_db.search = MagicMock(return_value=[
            {"id": "doc1", "source": "source1", "content": "content1", "embedding": [0.1, 0.2, 0.3, 0.4]}
        ])
        info_retriever.retrieve = InfoRetriever.retrieve
        results = info_retriever.retrieve(info_retriever, [
            "What is AI?", "How does machine learning work?"
        ], 1)
        assert mock_vector_db.search.call_count == 2
        assert len(results) == 1
        assert results[0]["id"] == "doc1"

    def test_retrieve_sort_by_relevance(self, info_retriever, mock_vector_db):
        mock_vector_db.search.side_effect = [
            [{"id": "Main_query_doc1", "source": "source1", "content": "content1", "embedding": [0.1, 0.2, 0.3, 0.4]}],
            [{"id": "Secondary_query_doc2", "source": "source2", "content": "content2", "embedding": [0.5, 0.6, 0.7, 0.8]}]
        ]
        info_retriever.retrieve = InfoRetriever.retrieve
        with patch('agentic_rag.agents.info_retriever.np.dot', side_effect=[0.7, 0.9]), \
             patch('agentic_rag.agents.info_retriever.np.linalg.norm', return_value=1.0):
            results = info_retriever.retrieve(info_retriever, ["Main query", "Secondary query"], 1)
        assert len(results) == 2
        assert "Secondary" in results[0]["id"]
        assert results[0]["similarity"] == 0.9


class TestResponseGenerator:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.provider = "openai"
        config.model = "gpt-4o"
        config.temperature = 0.1

        mock_router = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "Generated response from OpenAI"
        mock_router.chat.return_value = mock_resp
        config.client = mock_router
        return config

    @pytest.fixture
    def response_generator(self, mock_config):
        return ResponseGenerator(mock_config)

    def test_generate(self, response_generator, mock_config):
        response_generator.generate = ResponseGenerator.generate
        retrieved_info = [
            {"source": "source1", "content": "content1"},
            {"source": "source2", "content": "content2"}
        ]
        response = response_generator.generate(
            response_generator, "What is artificial intelligence?", retrieved_info
        )
        mock_config.client.chat.assert_called_once()
        call_kwargs = mock_config.client.chat.call_args[1]
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0].role == "system"
        assert "What is artificial intelligence?" in call_kwargs["messages"][1].content
        assert "Source: source1" in call_kwargs["messages"][1].content
        assert call_kwargs["temperature"] == 0.1
        assert response == "Generated response from OpenAI"

    def test_generate_error(self, response_generator, mock_config):
        mock_config.client.chat.side_effect = Exception("API error")
        response_generator.generate = ResponseGenerator.generate
        with patch('agentic_rag.agents.response_generator.logger.error') as mock_logging:
            result = response_generator.generate(
                response_generator, "What is AI?", [{"source": "s1", "content": "c1"}]
            )
            mock_logging.assert_called_once()
            assert "error" in result.lower()
