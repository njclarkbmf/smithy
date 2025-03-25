import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Implementation of reranking retrieved documents using cross-encoders.
    This technique improves retrieval quality by reranking initial results.

    Requires sentence-transformers:
    pip install sentence-transformers
    """

    def __init__(
        self, rag_system, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize the reranker.

        Args:
            rag_system: An instance of AgenticRAG
            model_name: Name of the cross-encoder model to use
        """
        self.rag_system = rag_system
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(self.model_name)
            self.model_loaded = True
        except ImportError:
            logger.warning(
                "sentence-transformers package not found. Install with: pip install sentence-transformers"
            )
            self.model_loaded = False

    def retrieve_with_reranking(
        self, queries: List[str], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and rerank documents for the given queries.

        Args:
            queries: List of query strings
            top_k: Number of top results to return

        Returns:
            List of reranked documents
        """
        if not self.model_loaded:
            logger.warning(
                "Cross-encoder model not loaded. Falling back to standard retrieval."
            )
            return self.rag_system.info_retriever.retrieve(queries, top_k)

        # Retrieve more documents than needed for reranking
        top_k_initial = (top_k or self.rag_system.config.top_k) * 3
        initial_results = self.rag_system.info_retriever.retrieve(
            queries, top_k_initial
        )

        # If no results were found, return empty list
        if not initial_results:
            return []

        # Prepare reranking inputs
        main_query = queries[0]
        rerank_pairs = [[main_query, result["content"]] for result in initial_results]

        # Rerank using cross-encoder
        rerank_scores = self.reranker.predict(rerank_pairs)

        # Add scores to results
        for i, result in enumerate(initial_results):
            result["rerank_score"] = float(rerank_scores[i])

        # Sort by reranking score
        reranked_results = sorted(
            initial_results, key=lambda x: x["rerank_score"], reverse=True
        )

        # Return top_k after reranking
        return reranked_results[: top_k or self.rag_system.config.top_k]

    def query(self, user_query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Perform a query using the reranking technique.

        Args:
            user_query: User's original query

        Returns:
            Generated response and debug info
        """
        # Plan the query
        subqueries = self.rag_system.query_planner.plan_query(user_query)

        # Retrieve and rerank
        retrieved_info = self.retrieve_with_reranking(subqueries)

        # Generate response
        response = self.rag_system.response_generator.generate(
            user_query, retrieved_info
        )

        # Prepare debug info
        debug_info = {
            "subqueries": subqueries,
            "retrieved_info": [
                {
                    "id": item["id"],
                    "source": item["source"],
                    "rerank_score": item.get("rerank_score"),
                    "similarity": item.get("similarity"),
                }
                for item in retrieved_info
            ],
        }

        return response, debug_info
