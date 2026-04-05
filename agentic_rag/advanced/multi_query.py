import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class MultiQueryFusion:
    """
    Implementation of Multi-Query Fusion technique.
    This technique generates multiple diverse queries and combines their results.
    """

    def __init__(self, rag_system, num_queries: int = 5):
        """
        Initialize the Multi-Query Fusion system.

        Args:
            rag_system: An instance of AgenticRAG
            num_queries: Number of alternative queries to generate
        """
        self.rag_system = rag_system
        self.config = rag_system.config
        self.num_queries = num_queries

    def generate_alternative_queries(self, query: str) -> List[str]:
        """
        Generate alternative versions of the original query.

        Args:
            query: Original user query

        Returns:
            List of alternative queries
        """
        system_prompt = f"""
        You are a Query Diversification Agent. Your job is to take a user's original query and create 
        {self.num_queries} diverse alternative queries that explore different aspects, angles, or phrasings 
        of the same information need.
        
        Make sure the queries are:
        1. Diverse in their approach and wording
        2. Cover different aspects of the topic
        3. Use varied terminology and synonyms
        4. Consider different levels of specificity
        5. Explore related concepts that might yield relevant information
        
        Return ONLY the alternative queries, one per line, without any explanations or additional text.
        """

        try:
            from ..providers.model_router import ChatMessage

            response = self.config.client.chat(
                messages=[
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=query),
                ],
                temperature=0.7,
            )
            alt_queries = response.content.strip().split("\n")

            # Filter out empty queries
            alt_queries = [q.strip() for q in alt_queries if q.strip()]

            return alt_queries

        except Exception as e:
            logger.error(f"Error generating alternative queries: {str(e)}")
            return []

    def query(self, user_query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Perform a query using Multi-Query Fusion technique.

        Args:
            user_query: User's original query

        Returns:
            Generated response and debug info
        """
        # Generate alternative queries
        alt_queries = self.generate_alternative_queries(user_query)

        # Combine original query with alternatives
        all_queries = [user_query] + alt_queries

        # Determine how many results to retrieve per query
        top_k_per_query = max(1, self.rag_system.config.top_k // len(all_queries))

        # Retrieve results for all queries
        all_results = []
        seen_ids = set()
        query_results = {}

        for q in all_queries:
            results = self.rag_system.vector_db.search(q, top_k_per_query)
            query_results[q] = []

            # Deduplicate results
            for result in results:
                query_results[q].append(result["id"])

                if result["id"] not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result["id"])

        # Generate response based on all retrieved documents
        response = self.rag_system.response_generator.generate(user_query, all_results)

        # Prepare debug info
        debug_info = {
            "original_query": user_query,
            "alternative_queries": alt_queries,
            "query_results": query_results,
            "total_unique_results": len(all_results),
            "retrieved_info": [
                {
                    "id": item["id"],
                    "source": item["source"],
                    "similarity": item.get("similarity"),
                }
                for item in all_results
            ],
        }

        return response, debug_info
