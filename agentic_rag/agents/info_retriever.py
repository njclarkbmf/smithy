from typing import Any, Dict, List

import numpy as np


class InfoRetriever:
    """Retrieves information from the vector database."""

    def __init__(self, vector_db):
        self.vector_db = vector_db

    def retrieve(self, queries: List[str], top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a list of queries."""
        all_results = []
        seen_ids = set()

        for query in queries:
            results = self.vector_db.search(query, top_k)

            # Deduplicate results
            for result in results:
                if result["id"] not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result["id"])

        # Sort by relevance (assuming the first query is the main one)
        if queries and all_results:
            main_query_embedding = self.vector_db._get_embeddings([queries[0]])[0]

            # Calculate similarity with main query
            for result in all_results:
                embedding = result["embedding"]
                similarity = np.dot(main_query_embedding, embedding) / (
                    np.linalg.norm(main_query_embedding) * np.linalg.norm(embedding)
                )
                result["similarity"] = float(similarity)

            all_results.sort(key=lambda x: x["similarity"], reverse=True)

        return all_results
