import logging
from typing import Any, Dict, List, Tuple

from ..config import Config

logger = logging.getLogger(__name__)


class HypotheticalDocumentEmbeddings:
    """Implementation of Hypothetical Document Embeddings (HyDE) technique."""

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.config = rag_system.config

    def generate_hypothetical_document(self, query: str) -> str:
        """Generate a hypothetical document that would perfectly answer the query."""
        system_prompt = """
        You are an AI assistant that generates a hypothetical document that would perfectly answer the user's query.
        The document should be comprehensive, detailed, and directly address all aspects of the query.
        Write this document as if it were written by an expert on the subject.
        """
        try:
            from ..providers.model_router import ChatMessage

            response = self.config.client.chat(
                messages=[
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=query),
                ],
                temperature=0.5,
            )
            return response.content
        except Exception as e:
            logger.error(f"Error generating hypothetical document: {str(e)}")
            return query

    def query(self, user_query: str) -> Tuple[str, Dict[str, Any]]:
        """Perform a query using the HyDE technique."""
        hyde_doc = self.generate_hypothetical_document(user_query)
        retrieved_info = self.rag_system.info_retriever.retrieve(
            [hyde_doc], self.config.top_k
        )
        response = self.rag_system.response_generator.generate(user_query, retrieved_info)
        debug_info = {
            "original_query": user_query,
            "hypothetical_document": hyde_doc,
            "retrieved_info": [
                {"id": item["id"], "source": item["source"], "similarity": item.get("similarity")}
                for item in retrieved_info
            ],
        }
        return response, debug_info
