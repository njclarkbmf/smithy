import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generates responses using retrieved information."""

    def __init__(self, config):
        self.config = config

    def generate(self, user_query: str, retrieved_info: List[Dict[str, Any]]) -> str:
        """Generate a response based on retrieved information."""
        # Combine retrieved information
        context = "\n\n---\n\n".join(
            [
                f"Source: {item['source']}\n\n{item['content']}"
                for item in retrieved_info
            ]
        )

        system_prompt = """
        You are a helpful AI assistant that provides accurate, reliable information based on the provided context.
        
        Guidelines:
        1. Only use information from the provided context to answer the question.
        2. If the context doesn't contain enough information to answer fully, acknowledge the limitations.
        3. If you're unsure about something, be transparent about your uncertainty.
        4. Cite sources when providing specific information.
        5. Provide a complete and thorough answer to the question based on the available information.
        6. Never make up or hallucinate information that is not in the context.
        """

        user_message = f"""
        Question: {user_query}
        
        Context:
        {context}
        
        Based on the context provided, please answer the question thoroughly and accurately.
        """

        try:
            from ..providers.model_router import ChatMessage

            response = self.config.client.chat(
                messages=[
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=user_message),
                ],
                temperature=self.config.temperature,
            )
            return response.content

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I encountered an error while generating a response: {str(e)}"
