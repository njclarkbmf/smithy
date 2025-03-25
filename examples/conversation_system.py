import logging
import os
import sys
from typing import Any, Dict, List

# Add parent directory to path to import agentic_rag
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_rag import AgenticRAG, Config, Document

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConversationalRAG:
    """A conversational interface for the Agentic RAG system."""

    def __init__(self, rag_system):
        """
        Initialize the conversational RAG system.

        Args:
            rag_system: An instance of AgenticRAG
        """
        self.rag = rag_system
        self.conversation_history = []
        self.max_history_tokens = 2000  # Approximate token limit for history

    def process_message(self, user_message: str) -> str:
        """
        Process a user message and maintain conversation history.

        Args:
            user_message: User's message

        Returns:
            Assistant's response
        """
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})

        # Check if it's a simple greeting or chitchat
        simple_responses = {
            "hello": "Hello! How can I help you today?",
            "hi": "Hi there! What can I assist you with?",
            "thank you": "You're welcome! Is there anything else you need help with?",
            "thanks": "You're welcome! Anything else I can help with?",
        }

        lower_message = user_message.lower()
        for key, response in simple_responses.items():
            if lower_message == key or lower_message.startswith(f"{key} "):
                self.conversation_history.append(
                    {"role": "assistant", "content": response}
                )
                return response

        # Prepare context from conversation history
        context = self._prepare_context()

        # Use RAG for more complex queries
        augmented_query = self._augment_query_with_context(user_message, context)
        rag_response, debug_info = self.rag.query(augmented_query)

        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": rag_response})

        # Truncate history if it gets too long
        self._truncate_history()

        return rag_response

    def _prepare_context(self) -> str:
        """Prepare context from conversation history."""
        if len(self.conversation_history) <= 1:
            return ""

        # Take the last few exchanges, skipping the current query
        recent_history = self.conversation_history[:-1][-6:]  # Last 3 exchanges (max)

        context_parts = []
        for item in recent_history:
            role = "User" if item["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {item['content']}")

        return "\n\n".join(context_parts)

    def _augment_query_with_context(self, query: str, context: str) -> str:
        """
        Augment the user query with conversation context.

        Args:
            query: Original user query
            context: Conversation context

        Returns:
            Augmented query
        """
        if not context:
            return query

        return f"""
        Previous conversation:
        {context}
        
        Current question: {query}
        
        Please answer the current question based on both your knowledge and the conversation context if relevant.
        """

    def _truncate_history(self):
        """Truncate history if it gets too long."""
        if len(self.conversation_history) < 5:
            return

        # Estimate tokens (rough approximation)
        total_chars = sum(len(item["content"]) for item in self.conversation_history)
        estimated_tokens = total_chars / 4  # Rough approximation

        # Truncate if needed
        while (
            estimated_tokens > self.max_history_tokens
            and len(self.conversation_history) > 2
        ):
            # Remove oldest exchange (2 messages)
            self.conversation_history.pop(0)
            if (
                self.conversation_history
                and self.conversation_history[0]["role"] == "assistant"
            ):
                self.conversation_history.pop(0)

            # Recalculate
            total_chars = sum(
                len(item["content"]) for item in self.conversation_history
            )
            estimated_tokens = total_chars / 4

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.conversation_history


def setup_test_data():
    """Create a test RAG system with sample data."""
    # Initialize the system
    config = Config(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o",
        provider="openai",
        embedding_model="text-embedding-3-large",
        db_path="./lancedb_conversation",
    )

    rag = AgenticRAG(config)

    # Add test documents
    docs = [
        Document(
            content="""
            The Python programming language was created by Guido van Rossum and was first released in 1991.
            Python is known for its readability and simple syntax that emphasizes code readability.
            It supports multiple programming paradigms, including procedural, object-oriented, and
            functional programming. Python is often described as a "batteries included" language due
            to its comprehensive standard library.
            """,
            source="python_info.txt",
            doc_id="python_info",
        ),
        Document(
            content="""
            JavaScript is a programming language that is one of the core technologies of the World Wide Web.
            It was created by Brendan Eich in 1995. JavaScript is high-level, often just-in-time compiled,
            and multi-paradigm. It has curly-bracket syntax, dynamic typing, prototype-based object-orientation,
            and first-class functions. Alongside HTML and CSS, JavaScript is one of the three core technologies
            of the World Wide Web.
            """,
            source="javascript_info.txt",
            doc_id="javascript_info",
        ),
        Document(
            content="""
            SQL (Structured Query Language) is a domain-specific language used in programming and designed
            for managing data held in a relational database management system. It is particularly useful
            in handling structured data, i.e., data incorporating relations among entities and variables.
            SQL was developed by IBM in the 1970s for use with System R.
            """,
            source="sql_info.txt",
            doc_id="sql_info",
        ),
        Document(
            content="""
            Machine Learning is a field of artificial intelligence that uses statistical techniques to give
            computer systems the ability to "learn" from data, without being explicitly programmed. The name
            Machine Learning was coined in 1959 by Arthur Samuel. Machine learning algorithms build a model
            based on sample data, known as "training data", in order to make predictions or decisions without
            being explicitly programmed to do so.
            """,
            source="ml_info.txt",
            doc_id="ml_info",
        ),
    ]

    print(f"Adding {len(docs)} test documents...")
    rag.add_documents(docs)

    return rag


def demo_conversation_system():
    """Demonstrate the Conversational RAG system."""
    print("\n=== Conversational RAG System Example ===")

    # Set up test data
    rag = setup_test_data()

    # Create ConversationalRAG instance
    conv_rag = ConversationalRAG(rag)

    # Simulate a conversation
    conversation = [
        "Hello there!",
        "Who created Python?",
        "When was it released?",
        "How does it compare to JavaScript?",
        "What about machine learning? Can Python be used for that?",
        "Thanks for the information!",
    ]

    for i, message in enumerate(conversation):
        print(f"\nUser: {message}")
        response = conv_rag.process_message(message)
        print(f"Assistant: {response}")

    # Print conversation history
    print("\nConversation History:")
    history = conv_rag.get_history()
    for item in history:
        role = "User" if item["role"] == "user" else "Assistant"
        print(
            f"{role}: {item['content'][:50]}..."
            if len(item["content"]) > 50
            else f"{role}: {item['content']}"
        )


if __name__ == "__main__":
    demo_conversation_system()
