import os
import logging
from typing import List, Dict, Any, Tuple, Optional

from .config import Config
from .document import Document, DocumentProcessor
from .vectordb import VectorDBManager
from .agents.query_planner import QueryPlanner
from .agents.info_retriever import InfoRetriever
from .agents.response_generator import ResponseGenerator

logger = logging.getLogger(__name__)

class AgenticRAG:
    """Main class that orchestrates the Agentic RAG process."""
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.document_processor = DocumentProcessor(self.config)
        self.vector_db = VectorDBManager(self.config)
        self.query_planner = QueryPlanner(self.config)
        self.info_retriever = InfoRetriever(self.vector_db)
        self.response_generator = ResponseGenerator(self.config)
    
    def add_document(self, document: Document) -> int:
        """Add a single document to the knowledge base."""
        return self.vector_db.add_documents([document], self.document_processor)
    
    def add_documents(self, documents: List[Document]) -> int:
        """Add multiple documents to the knowledge base."""
        return self.vector_db.add_documents(documents, self.document_processor)
    
    def add_text_file(self, file_path: str) -> int:
        """Add a text file to the knowledge base."""
        document = self.document_processor.load_text_file(file_path)
        return self.add_document(document)
    
    def add_url(self, url: str) -> int:
        """Add content from a URL to the knowledge base."""
        document = self.document_processor.load_url(url)
        return self.add_document(document)
    
    def query(self, user_query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a user query through the Agentic RAG pipeline.
        Returns the generated response and debugging information.
        """
        # 1. Plan the query
        subqueries = self.query_planner.plan_query(user_query)
        
        # 2. Retrieve information
        retrieved_info = self.info_retriever.retrieve(subqueries, self.config.top_k)
        
        # 3. Generate response
        response = self.response_generator.generate(user_query, retrieved_info)
        
        # 4. Return response and debug info
        debug_info = {
            "subqueries": subqueries,
            "retrieved_info": [
                {"id": item["id"], "source": item["source"], "similarity": item.get("similarity")}
                for item in retrieved_info
            ]
        }
        
        return response, debug_info

# Simple demo function
def demo():
    """Run a demonstration of the Agentic RAG system."""
    # Initialize the system
    config = Config(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o",
        provider="openai",
        embedding_model="text-embedding-3-large",
        db_path="./lancedb_demo"
    )
    
    rag = AgenticRAG(config)
    
    # Add some example documents
    example_doc1 = Document(
        content="""
        Python is a high-level, interpreted programming language known for its readability and simplicity.
        It was created by Guido van Rossum and first released in 1991. Python supports multiple programming
        paradigms, including procedural, object-oriented, and functional programming. It has a comprehensive
        standard library and a large ecosystem of third-party packages for various applications.
        """,
        source="python_info.txt",
        doc_id="doc1"
    )
    
    example_doc2 = Document(
        content="""
        TensorFlow is an open-source machine learning framework developed by Google. It was released in 2015
        and has become one of the most popular frameworks for deep learning. TensorFlow provides a flexible
        ecosystem of tools, libraries, and community resources that lets researchers push the state-of-the-art
        in ML and developers easily build and deploy ML-powered applications.
        """,
        source="tensorflow_info.txt",
        doc_id="doc2"
    )
    
    print("Adding example documents...")
    rag.add_documents([example_doc1, example_doc2])
    
    # Try a query
    print("\nQuerying the system...")
    user_query = "What is Python and when was it created?"
    
    response, debug_info = rag.query(user_query)
    
    print(f"\nUser Query: {user_query}")
    print(f"\nSubqueries: {', '.join(debug_info['subqueries'])}")
    print("\nRetrieved Information:")
    for item in debug_info['retrieved_info']:
        print(f"- {item['source']} (similarity: {item.get('similarity', 'N/A'):.4f})")
    
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    demo()
