import logging
import os
import sys

# Add parent directory to path to import agentic_rag
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_rag import AgenticRAG, Config, Document
from agentic_rag.advanced.hyde import HypotheticalDocumentEmbeddings
from agentic_rag.advanced.multi_query import MultiQueryFusion
from agentic_rag.advanced.reranking import CrossEncoderReranker
from agentic_rag.advanced.self_improving import SelfImprovingRAG

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_test_data():
    """Create a test RAG system with sample data."""
    # Initialize the system
    config = Config(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o",
        provider="openai",
        embedding_model="text-embedding-3-large",
        db_path="./lancedb_advanced",
    )

    rag = AgenticRAG(config)

    # Add test documents
    docs = [
        Document(
            content="""
            Artificial Intelligence (AI) is the simulation of human intelligence processes by machines,
            especially computer systems. These processes include learning (the acquisition of information
            and rules for using the information), reasoning (using rules to reach approximate or definite
            conclusions) and self-correction. Particular applications of AI include expert systems, speech
            recognition and machine vision.
            """,
            source="ai_definition.txt",
            doc_id="ai_def",
        ),
        Document(
            content="""
            Machine Learning is a subset of artificial intelligence (AI) that provides systems the ability
            to automatically learn and improve from experience without being explicitly programmed. Machine
            learning focuses on the development of computer programs that can access data and use it to learn
            for themselves. The process of learning begins with observations or data, such as examples, direct
            experience, or instruction, in order to look for patterns in data and make better decisions in the
            future based on the examples that we provide.
            """,
            source="machine_learning.txt",
            doc_id="ml_info",
        ),
        Document(
            content="""
            Deep Learning is a subset of machine learning that uses neural networks with many layers
            (hence "deep") to analyze various factors of data. Deep learning is a key technology behind
            driverless cars, enabling them to recognize a stop sign or distinguish a pedestrian from a
            lamppost. It is also used in voice control in consumer devices like phones, tablets, TVs,
            and hands-free speakers.
            """,
            source="deep_learning.txt",
            doc_id="dl_info",
        ),
        Document(
            content="""
            Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers
            understand, interpret and manipulate human language. NLP draws from many disciplines, including
            computer science and computational linguistics, in its pursuit to fill the gap between human
            communication and computer understanding. NLP enables computers to perform language-related tasks
            like translation, sentiment analysis, and speech recognition.
            """,
            source="nlp.txt",
            doc_id="nlp_info",
        ),
        Document(
            content="""
            Computer Vision is a field of artificial intelligence that trains computers to interpret and
            understand the visual world. Using digital images from cameras and videos and deep learning models,
            machines can accurately identify and classify objects — and then react to what they "see." Computer
            vision is used in industries ranging from energy and utilities to manufacturing and automotive.
            """,
            source="computer_vision.txt",
            doc_id="cv_info",
        ),
        Document(
            content="""
            Quantum computing is an area of computing focused on developing computer technology based on the
            principles of quantum theory. Quantum computers use quantum bits or qubits instead of traditional
            binary bits. Unlike normal computers that encode data in binary digits (0s and 1s), quantum computers
            use quantum bits that can exist in any superposition of values between 0 and 1.
            """,
            source="quantum_computing.txt",
            doc_id="quantum_info",
        ),
    ]

    print(f"Adding {len(docs)} test documents...")
    rag.add_documents(docs)

    return rag


def demo_hypothetical_document_embeddings():
    """Demonstrate the Hypothetical Document Embeddings (HyDE) technique."""
    print("\n=== Hypothetical Document Embeddings (HyDE) Example ===")

    # Set up test data
    rag = setup_test_data()

    # Create HyDE instance
    hyde = HypotheticalDocumentEmbeddings(rag)

    # Test query
    query = "How do computers process and understand human language?"

    print(f"Original Query: {query}\n")

    # Standard RAG approach
    print("Standard RAG:")
    std_response, std_debug = rag.query(query)

    print(f"Retrieved {len(std_debug['retrieved_info'])} documents")
    print(
        "Sources:", ", ".join([item["source"] for item in std_debug["retrieved_info"]])
    )
    print(f"Response: {std_response[:200]}...\n")

    # HyDE approach
    print("HyDE RAG:")
    hyde_response, hyde_debug = hyde.query(query)

    print(
        f"Generated a hypothetical document of length {len(hyde_debug['hypothetical_document'])}"
    )
    print(f"Retrieved {len(hyde_debug['retrieved_info'])} documents")
    print(
        "Sources:", ", ".join([item["source"] for item in hyde_debug["retrieved_info"]])
    )
    print(f"Response: {hyde_response[:200]}...\n")

    # Compare results
    print("Hypothetical Document Preview:")
    print(hyde_debug["hypothetical_document"][:300], "...\n")


def demo_cross_encoder_reranking():
    """Demonstrate cross-encoder reranking technique."""
    print("\n=== Cross-Encoder Reranking Example ===")

    try:
        # Check if sentence-transformers is installed
        import sentence_transformers

        has_sentence_transformers = True
    except ImportError:
        print("sentence-transformers not installed. Skipping reranking demo.")
        print("To run this demo, install with: pip install sentence-transformers")
        has_sentence_transformers = False
        return

    # Set up test data
    rag = setup_test_data()

    # Create reranker
    reranker = CrossEncoderReranker(rag)

    # Test query
    query = "What are the differences between machine learning and deep learning?"

    print(f"Query: {query}\n")

    # Standard RAG approach
    print("Standard RAG:")
    std_response, std_debug = rag.query(query)

    print(f"Retrieved {len(std_debug['retrieved_info'])} documents")
    print("Sources in order of retrieval:")
    for i, item in enumerate(std_debug["retrieved_info"]):
        print(
            f"  {i+1}. {item['source']} (similarity: {item.get('similarity', 'N/A'):.4f})"
        )
    print(f"Response: {std_response[:200]}...\n")

    # Reranking approach
    print("Cross-Encoder Reranking:")
    rerank_response, rerank_debug = reranker.query(query)

    print(f"Retrieved {len(rerank_debug['retrieved_info'])} documents")
    print("Sources in order of retrieval:")
    for i, item in enumerate(rerank_debug["retrieved_info"]):
        print(
            f"  {i+1}. {item['source']} (rerank_score: {item.get('rerank_score', 'N/A'):.4f})"
        )
    print(f"Response: {rerank_response[:200]}...\n")


def demo_multi_query_fusion():
    """Demonstrate Multi-Query Fusion technique."""
    print("\n=== Multi-Query Fusion Example ===")

    # Set up test data
    rag = setup_test_data()

    # Create Multi-Query Fusion instance
    mqf = MultiQueryFusion(rag)

    # Test query
    query = "How is artificial intelligence applied in understanding images and video?"

    print(f"Original Query: {query}\n")

    # Standard RAG approach
    print("Standard RAG:")
    std_response, std_debug = rag.query(query)

    print(f"Retrieved {len(std_debug['retrieved_info'])} documents")
    print(
        "Sources:", ", ".join([item["source"] for item in std_debug["retrieved_info"]])
    )
    print(f"Response: {std_response[:200]}...\n")

    # Multi-Query Fusion approach
    print("Multi-Query Fusion:")
    mqf_response, mqf_debug = mqf.query(query)

    print(f"Generated {len(mqf_debug['alternative_queries'])} alternative queries:")
    for i, alt_query in enumerate(mqf_debug["alternative_queries"]):
        print(f"  {i+1}. {alt_query}")

    print(f"\nRetrieved {mqf_debug['total_unique_results']} unique documents")
    print(
        "Sources:", ", ".join([item["source"] for item in mqf_debug["retrieved_info"]])
    )
    print(f"Response: {mqf_response[:200]}...\n")


def demo_self_improving_rag():
    """Demonstrate Self-Improving RAG system."""
    print("\n=== Self-Improving RAG Example ===")

    # Set up test data
    rag = setup_test_data()

    # Create Self-Improving RAG instance
    si_rag = SelfImprovingRAG(rag, feedback_db_path="./feedback_test.json")

    # Test queries
    queries = [
        "What is artificial intelligence?",
        "How does machine learning relate to AI?",
        "Explain deep learning in simple terms",
        "How is computer vision used in real life?",
        "What are the applications of NLP?",
    ]

    # Run queries and collect automatic feedback
    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: {query}")

        response, debug_info = si_rag.query(query)

        print(f"Retrieved {len(debug_info['retrieved_info'])} documents")
        sources = ", ".join([item["source"] for item in debug_info["retrieved_info"]])
        print(f"Sources: {sources}")
        print(f"Response: {response[:150]}...")

        # Simulate automatic evaluation
        rating = si_rag.automatic_evaluation(
            query, response, debug_info["retrieved_info"]
        )
        print(f"Automatic rating: {rating}/5")

    # Generate improvement report
    print("\nGenerating improvement report...")
    report = si_rag.generate_improvement_report()

    if report["status"] == "Report generated":
        print("\nImprovement Report Summary:")
        print(report["report"][:500], "...")
    else:
        print(f"\nStatus: {report['status']}")


if __name__ == "__main__":
    demo_hypothetical_document_embeddings()
    demo_cross_encoder_reranking()
    demo_multi_query_fusion()
    demo_self_improving_rag()
