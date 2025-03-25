import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

from .advanced.hyde import HypotheticalDocumentEmbeddings
from .advanced.multi_query import MultiQueryFusion
from .config import Config
from .document import Document, DocumentProcessor
from .main import AgenticRAG
from .utils.env_loader import load_environment

# Try to import optional components
try:
    from .advanced.reranking import CrossEncoderReranker

    has_reranker = True
except ImportError:
    has_reranker = False

try:
    from .evaluation.benchmarks import QABenchmarkDataset, create_test_dataset
    from .evaluation.evaluator import RAGEvaluator

    has_evaluation = True
except ImportError:
    has_evaluation = False


def setup_logging(log_level: str) -> None:
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("agentic_rag.log"),
        ],
    )


def add_document_command(args):
    """Handle the add-document command."""
    # Load configuration
    config_dict = load_environment(args.env_file)
    config = Config.from_dict(config_dict)

    # Initialize the RAG system
    rag = AgenticRAG(config)

    # Handle different input types
    if args.file:
        if os.path.isfile(args.file):
            num_chunks = rag.add_text_file(args.file)
            print(f"Added document from '{args.file}' ({num_chunks} chunks)")
        else:
            print(f"Error: File '{args.file}' not found")
            return 1

    elif args.url:
        num_chunks = rag.add_url(args.url)
        print(f"Added content from URL '{args.url}' ({num_chunks} chunks)")

    elif args.text:
        doc = Document(
            content=args.text, source="cli_input", doc_id=f"cli_{len(args.text)[:10]}"
        )
        num_chunks = rag.add_document(doc)
        print(f"Added text document ({num_chunks} chunks)")

    elif args.directory:
        if os.path.isdir(args.directory):
            total_chunks = 0
            for root, _, files in os.walk(args.directory):
                for file in files:
                    if file.endswith((".txt", ".md")):
                        file_path = os.path.join(root, file)
                        try:
                            num_chunks = rag.add_text_file(file_path)
                            total_chunks += num_chunks
                            print(
                                f"Added document from '{file_path}' ({num_chunks} chunks)"
                            )
                        except Exception as e:
                            print(f"Error processing '{file_path}': {str(e)}")

            print(
                f"Added {total_chunks} total chunks from directory '{args.directory}'"
            )
        else:
            print(f"Error: Directory '{args.directory}' not found")
            return 1

    else:
        print(
            "Error: No input provided. Use one of: --file, --url, --text, or --directory"
        )
        return 1

    return 0


def query_command(args):
    """Handle the query command."""
    # Load configuration
    config_dict = load_environment(args.env_file)
    config = Config.from_dict(config_dict)

    # Initialize the RAG system
    rag = AgenticRAG(config)

    # Initialize the appropriate RAG method
    if args.method == "standard":
        query_system = rag
    elif args.method == "hyde":
        query_system = HypotheticalDocumentEmbeddings(rag)
    elif args.method == "reranker":
        if has_reranker:
            query_system = CrossEncoderReranker(rag)
        else:
            print(
                "Error: sentence-transformers not installed. Cannot use reranker method."
            )
            return 1
    elif args.method == "multi-query":
        query_system = MultiQueryFusion(rag)
    else:
        print(f"Error: Unknown method '{args.method}'")
        return 1

    # Process the query
    response, debug_info = query_system.query(args.query)

    # Print the response
    print("\n=== Query ===")
    print(args.query)

    print("\n=== Response ===")
    print(response)

    # Print debug info if verbose flag is set
    if args.verbose:
        print("\n=== Debug Info ===")

        if "subqueries" in debug_info:
            print("\nSubqueries:")
            for i, subquery in enumerate(debug_info["subqueries"]):
                print(f"{i+1}. {subquery}")

        if "hypothetical_document" in debug_info:
            print("\nHypothetical Document:")
            print(debug_info["hypothetical_document"][:300] + "...")

        if "alternative_queries" in debug_info:
            print("\nAlternative Queries:")
            for i, alt_query in enumerate(debug_info["alternative_queries"]):
                print(f"{i+1}. {alt_query}")

        print("\nRetrieved Documents:")
        for i, doc in enumerate(debug_info.get("retrieved_info", [])):
            similarity = doc.get("similarity", "N/A")
            rerank_score = doc.get("rerank_score", "N/A")
            score = rerank_score if rerank_score != "N/A" else similarity

            print(f"{i+1}. {doc['source']} (Score: {score})")

    # Save response to file if specified
    if args.output:
        with open(args.output, "w") as f:
            f.write(response)
        print(f"\nResponse saved to {args.output}")

    return 0


def evaluate_command(args):
    """Handle the evaluate command."""
    if not has_evaluation:
        print(
            "Error: Evaluation components not available. Install with 'pip install agentic-rag[eval]'"
        )
        return 1

    # Load configuration
    config_dict = load_environment(args.env_file)
    config = Config.from_dict(config_dict)

    # Initialize the RAG system
    rag = AgenticRAG(config)

    # Load or create the test dataset
    if args.dataset:
        # Load from file
        if not os.path.exists(args.dataset):
            print(f"Error: Dataset file '{args.dataset}' not found")
            return 1

        dataset = QABenchmarkDataset("cli_dataset", args.dataset)
        questions = dataset.load()
    else:
        # Create a test dataset
        print(f"Creating a test dataset with {args.num_questions} questions")
        questions = create_test_dataset(args.num_questions)

    # Initialize the evaluator
    evaluator = RAGEvaluator(rag)

    # Run the evaluation
    print(f"Evaluating {len(questions)} questions...")
    results_df = evaluator.evaluate_questions(
        questions,
        use_llm_eval=args.llm_eval,
        save_results=True,
        output_path=args.output_results,
    )

    # Generate the report if requested
    if args.generate_report:
        report_path = args.output_report or "evaluation_report.md"
        evaluator.generate_report(report_path)
        print(f"Evaluation report saved to {report_path}")

    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total Questions: {len(questions)}")

    # Calculate average metrics
    avg_metrics = {
        "ROUGE-1": results_df["rouge1"].mean(),
        "ROUGE-2": results_df["rouge2"].mean(),
        "ROUGE-L": results_df["rougeL"].mean(),
        "Semantic Similarity": results_df["semantic_similarity"].mean(),
        "Overall Score": results_df["overall_score"].mean(),
    }

    # Print average metrics
    for metric, score in avg_metrics.items():
        print(f"{metric}: {score:.4f}")

    return 0


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Agentic RAG System")
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Add document command
    add_doc_parser = subparsers.add_parser(
        "add-document", help="Add a document to the knowledge base"
    )
    add_doc_group = add_doc_parser.add_mutually_exclusive_group(required=True)
    add_doc_group.add_argument("--file", help="Path to a text file")
    add_doc_group.add_argument("--url", help="URL to fetch content from")
    add_doc_group.add_argument("--text", help="Text content")
    add_doc_group.add_argument("--directory", help="Directory containing text files")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("query", help="Query to process")
    query_parser.add_argument(
        "--method",
        choices=["standard", "hyde", "reranker", "multi-query"],
        default="standard",
        help="RAG method to use",
    )
    query_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print debug information"
    )
    query_parser.add_argument("--output", "-o", help="Save response to file")

    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the RAG system")
    evaluate_parser.add_argument(
        "--dataset", help="Path to evaluation dataset (JSON or CSV)"
    )
    evaluate_parser.add_argument(
        "--num-questions",
        type=int,
        default=5,
        help="Number of test questions to generate (if no dataset)",
    )
    evaluate_parser.add_argument(
        "--llm-eval",
        action="store_true",
        help="Use LLM for evaluation (can be expensive)",
    )
    evaluate_parser.add_argument(
        "--generate-report", action="store_true", help="Generate an evaluation report"
    )
    evaluate_parser.add_argument(
        "--output-results",
        default="evaluation_results.json",
        help="Path to save evaluation results",
    )
    evaluate_parser.add_argument(
        "--output-report", help="Path to save evaluation report"
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)

    # Handle commands
    if args.command == "add-document":
        return add_document_command(args)
    elif args.command == "query":
        return query_command(args)
    elif args.command == "evaluate":
        return evaluate_command(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
