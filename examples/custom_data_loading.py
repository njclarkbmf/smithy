import csv
import json
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


def load_csv_documents(
    csv_path: str,
    text_column: str = "text",
    id_column: str = "id",
    source_prefix: str = "csv",
) -> List[Document]:
    """
    Load documents from a CSV file.

    Args:
        csv_path: Path to the CSV file
        text_column: Name of the column containing the text
        id_column: Name of the column to use as document ID
        source_prefix: Prefix to use for the source field

    Returns:
        List of Document objects
    """
    documents = []

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                # Skip if required columns are missing
                if text_column not in row or not row[text_column]:
                    logger.warning(
                        f"Row {i} is missing the text column '{text_column}', skipping."
                    )
                    continue

                # Use the ID column if available, otherwise use row index
                doc_id = row.get(id_column, f"row_{i}")

                # Create metadata from all other columns
                metadata = {k: v for k, v in row.items() if k != text_column}

                # Create document
                doc = Document(
                    content=row[text_column],
                    source=f"{source_prefix}:{csv_path}",
                    doc_id=doc_id,
                    metadata=metadata,
                )
                documents.append(doc)

    except Exception as e:
        logger.error(f"Error loading CSV from {csv_path}: {str(e)}")

    return documents


def load_website_content(urls: List[str]) -> List[Document]:
    """
    Load content from multiple websites and create documents.
    Requires BeautifulSoup and requests:
    pip install beautifulsoup4 requests
    """
    try:
        from urllib.parse import urlparse

        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error(
            "Required packages not installed. Install with: pip install beautifulsoup4 requests"
        )
        return []

    documents = []

    for url in urls:
        try:
            # Send a request to the website
            response = requests.get(url, headers={"User-Agent": "AgenticRAG/1.0"})
            response.raise_for_status()

            # Parse the HTML content
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()

            # Get the title
            title = soup.title.string if soup.title else urlparse(url).netloc

            # Get the main content
            main_content = soup.find("main") or soup.find("article") or soup.body
            if main_content:
                text = main_content.get_text(separator="\n")
            else:
                text = soup.get_text(separator="\n")

            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)

            # Create document
            doc = Document(
                content=text,
                source=url,
                doc_id=f"web_{urlparse(url).netloc.replace('.', '_')}_{''.join(e for e in url if e.isalnum())[-8:]}",
                metadata={"title": title, "type": "webpage"},
            )
            documents.append(doc)

        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")

    return documents


def demo_custom_data_loading():
    """Demonstrate custom data loading methods."""
    # Initialize the RAG system
    config = Config(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o",
        provider="openai",
        db_path="./lancedb_custom",
    )

    rag = AgenticRAG(config)

    # Create example data files
    os.makedirs("./example_data", exist_ok=True)

    # Create example CSV
    with open("./example_data/articles.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "author", "content"])
        writer.writerow(
            [
                "article001",
                "Introduction to Artificial Intelligence",
                "John Smith",
                """Artificial Intelligence (AI) refers to the simulation of human intelligence in machines 
            that are programmed to think like humans and mimic their actions. The term may also be 
            applied to any machine that exhibits traits associated with a human mind such as learning 
            and problem-solving.""",
            ]
        )
        writer.writerow(
            [
                "article002",
                "The Future of Renewable Energy",
                "Jane Doe",
                """Renewable energy is energy that is collected from renewable resources, which are naturally 
            replenished on a human timescale, such as sunlight, wind, rain, tides, waves, and geothermal heat. 
            Renewable energy often provides energy in four important areas: electricity generation, 
            air and water heating/cooling, transportation, and rural (off-grid) energy services.""",
            ]
        )

    # Create example JSON
    with open("./example_data/books.json", "w") as f:
        json.dump(
            [
                {
                    "id": "book001",
                    "title": "Machine Learning Basics",
                    "author": "Alex Johnson",
                    "year": 2022,
                    "text": """Machine learning is a method of data analysis that automates analytical model building. 
                It is a branch of artificial intelligence based on the idea that systems can learn from data, 
                identify patterns and make decisions with minimal human intervention.""",
                },
                {
                    "id": "book002",
                    "title": "Quantum Computing for Beginners",
                    "author": "Lisa Chen",
                    "year": 2023,
                    "text": """Quantum computing is the study of a non-classical model of computation.
                Quantum computers are different from binary digital electronic computers based on transistors.
                Whereas common digital computing requires that the data be encoded into binary digits (bits),
                quantum computation uses quantum bits (qubits), which can be in superposition of states.""",
                },
            ],
            f,
            indent=2,
        )

    # Load from different sources
    print("\n=== Loading from CSV ===")
    csv_docs = load_csv_documents(
        "./example_data/articles.csv", text_column="content", id_column="id"
    )
    print(f"Loaded {len(csv_docs)} documents from CSV")

    print("\n=== Loading from JSON ===")
    json_docs = load_json_documents(
        "./example_data/books.json", text_key="text", id_key="id"
    )
    print(f"Loaded {len(json_docs)} documents from JSON")

    # URLs would typically be real websites, but for the example we'll use existing documents
    print("\n=== Loading website content (simulated) ===")
    # Here we would normally call load_website_content() with real URLs

    # Add all documents to the RAG system
    all_docs = csv_docs + json_docs
    print(f"\nAdding {len(all_docs)} documents to the knowledge base...")
    rag.add_documents(all_docs)

    # Try a query
    query = (
        "What is artificial intelligence and how does it relate to machine learning?"
    )
    print(f"\nUser Query: {query}")

    response, debug_info = rag.query(query)

    print(f"\nSubqueries: {', '.join(debug_info['subqueries'])}")
    print("\nRetrieved Information:")
    for item in debug_info["retrieved_info"]:
        print(f"- {item['source']} (similarity: {item.get('similarity', 'N/A'):.4f})")

    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    demo_custom_data_loading()


def load_json_documents(
    json_path: str,
    text_key: str = "text",
    id_key: str = "id",
    source_prefix: str = "json",
) -> List[Document]:
    """
    Load documents from a JSON file.

    Args:
        json_path: Path to the JSON file
        text_key: Key for the text content
        id_key: Key to use as document ID
        source_prefix: Prefix to use for the source field

    Returns:
        List of Document objects
    """
    documents = []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both array of objects and object of objects
        if isinstance(data, dict):
            items = data.values()
        elif isinstance(data, list):
            items = data
        else:
            logger.error(f"Unexpected JSON structure in {json_path}")
            return []

        for i, item in enumerate(items):
            # Skip if text key is missing
            if text_key not in item or not item[text_key]:
                logger.warning(
                    f"Item {i} is missing the text key '{text_key}', skipping."
                )
                continue

            # Use the ID key if available, otherwise use item index
            doc_id = item.get(id_key, f"item_{i}")

            # Create metadata from all other keys
            metadata = {k: v for k, v in item.items() if k != text_key}

            # Create document
            doc = Document(
                content=item[text_key],
                source=f"{source_prefix}:{json_path}",
                doc_id=doc_id,
                metadata=metadata,
            )
            documents.append(doc)

    except Exception as e:
        logger.error(f"Error loading JSON from {json_path}: {str(e)}")

    return documents


def load_pdf_documents(pdf_dir: str) -> List[Document]:
    """
    Load PDF documents from a directory.
    Requires PyPDF2 or pdfplumber:
    pip install PyPDF2
    """
    try:
        import PyPDF2
    except ImportError:
        logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
        return []

    from pathlib import Path

    documents = []

    for pdf_path in Path(pdf_dir).glob("*.pdf"):
        try:
            # Open the PDF file
            with open(pdf_path, "rb") as file:
                # Create PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)

                # Extract text from all pages
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n\n"

                # Create document object
                doc = Document(
                    content=text,
                    source=str(pdf_path),
                    doc_id=pdf_path.stem,
                    metadata={"type": "pdf", "pages": len(pdf_reader.pages)},
                )
                documents.append(doc)

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")

    return documents
