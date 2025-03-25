# Smithy: Agentic RAG Framework

A modular implementation of an Agentic Retrieval-Augmented Generation (RAG) framework with an emphasis on testability and extensibility.

## Features

- **Multi-provider Support**: Compatible with both OpenAI and Anthropic models
- **Vector Database Integration**: Uses LanceDB for efficient embedding storage and retrieval
- **Agent-based Architecture**: Separate agents for query planning, information retrieval, and response generation
- **Flexible Document Processing**: Support for text files, web content via URLs, and custom document formats
- **Configurable Parameters**: Easily adjust chunk size, overlap, temperature, and other settings
- **Advanced Techniques Ready**: Architecture designed to support:
  - Hypothetical Document Embeddings (HyDE)
  - Cross-encoder Reranking
  - Multi-query Fusion
  - Self-improving RAG (in development)
- **Evaluation Framework**: Tools for performance assessment and benchmarking
- **Docker Support**: Complete containerization for easy deployment

## Installation

### Method 1: Using the Makefile (Recommended)

The repository includes a Makefile with convenient commands:

```bash
# Clone the repository
git clone https://github.com/your-username/smithy.git
cd smithy

# Create a .env file with your API keys
cp .env-example .env
# Edit the .env file with your preferred text editor

# Install dependencies
make setup

# Run tests
make test

# Build the package
make build
```

### Method 2: Using Docker

The system includes complete Docker support for easy deployment:

```bash
# Clone the repository
git clone https://github.com/your-username/smithy.git
cd smithy

# Create a .env file with your API keys
cp .env-example .env
# Edit the .env file with your preferred text editor

# Build and run with docker-compose
docker-compose up -d

# Or use make to build and run the Docker container
make docker
```

### Method 3: Manual Installation

```bash
# Clone the repository
git clone https://github.com/your-username/smithy.git
cd smithy

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .[dev]

# Create a .env file with your API keys
cp .env-example .env
# Edit the .env file with your preferred text editor
```

## Setup Environment Variables

Create a `.env` file in your project root with your API keys:

```
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# LLM Configuration
LLM_PROVIDER=openai  # Options: openai, anthropic
LLM_MODEL=gpt-4o-mini     # For OpenAI: gpt-4o, gpt-4o-mini, etc.
                     # For Anthropic: claude-3-opus, claude-3-sonnet, etc.
EMBEDDING_MODEL=text-embedding-3-large
TEMPERATURE=0.1

# Vector Database Configuration
DB_PATH=/app/data/lancedb
CHUNK_SIZE=1500
CHUNK_OVERLAP=150
TOP_K=5

# Logging Configuration
LOG_LEVEL=INFO       # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Basic Usage

```python
from agentic_rag import Config, AgenticRAG
from agentic_rag.utils.env_loader import load_environment

# Load configuration from environment variables
config_dict = load_environment()
config = Config.from_dict(config_dict)

# Initialize the RAG system
rag = AgenticRAG(config)

# Add a text file to the knowledge base
rag.add_text_file("path/to/your/document.txt")

# Add content from a URL
rag.add_url("https://example.com/article")

# Query the system
response, debug_info = rag.query("What information can you provide about this topic?")
print(response)

# Examine debug information if needed
print(f"Subqueries: {debug_info['subqueries']}")
print(f"Retrieved chunks: {len(debug_info['retrieved_info'])}")
```

## Working with Documents Directly

```python
from agentic_rag import Config, Document, AgenticRAG

# Initialize the system
config = Config(
    api_key="your_api_key_here",  # Or loaded from environment
    model="gpt-4o",
    provider="openai",
    embedding_model="text-embedding-3-large"
)

rag = AgenticRAG(config)

# Create a custom document
doc = Document(
    content="This is a sample document about AI. It contains information about machine learning and neural networks.",
    source="custom_document.txt",
    metadata={"author": "Jane Doe", "topic": "AI"}
)

# Add the document
rag.add_document(doc)

# Add multiple documents at once
documents = [
    Document(content="Document 1 content", source="doc1.txt"),
    Document(content="Document 2 content", source="doc2.txt")
]
rag.add_documents(documents)

# Query across all documents
response, _ = rag.query("What do these documents say about AI?")
```

## Current Project Structure

```
smithy/                          # Repository root
├── agentic_rag/                 # Main package (to be renamed to smithy)
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration settings
│   ├── document.py              # Document classes and processing
│   ├── vectordb.py              # Vector database integration
│   ├── main.py                  # Main AgenticRAG class
│   ├── cli.py                   # Command-line interface
│   ├── agents/                  # Agent modules
│   │   ├── info_retriever.py    # Information retrieval agent
│   │   ├── query_planner.py     # Query planning agent
│   │   └── response_generator.py# Response generation agent
│   ├── utils/                   # Utility modules
│   │   ├── env_loader.py        # Environment variable loader
│   │   └── text_processing.py   # Text processing utilities
│   ├── evaluation/              # Evaluation framework
│   └── advanced/                # Advanced RAG techniques
│       ├── hyde.py              # Hypothetical Document Embeddings
│       ├── multi_query.py       # Multi-query fusion
│       ├── reranking.py         # Cross-encoder reranking
│       └── self_improving.py    # Self-improving RAG
│
├── examples/                    # Usage examples
├── tests/                       # Test suite
├── .env-example                 # Example environment file
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker Compose configuration
├── Makefile                     # Build and development commands
├── requirements.txt             # Package dependencies
├── setup.py                     # Package setup script
└── app.py                       # Web application entry point
```

> **Note**: The module is currently named "agentic_rag" in the codebase but is referred to as "smithy" in documentation. To fully rename the project, you would need to rename the "agentic_rag" directory to "smithy" and update all import statements in the code.

## Advanced Features

The system includes the foundation for several advanced RAG techniques:

1. **Hypothetical Document Embeddings (HyDE)**: Uses LLM-generated hypothetical documents to improve retrieval
2. **Cross-encoder Reranking**: Further refines retrieval results using more complex models
3. **Multi-query Fusion**: Generates multiple query variations to improve retrieval coverage
4. **Self-improving RAG**: Learning from user feedback to improve over time

Example (HyDE):
```python
from agentic_rag.advanced.hyde import HypotheticalDocumentEmbeddings

hyde = HypotheticalDocumentEmbeddings(rag)
response, debug_info = hyde.query("Your complex query here")
```

## Development Commands

The Makefile provides several convenient commands for development:

```bash
# Install dependencies
make setup

# Clean build files
make clean

# Run tests
make test

# Run linting
make lint

# Run test coverage
make coverage

# Build documentation
make docs

# Build Python package
make build

# Docker commands
make docker-build   # Build Docker image
make docker-run     # Run Docker container
make docker         # Build and run Docker
```

## Troubleshooting

### API Key Issues
- Ensure your API keys are correctly set in the `.env` file or as environment variables
- For tests, the framework uses mocks so real API keys are not required

### Vector Database
- If you encounter database errors, check the `DB_PATH` in your config
- Ensure the directory exists and is writable
- Try clearing the database directory and re-adding your documents

### Embedding Dimensions
- If you change embedding models, make sure to update the `embedding_dimensions` parameter in your config
- For OpenAI models, dimensions are:
  - text-embedding-3-large: 3072
  - text-embedding-3-small: 1536
  - text-embedding-ada-002: 1536

### Model Errors
- Check that the specified model is available in your API account
- Verify that the provider setting matches the API key you're using

### Docker Issues
- If Docker container fails to start, check the logs: `docker-compose logs`
- Ensure the volumes are properly mounted with the .env file
- The container uses a non-root user (appuser) for security

## License

MIT

## Contributors

- John Clark
