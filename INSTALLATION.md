# Smithy Installation and Usage Guide

## Installation Options

### Method 1: Using the Makefile (Recommended)

The repository includes a comprehensive Makefile that simplifies installation and common tasks:

```bash
# Clone the repository
git clone https://github.com/your-username/smithy.git
cd smithy

# Install dependencies
make setup

# Run tests
make test

# Build the package
make build
```

Available Makefile commands:
```
setup        Install dependencies
clean        Clean build files
test         Run tests
lint         Run code linting
coverage     Run test coverage
docs         Build documentation
build        Build Python package
docker       Build Docker image and run container
docker-build Build Docker image
docker-run   Run Docker container
```

### Method 2: Docker Installation

Smithy includes complete Docker support for easy setup and isolation:

1. Clone the repository:
```bash
git clone https://github.com/your-username/smithy.git
cd smithy
```

2. Create a `.env` file with your API keys:
```bash
cp .env-example .env
# Edit the .env file with your preferred text editor
```

3. Build and run with docker-compose:
```bash
docker-compose up -d
```

The Docker container:
- Exposes port 7860 for the web interface
- Mounts a data volume for persistent storage at `/app/data`
- Uses your `.env` file for configuration

You can also use the Makefile for Docker operations:
```bash
make docker-build   # Build Docker image
make docker-run     # Run Docker container
make docker         # Build and run Docker
```

### Method 3: Manual Development Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/smithy.git
cd smithy
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Method 4: Using Pre-built Package

If you've already built the package:

```bash
pip install dist/agentic_rag-0.1.0-py3-none-any.whl
```

> **Note**: The wheel file is currently named "agentic_rag-0.1.0-py3-none-any.whl". After renaming the project to "smithy", future builds will generate "smithy-0.1.0-py3-none-any.whl".

## Current Package Structure

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
│
├── examples/                    # Usage examples
├── tests/                       # Test suite
├── app.py                       # Web application entry point
└── setup.py                     # Package setup script
```

## Required Dependencies

The core dependencies are:
- `openai`: For OpenAI model access
- `anthropic`: For Anthropic model access
- `lancedb`: For vector database functionality
- `numpy`: For numerical operations
- `pandas`: For data handling
- `bs4` (BeautifulSoup): For web scraping
- `requests`: For HTTP requests
- `python-dotenv`: For environment variable loading

Development dependencies include:
- `pytest`: For testing
- `pytest-cov`: For test coverage
- `black`: For code formatting
- `isort`: For import sorting
- `flake8`: For linting

Optional dependencies (installed in Docker):
- `PyPDF2`: For PDF processing
- `sentence-transformers`: For reranking
- `rouge-score`: For evaluation metrics

## Environment Setup

Create a `.env` file with your API keys and configuration settings:

```
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# LLM Configuration
LLM_PROVIDER=openai  # Options: openai, anthropic
LLM_MODEL=gpt-4o-mini  # Current default
EMBEDDING_MODEL=text-embedding-3-large
TEMPERATURE=0.1

# Vector Database Configuration
DB_PATH=/app/data/lancedb
CHUNK_SIZE=1500
CHUNK_OVERLAP=150
TOP_K=5

# Logging Configuration
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Docker Configuration

Smithy includes a complete Docker setup with:

1. **Dockerfile**: Based on Python 3.10-slim with all dependencies pre-installed
2. **docker-compose.yml**: Configures services and volumes
3. **Non-root user**: Runs as `appuser` for enhanced security

The Docker container:
- Uses Python 3.10
- Installs all core and optional dependencies
- Creates necessary directories
- Runs as a non-root user for security
- Exposes port 7860 for web interface access
- Mounts volumes for data persistence and configuration

### Docker Volumes:
- `./data:/app/data`: Persistently stores vector database and other data
- `./.env:/app/.env`: Mounts your environment configuration

## Detailed Usage Examples

### Basic RAG Workflow

```python
# Import using the current package name
from agentic_rag import Config, AgenticRAG
from agentic_rag.utils.env_loader import load_environment

# Initialize from environment variables
config_dict = load_environment()
config = Config.from_dict(config_dict)
rag = AgenticRAG(config)

# Add content (multiple methods available)
rag.add_text_file("data/sample.txt")  # From text file
rag.add_url("https://example.com")  # From a webpage

# Make a query
response, debug_info = rag.query("What is the main topic of this content?")
print(response)
```

### Using Custom Configuration

```python
from agentic_rag import Config, AgenticRAG

# Create configuration with custom parameters
config = Config(
    api_key="your_api_key",  # Will default to env vars if not provided
    provider="openai",  # or "anthropic"
    model="gpt-4o",
    embedding_model="text-embedding-3-large",
    embedding_dimensions=3072,  # Important to match the model
    db_path="./my_vector_db",
    chunk_size=2000,  # Larger chunks
    chunk_overlap=200,  # More overlap
    temperature=0.3,  # More creative responses
    top_k=7  # Retrieve more chunks
)

rag = AgenticRAG(config)
```

### Understanding the Information Retrieval Process

The retrieval process works as follows:

1. **Query Planning**: The query planner breaks down complex queries into simpler subqueries
2. **Information Retrieval**: Each subquery is used to search the vector database
3. **Response Generation**: Retrieved information chunks are used to generate a response

You can examine this process through the debug information:

```python
response, debug_info = rag.query("Tell me about machine learning applications")

# Examine the planning process
print(f"Original query: Tell me about machine learning applications")
print(f"Planned subqueries: {debug_info['subqueries']}")

# Examine retrieved information
print(f"Number of relevant chunks found: {len(debug_info['retrieved_info'])}")
for i, chunk in enumerate(debug_info['retrieved_info']):
    print(f"Chunk {i+1} from {chunk['source']} (similarity: {chunk['similarity']:.2f})")
    print(f"Content: {chunk['content'][:100]}...")
```

## Project Renaming Information

If you want to rename the project from "agentic_rag" to "smithy", here are the steps:

1. **Rename the directory**: 
   ```bash
   mv agentic_rag smithy
   ```

2. **Update import statements** throughout the codebase:
   ```bash
   # You can use a tool like grep to find all imports
   grep -r "from agentic_rag" .
   grep -r "import agentic_rag" .
   ```

3. **Update setup.py** to use the new package name

4. **Update Dockerfile and docker-compose.yml** if they reference the package name

Until the directory is renamed, continue using the original imports:
```python
from agentic_rag import Config, AgenticRAG
from agentic_rag.document import Document
```

## Working with Documents

The system uses a Document class to represent content:

```python
from agentic_rag.document import Document, DocumentProcessor
from agentic_rag import Config

# Create a configuration
config = Config()

# Create a document processor
processor = DocumentProcessor(config)

# Create a document
document = Document(
    content="This is the content of my document.",
    source="my_document.txt",
    metadata={"author": "John Doe", "date": "2025-03-25"}
)

# Process document into chunks
chunks = processor.process_document(document)
print(f"Document split into {len(chunks)} chunks")
```

## Working with the Vector Database

```python
from agentic_rag import Config
from agentic_rag.vectordb import VectorDBManager
from agentic_rag.document import Document, DocumentProcessor

# Setup
config = Config()
vectordb = VectorDBManager(config)
processor = DocumentProcessor(config)

# Add documents
document = Document(content="Sample content", source="sample.txt")
chunks = processor.process_document(document)
vectordb.add_documents([document], processor)

# Search
results = vectordb.search("sample query", top_k=3)
for result in results:
    print(f"Chunk {result['id']} - Similarity: {result['similarity']}")
    print(result['content'])
```

## Troubleshooting

### Common Issues and Solutions

#### API Key Errors

**Problem**: `OpenAIError: The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable`

**Solutions**:
1. Check that your `.env` file contains the correct API key
2. Ensure the `.env` file is in the correct directory
3. Try setting the environment variable directly:
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```
4. When running tests, mocks are used so real API keys aren't needed

#### Vector Database Errors

**Problem**: `Error connecting to LanceDB database`

**Solutions**:
1. Ensure the DB_PATH directory exists and is writable
2. Check if you have sufficient permissions for the directory
3. Try using an absolute path instead of a relative path
4. If the database is corrupted, remove the directory and recreate it

#### Docker-specific Issues

**Problem**: Docker container fails to start or runs into errors

**Solutions**:
1. Check container logs: `docker-compose logs`
2. Verify that your `.env` file is properly mounted
3. Make sure the data directory is writable by the container
4. Try rebuilding the image: `make docker-build`
5. Check that the port 7860 is not already in use on your host machine

#### Provider Errors

**Problem**: `ValueError: Unsupported provider: ...`

**Solution**: Ensure the provider is set to either "openai" or "anthropic"

#### Model Not Found Errors

**Problem**: Model-related errors from the API provider

**Solutions**:
1. Verify the model name is correct (e.g., "gpt-4o", "gpt-4o-mini", "claude-3-opus")
2. Ensure your API key has access to the specified model
3. Check if the model has been deprecated or renamed

#### Embedding Dimension Mismatch

**Problem**: Dimension mismatch errors with embeddings

**Solution**: Ensure your `embedding_dimensions` setting matches the model:
- text-embedding-3-large: 3072 dimensions
- text-embedding-3-small: 1536 dimensions
- text-embedding-ada-002: 1536 dimensions

## Advanced Usage: Testing

Smithy includes a comprehensive test suite using pytest:

```bash
# Run basic tests
pytest

# Run tests with coverage
pytest --cov=agentic_rag  # Use the current module name

# Generate a coverage report
make coverage

# Run linting
make lint
```

When writing your own tests, you can use the built-in mocks and fixtures:

```python
def test_my_function(mock_config, mock_vector_db):
    # These fixtures are provided in conftest.py
    # They mock external dependencies like OpenAI API
    # See tests/conftest.py for details
    ...
```

## Configuration Reference

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| api_key | API key for the provider | From env var | Any valid API key |
| provider | LLM provider | "openai" | "openai", "anthropic" |
| model | LLM model name | "gpt-4o-mini" | Any available model |
| embedding_model | Embedding model | "text-embedding-3-large" | OpenAI embedding models |
| embedding_dimensions | Dimensions of embeddings | 3072 | Depends on the model |
| db_path | Path to LanceDB database | "/app/data/lancedb" | Any valid path |
| chunk_size | Size of document chunks | 1500 | Recommended: 1000-2000 |
| chunk_overlap | Overlap between chunks | 150 | Recommended: 10-20% of chunk_size |
| temperature | LLM temperature | 0.1 | 0.0-1.0 |
| top_k | Number of chunks to retrieve | 5 | Recommended: 3-10 |
