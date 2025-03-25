import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

# Import main components for easy access
from .config import Config
from .document import Document, DocumentProcessor
from .main import AgenticRAG

__version__ = "0.1.0"
__all__ = ["Config", "Document", "DocumentProcessor", "AgenticRAG"]
