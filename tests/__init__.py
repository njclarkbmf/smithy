# Test package initialization

# Set up mock modules before any imports
import sys
from unittest.mock import MagicMock

# Mock imports that might cause issues
sys.modules['lancedb'] = MagicMock()
sys.modules['lancedb.pydantic'] = MagicMock()
sys.modules['lancedb.embeddings'] = MagicMock()
sys.modules['dashscope'] = MagicMock()
sys.modules['dashscope.Generation'] = MagicMock()
sys.modules['dashscope.TextEmbedding'] = MagicMock()
