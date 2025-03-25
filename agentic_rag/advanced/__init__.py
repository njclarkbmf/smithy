from .hyde import HypotheticalDocumentEmbeddings
from .multi_query import MultiQueryFusion
from .self_improving import FeedbackCollector, SelfImprovingRAG

# Only import reranking if sentence-transformers is installed
try:
    from .reranking import CrossEncoderReranker

    has_reranker = True
except ImportError:
    has_reranker = False
    CrossEncoderReranker = None

if has_reranker:
    __all__ = [
        "HypotheticalDocumentEmbeddings",
        "CrossEncoderReranker",
        "MultiQueryFusion",
        "SelfImprovingRAG",
        "FeedbackCollector",
    ]
else:
    __all__ = [
        "HypotheticalDocumentEmbeddings",
        "MultiQueryFusion",
        "SelfImprovingRAG",
        "FeedbackCollector",
    ]
