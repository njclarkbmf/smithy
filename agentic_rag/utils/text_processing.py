import logging
import re
import string
from typing import Any, Dict, List, Optional

import nltk

# Ensure NLTK data is downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace, special characters, etc.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using NLTK.

    Args:
        text: Text to split

    Returns:
        List of sentences
    """
    return nltk.sent_tokenize(text)


def remove_stopwords(text: str, language: str = "english") -> str:
    """
    Remove stopwords from text.

    Args:
        text: Text to process
        language: Language of stopwords

    Returns:
        Text with stopwords removed
    """
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words(language))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]

    return " ".join(filtered_words)


def extract_keywords(
    text: str, top_n: int = 10, language: str = "english"
) -> List[str]:
    """
    Extract keywords from text using a simple frequency-based approach.

    Args:
        text: Text to analyze
        top_n: Number of top keywords to return
        language: Language of text

    Returns:
        List of top keywords
    """
    from collections import Counter

    from nltk.corpus import stopwords

    # Get stopwords
    stop_words = set(stopwords.words(language))

    # Tokenize and filter
    words = re.findall(r"\b\w+\b", text.lower())
    filtered_words = [
        word for word in words if word not in stop_words and len(word) > 2
    ]

    # Count and get top N
    word_counts = Counter(filtered_words)
    top_keywords = [word for word, _ in word_counts.most_common(top_n)]

    return top_keywords


def truncate_text(text: str, max_length: int, add_ellipsis: bool = True) -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        add_ellipsis: Whether to add "..." at the end

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    truncated = text[:max_length]

    # Try to truncate at word boundary
    last_space = truncated.rfind(" ")
    if (
        last_space > max_length * 0.8
    ):  # Only truncate at word boundary if not too far back
        truncated = truncated[:last_space]

    if add_ellipsis:
        truncated += "..."

    return truncated


def calculate_text_stats(text: str) -> Dict[str, Any]:
    """
    Calculate various statistics about the text.

    Args:
        text: Text to analyze

    Returns:
        Dictionary of statistics
    """
    sentences = split_into_sentences(text)
    words = re.findall(r"\b\w+\b", text)

    stats = {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_length": (
            sum(len(word) for word in words) / len(words) if words else 0
        ),
        "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
    }

    return stats


def detect_language(text: str) -> Optional[str]:
    """
    Detect the language of the text.

    Args:
        text: Text to analyze

    Returns:
        ISO language code or None if detection fails
    """
    try:
        from langdetect import detect

        return detect(text)
    except:
        logger.warning("Language detection failed. langdetect may not be installed.")
        return None


def summarize_text(text: str, ratio: float = 0.2) -> str:
    """
    Create a simple extractive summary of the text.

    Args:
        text: Text to summarize
        ratio: Ratio of the original text to keep (0.0-1.0)

    Returns:
        Summarized text
    """
    try:
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.summarizers.lsa import LsaSummarizer

        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()

        sentence_count = max(1, int(len(split_into_sentences(text)) * ratio))
        summary = summarizer(parser.document, sentence_count)

        return " ".join(str(sentence) for sentence in summary)
    except:
        logger.warning("Text summarization failed. sumy may not be installed.")

        # Fallback to a simple sentence extraction
        sentences = split_into_sentences(text)
        sentence_count = max(1, int(len(sentences) * ratio))

        return " ".join(sentences[:sentence_count])
