import os
import uuid
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Document:
    """Represents a document with its metadata and content."""

    def __init__(
        self,
        content: str,
        source: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.content = content
        self.source = source
        self.doc_id = doc_id or str(uuid.uuid4())
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(id={self.doc_id}, source={self.source}, content_length={len(self.content)})"


class DocumentProcessor:
    """Handles document chunking and preparation for embedding."""

    def __init__(self, config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def process_document(self, document: Document) -> List[Dict[str, Any]]:
        """Process a document into chunks ready for embedding."""
        chunks = self.text_splitter.split_text(document.content)

        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_doc = {
                "id": f"{document.doc_id}_{i}",
                "doc_id": document.doc_id,
                "source": document.source,
                "content": chunk,
                "chunk_index": i,
                "metadata": document.metadata,
            }
            processed_chunks.append(chunk_doc)

        return processed_chunks

    def load_text_file(self, file_path: str) -> Document:
        """Load a text file into a Document object."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        doc_id = os.path.basename(file_path)
        return Document(content=content, source=file_path, doc_id=doc_id)

    def load_url(self, url: str) -> Document:
        """Load content from a URL into a Document object."""
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        # Extract just the text content
        for script in soup(["script", "style"]):
            script.extract()
        content = soup.get_text(separator="\n")

        return Document(content=content, source=url)
