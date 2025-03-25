import logging
import os

import gradio as gr

from agentic_rag import AgenticRAG, Config, Document
from agentic_rag.advanced.hyde import HypotheticalDocumentEmbeddings
from agentic_rag.advanced.multi_query import MultiQueryFusion
from agentic_rag.utils.env_loader import load_environment

try:
    from agentic_rag.advanced.reranking import CrossEncoderReranker

    has_reranker = True
except ImportError:
    has_reranker = False

# Load environment variables
config_dict = load_environment()
config = Config.from_dict(config_dict)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize the RAG system
rag = AgenticRAG(config)

# Initialize advanced components
hyde = HypotheticalDocumentEmbeddings(rag)
multi_query = MultiQueryFusion(rag)
if has_reranker:
    reranker = CrossEncoderReranker(rag)


def add_document(file_content, file_name, source_type, document_text=None):
    """
    Add a document to the RAG system.

    Args:
        file_content: Content of the uploaded file (if any)
        file_name: Name of the uploaded file (if any)
        source_type: Type of the source (file, text, url)
        document_text: Text content (if source_type is 'text')

    Returns:
        Status message
    """
    try:
        if source_type == "file" and file_content is not None:
            # Save the file temporarily
            temp_path = f"./temp_{file_name}"
            with open(temp_path, "wb") as f:
                f.write(file_content)

            # Process file based on its extension
            if file_name.endswith((".txt", ".md")):
                num_chunks = rag.add_text_file(temp_path)
            elif file_name.endswith(".pdf"):
                from agentic_rag.document import Document, DocumentProcessor
                from examples.custom_data_loading import load_pdf_documents

                docs = load_pdf_documents(os.path.dirname(temp_path))
                num_chunks = rag.add_documents(docs)
            else:
                os.remove(temp_path)
                return f"Unsupported file type: {file_name.split('.')[-1]}"

            # Clean up
            os.remove(temp_path)
            return f"Added document '{file_name}' to the knowledge base ({num_chunks} chunks)"

        elif source_type == "text" and document_text:
            doc = Document(
                content=document_text,
                source="user_input",
                doc_id=f"text_{len(document_text)[:10]}",
            )
            num_chunks = rag.add_document(doc)
            return f"Added text document to the knowledge base ({num_chunks} chunks)"

        elif source_type == "url" and document_text:
            num_chunks = rag.add_url(document_text)
            return f"Added content from URL '{document_text}' to the knowledge base ({num_chunks} chunks)"

        else:
            return "No document provided or unsupported source type"

    except Exception as e:
        logger.error(f"Error adding document: {str(e)}")
        return f"Error adding document: {str(e)}"


def query_rag(query, method):
    """
    Query the RAG system with different methods.

    Args:
        query: User query
        method: RAG method to use (standard, hyde, reranker, multi-query)

    Returns:
        Response and debug info
    """
    try:
        if not query:
            return "Please enter a query.", ""

        if method == "standard":
            response, debug_info = rag.query(query)
        elif method == "hyde":
            response, debug_info = hyde.query(query)
        elif method == "reranker" and has_reranker:
            response, debug_info = reranker.query(query)
        elif method == "multi-query":
            response, debug_info = multi_query.query(query)
        else:
            response, debug_info = rag.query(query)

        # Format debug info for display
        debug_str = "Debug Information:\n\n"

        if "subqueries" in debug_info:
            debug_str += "Subqueries:\n"
            for i, subquery in enumerate(debug_info["subqueries"]):
                debug_str += f"{i+1}. {subquery}\n"
            debug_str += "\n"

        if "hypothetical_document" in debug_info:
            debug_str += "Hypothetical Document:\n"
            debug_str += f"{debug_info['hypothetical_document'][:300]}...\n\n"

        if "alternative_queries" in debug_info:
            debug_str += "Alternative Queries:\n"
            for i, alt_query in enumerate(debug_info["alternative_queries"]):
                debug_str += f"{i+1}. {alt_query}\n"
            debug_str += "\n"

        debug_str += "Retrieved Documents:\n"
        for i, doc in enumerate(debug_info.get("retrieved_info", [])):
            similarity = doc.get("similarity", "N/A")
            rerank_score = doc.get("rerank_score", "N/A")
            score = rerank_score if rerank_score != "N/A" else similarity

            debug_str += f"{i+1}. {doc['source']} (Score: {score})\n"

        return response, debug_str

    except Exception as e:
        logger.error(f"Error querying RAG: {str(e)}")
        return f"Error: {str(e)}", ""


# Create Gradio interface
with gr.Blocks(title="Agentic RAG System") as app:
    gr.Markdown("# Agentic RAG System")

    with gr.Tab("Add Documents"):
        with gr.Row():
            with gr.Column():
                source_type = gr.Radio(
                    ["file", "text", "url"], label="Source Type", value="file"
                )
                file_input = gr.File(label="Upload Document")
                text_input = gr.Textbox(
                    label="Text Content or URL",
                    placeholder="Enter document text or URL...",
                    lines=10,
                    visible=False,
                )
                add_btn = gr.Button("Add to Knowledge Base")

            with gr.Column():
                add_status = gr.Textbox(label="Status", lines=1)

        # Toggle input visibility based on source type
        def update_input_visibility(source):
            if source == "file":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        source_type.change(
            update_input_visibility,
            inputs=[source_type],
            outputs=[file_input, text_input],
        )

        # Add document button
        add_btn.click(
            add_document,
            inputs=[
                file_input,
                gr.Textbox(value=lambda: getattr(file_input, "name", "")),
                source_type,
                text_input,
            ],
            outputs=[add_status],
        )

    with gr.Tab("Query System"):
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Query", placeholder="Enter your query...", lines=3
                )
                method_select = gr.Radio(
                    ["standard", "hyde", "reranker", "multi-query"],
                    label="RAG Method",
                    value="standard",
                )
                query_btn = gr.Button("Submit Query")

            with gr.Column(scale=3):
                response_output = gr.Textbox(label="Response", lines=10)
                debug_output = gr.Textbox(label="Debug Information", lines=15)

        # Query button
        query_btn.click(
            query_rag,
            inputs=[query_input, method_select],
            outputs=[response_output, debug_output],
        )

# Launch the app
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
