import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Calculates various metrics for evaluating RAG system performance."""

    def __init__(self):
        # Initialize ROUGE metrics if available
        try:
            from rouge_score import rouge_scorer

            self.rouge_scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
            self.has_rouge = True
        except ImportError:
            logger.warning(
                "rouge_score not installed. Install with: pip install rouge-score"
            )
            self.has_rouge = False

    def calculate_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        if not self.has_rouge:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        scores = self.rouge_scorer.score(reference, prediction)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }

    def calculate_retrieval_precision(
        self, retrieved_docs: List[Dict], relevant_docs: List[str]
    ) -> float:
        """
        Calculate precision of retrieved documents.

        Args:
            retrieved_docs: List of retrieved documents
            relevant_docs: List of relevant document IDs

        Returns:
            Precision score
        """
        if not retrieved_docs or not relevant_docs:
            return 0.0

        # Count relevant documents that were retrieved
        relevant_retrieved = sum(
            1 for doc in retrieved_docs if doc.get("doc_id") in relevant_docs
        )

        # Calculate precision
        return relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0.0

    def calculate_retrieval_recall(
        self, retrieved_docs: List[Dict], relevant_docs: List[str]
    ) -> float:
        """
        Calculate recall of retrieved documents.

        Args:
            retrieved_docs: List of retrieved documents
            relevant_docs: List of relevant document IDs

        Returns:
            Recall score
        """
        if not retrieved_docs or not relevant_docs:
            return 0.0

        # Count relevant documents that were retrieved
        retrieved_doc_ids = [doc.get("doc_id") for doc in retrieved_docs]
        relevant_retrieved = sum(
            1 for doc_id in relevant_docs if doc_id in retrieved_doc_ids
        )

        # Calculate recall
        return relevant_retrieved / len(relevant_docs) if relevant_docs else 0.0

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        Requires sentence-transformers package.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Semantic similarity score (0-1)
        """
        try:
            from sentence_transformers import SentenceTransformer

            # Load model (only once)
            if not hasattr(self, "similarity_model"):
                self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

            # Calculate embeddings
            embedding1 = self.similarity_model.encode(text1, convert_to_tensor=True)
            embedding2 = self.similarity_model.encode(text2, convert_to_tensor=True)

            # Calculate cosine similarity
            from torch.nn.functional import cosine_similarity

            return cosine_similarity(
                embedding1.unsqueeze(0), embedding2.unsqueeze(0)
            ).item()

        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0

    def calculate_llm_evaluation(
        self, rag_system, prediction: str, reference: str, question: str
    ) -> Dict[str, Any]:
        """
        Use the LLM to evaluate the quality of the response.

        Args:
            rag_system: The RAG system (used for LLM access)
            prediction: Generated response
            reference: Reference answer
            question: Original question

        Returns:
            Dictionary with evaluation scores
        """
        eval_prompt = f"""
        You are an expert evaluator for question-answering systems. Please evaluate the following response to a question.
        
        Question: {question}
        
        Reference Answer: {reference}
        
        System Response: {prediction}
        
        Please rate the system response on the following criteria on a scale of 1-10:
        
        1. Relevance: How relevant is the response to the question? (1 = completely irrelevant, 10 = perfectly relevant)
        2. Completeness: How completely does the response answer the question? (1 = completely incomplete, 10 = perfectly complete)
        3. Accuracy: How factually accurate is the response compared to the reference? (1 = completely inaccurate, 10 = perfectly accurate)
        4. Conciseness: How concise is the response? (1 = too verbose or too brief, 10 = optimal length)
        
        For each criterion, provide a score and a brief justification. 
        Finally, provide an overall score from 1-10 and a brief summary of your evaluation.
        
        Format your response as a JSON object with the following structure:
        {
            "relevance": {"score": X, "justification": "..."},
            "completeness": {"score": X, "justification": "..."},
            "accuracy": {"score": X, "justification": "..."},
            "conciseness": {"score": X, "justification": "..."},
            "overall": {"score": X, "summary": "..."}
        }
        """

        try:
            if rag_system.config.provider == "openai":
                response = rag_system.config.client.chat.completions.create(
                    model=rag_system.config.model,
                    messages=[{"role": "user", "content": eval_prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                eval_result = json.loads(response.choices[0].message.content)

            elif rag_system.config.provider == "anthropic":
                response = rag_system.config.client.messages.create(
                    model=rag_system.config.model,
                    messages=[{"role": "user", "content": eval_prompt}],
                    temperature=0.0,
                )
                # Extract JSON from the response
                result_text = response.content[0].text
                eval_result = json.loads(result_text)

            return eval_result

        except Exception as e:
            logger.error(f"Error in LLM evaluation: {str(e)}")
            return {
                "error": str(e),
                "overall": {"score": 5, "summary": "Evaluation failed"},
            }

    def calculate_faithfulness(
        self, prediction: str, retrieved_docs: List[Dict]
    ) -> Dict[str, Any]:
        """
        Evaluate how faithful the response is to the retrieved documents.

        Args:
            prediction: Generated response
            retrieved_docs: List of retrieved documents

        Returns:
            Dictionary with faithfulness scores
        """
        # Combine all retrieved documents into a single context
        context = " ".join([doc.get("content", "") for doc in retrieved_docs])

        # Calculate semantic similarity
        similarity = self.calculate_semantic_similarity(prediction, context)

        # Simple hallucination check - if the response mentions concepts not in the context
        # This is a basic check and would need to be improved for a production system
        hallucination_score = 0.0

        try:
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize

            # Extract keywords from prediction and context
            pred_words = set(word_tokenize(prediction.lower()))
            context_words = set(word_tokenize(context.lower()))

            # Remove stopwords
            stop_words = set(stopwords.words("english"))
            pred_keywords = [
                w
                for w in pred_words
                if w not in stop_words and w.isalpha() and len(w) > 3
            ]
            context_keywords = [
                w
                for w in context_words
                if w not in stop_words and w.isalpha() and len(w) > 3
            ]

            # Calculate what percentage of prediction keywords are in context
            if pred_keywords:
                hallucination_score = len(
                    [w for w in pred_keywords if w in context_keywords]
                ) / len(pred_keywords)
        except Exception as e:
            logger.warning(f"Error calculating hallucination score: {str(e)}")

        return {
            "semantic_similarity": similarity,
            "hallucination_score": hallucination_score,
            "faithfulness_score": (similarity + hallucination_score) / 2,
        }
