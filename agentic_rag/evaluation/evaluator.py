import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from .metrics import EvaluationMetrics

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluates RAG system performance on a set of test questions."""

    def __init__(self, rag_system):
        """
        Initialize the evaluator.

        Args:
            rag_system: An instance of AgenticRAG
        """
        self.rag = rag_system
        self.metrics = EvaluationMetrics()
        self.results = []

    def evaluate_questions(
        self,
        questions: List[Dict[str, str]],
        use_llm_eval: bool = False,
        save_results: bool = True,
        output_path: str = "./evaluation_results.json",
    ) -> pd.DataFrame:
        """
        Evaluate the RAG system on a set of questions.

        Args:
            questions: List of dictionaries with 'question' and 'answer' keys
            use_llm_eval: Whether to use LLM for evaluation (can be expensive)
            save_results: Whether to save detailed results to a file
            output_path: Path to save results

        Returns:
            DataFrame with evaluation results
        """
        results = []

        for i, item in enumerate(tqdm(questions, desc="Evaluating questions")):
            question = item["question"]
            reference = item["answer"]

            # Optional relevant document IDs for retrieval evaluation
            relevant_docs = item.get("relevant_docs", [])

            logger.info(
                f"Evaluating question {i+1}/{len(questions)}: {question[:50]}..."
            )

            # Get RAG response
            response, debug_info = self.rag.query(question)

            # Calculate metrics
            rouge_scores = self.metrics.calculate_rouge(response, reference)

            try:
                semantic_similarity = self.metrics.calculate_semantic_similarity(
                    response, reference
                )
            except:
                semantic_similarity = 0.0

            # Calculate retrieval metrics if relevant docs are provided
            retrieval_precision = 0.0
            retrieval_recall = 0.0
            if relevant_docs:
                retrieval_precision = self.metrics.calculate_retrieval_precision(
                    debug_info["retrieved_info"], relevant_docs
                )
                retrieval_recall = self.metrics.calculate_retrieval_recall(
                    debug_info["retrieved_info"], relevant_docs
                )

            # Calculate faithfulness metrics
            faithfulness = self.metrics.calculate_faithfulness(
                response, debug_info["retrieved_info"]
            )

            # LLM evaluation (optional, can be expensive)
            llm_eval = {}
            if use_llm_eval:
                llm_eval = self.metrics.calculate_llm_evaluation(
                    self.rag, response, reference, question
                )
                overall_score = llm_eval.get("overall", {}).get("score", 0)
            else:
                # Use ROUGE-L as overall score if LLM evaluation is not used
                overall_score = rouge_scores["rougeL"] * 10  # Scale to 0-10

            # Compile results
            result = {
                "question": question,
                "reference": reference,
                "response": response,
                "rouge1": rouge_scores["rouge1"],
                "rouge2": rouge_scores["rouge2"],
                "rougeL": rouge_scores["rougeL"],
                "semantic_similarity": semantic_similarity,
                "retrieval_precision": retrieval_precision,
                "retrieval_recall": retrieval_recall,
                "faithfulness_score": faithfulness["faithfulness_score"],
                "hallucination_score": faithfulness["hallucination_score"],
                "overall_score": overall_score,
                "retrieved_docs": len(debug_info["retrieved_info"]),
                "sources": [item["source"] for item in debug_info["retrieved_info"]],
                "subqueries": debug_info.get("subqueries", []),
                "llm_evaluation": llm_eval,
            }

            results.append(result)

        # Save detailed results if requested
        if save_results:
            self.results = results
            self.save_results(output_path)

        # Convert to DataFrame for easier analysis
        return self._create_results_dataframe(results)

    def _create_results_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create a DataFrame from the evaluation results."""
        return pd.DataFrame(
            [
                {
                    "question": r["question"],
                    "reference_length": len(r["reference"]),
                    "response_length": len(r["response"]),
                    "rouge1": r["rouge1"],
                    "rouge2": r["rouge2"],
                    "rougeL": r["rougeL"],
                    "semantic_similarity": r["semantic_similarity"],
                    "retrieval_precision": r["retrieval_precision"],
                    "retrieval_recall": r["retrieval_recall"],
                    "faithfulness_score": r["faithfulness_score"],
                    "hallucination_score": r["hallucination_score"],
                    "overall_score": r["overall_score"],
                    "retrieved_docs": r["retrieved_docs"],
                }
                for r in results
            ]
        )

    def save_results(self, path: str):
        """Save detailed evaluation results to a JSON file."""
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Evaluation results saved to {path}")

    def load_results(self, path: str) -> pd.DataFrame:
        """Load evaluation results from a JSON file."""
        with open(path, "r") as f:
            self.results = json.load(f)

        logger.info(f"Loaded evaluation results from {path}")
        return self._create_results_dataframe(self.results)

    def generate_report(self, output_path: str = "./evaluation_report.md") -> str:
        """
        Generate a markdown report from the evaluation results.

        Args:
            output_path: Path to save the report

        Returns:
            Report text
        """
        if not self.results:
            return "No evaluation results available."

        # Calculate average metrics
        df = self._create_results_dataframe(self.results)

        avg_metrics = {
            "ROUGE-1": df["rouge1"].mean(),
            "ROUGE-2": df["rouge2"].mean(),
            "ROUGE-L": df["rougeL"].mean(),
            "Semantic Similarity": df["semantic_similarity"].mean(),
            "Retrieval Precision": df["retrieval_precision"].mean(),
            "Retrieval Recall": df["retrieval_recall"].mean(),
            "Faithfulness Score": df["faithfulness_score"].mean(),
            "Hallucination Score": df["hallucination_score"].mean(),
            "Overall Score": df["overall_score"].mean(),
        }

        # Generate report
        report = "# RAG Evaluation Report\n\n"

        report += "## Summary\n\n"
        report += f"Total Questions Evaluated: {len(self.results)}\n\n"

        report += "### Average Metrics\n\n"
        report += "| Metric | Score |\n"
        report += "| ------ | ----- |\n"
        for metric, score in avg_metrics.items():
            report += f"| {metric} | {score:.4f} |\n"

        report += "\n## Question-by-Question Results\n\n"

        for i, result in enumerate(self.results):
            report += f"### Question {i+1}\n\n"
            report += f"**Question:** {result['question']}\n\n"
            report += f"**Reference Answer:** {result['reference']}\n\n"
            report += f"**System Response:** {result['response']}\n\n"

            report += "**Metrics:**\n"
            report += f"- ROUGE-1: {result['rouge1']:.4f}\n"
            report += f"- ROUGE-2: {result['rouge2']:.4f}\n"
            report += f"- ROUGE-L: {result['rougeL']:.4f}\n"
            report += f"- Semantic Similarity: {result['semantic_similarity']:.4f}\n"
            report += f"- Faithfulness Score: {result['faithfulness_score']:.4f}\n"
            report += f"- Overall Score: {result['overall_score']:.4f}\n\n"

            report += "**Retrieval:**\n"
            report += f"- Documents Retrieved: {result['retrieved_docs']}\n"
            report += "- Sources: " + ", ".join(result["sources"]) + "\n"
            report += f"- Subqueries: {', '.join(result.get('subqueries', []))}\n\n"

            if result.get("llm_evaluation"):
                report += "**LLM Evaluation:**\n"
                llm_eval = result["llm_evaluation"]
                for criterion, data in llm_eval.items():
                    if (
                        criterion != "overall"
                        and isinstance(data, dict)
                        and "score" in data
                    ):
                        report += f"- {criterion.capitalize()}: {data['score']} - {data['justification']}\n"

                if "overall" in llm_eval:
                    report += f"\n**Overall LLM Evaluation:** {llm_eval['overall'].get('score')} - {llm_eval['overall'].get('summary')}\n"

            report += "\n---\n\n"

        # Save report if path is provided
        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {output_path}")

        return report
