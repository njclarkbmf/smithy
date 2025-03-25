import json
import logging
import os
import sys
from typing import Any, Dict, List

import pandas as pd

# Add parent directory to path to import agentic_rag
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_rag import AgenticRAG, Config, Document

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Calculates various metrics for evaluating RAG system performance."""

    def __init__(self):
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

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        Requires sentence-transformers package.
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

    def evaluate_questions(self, questions: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Evaluate the RAG system on a set of questions.

        Args:
            questions: List of dictionaries with 'question' and 'answer' keys

        Returns:
            DataFrame with evaluation results
        """
        results = []

        for i, item in enumerate(questions):
            question = item["question"]
            reference = item["answer"]

            print(f"\nEvaluating question {i+1}/{len(questions)}: {question[:50]}...")

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

            # LLM evaluation (optional, can be expensive)
            llm_eval = self.metrics.calculate_llm_evaluation(
                self.rag, response, reference, question
            )

            # Compile results
            result = {
                "question": question,
                "reference": reference,
                "response": response,
                "rouge1": rouge_scores["rouge1"],
                "rouge2": rouge_scores["rouge2"],
                "rougeL": rouge_scores["rougeL"],
                "semantic_similarity": semantic_similarity,
                "overall_score": llm_eval.get("overall", {}).get("score", 0),
                "retrieved_docs": len(debug_info["retrieved_info"]),
                "sources": [item["source"] for item in debug_info["retrieved_info"]],
                "llm_evaluation": llm_eval,
            }

            results.append(result)

        # Convert to DataFrame for easier analysis
        self.results = results
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
                    "overall_score": r["overall_score"],
                    "retrieved_docs": r["retrieved_docs"],
                }
                for r in results
            ]
        )

    def save_results(self, path: str):
        """Save detailed evaluation results to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)


def setup_test_data():
    """Create a test RAG system with sample data for evaluation."""
    # Initialize the system
    config = Config(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o",
        provider="openai",
        embedding_model="text-embedding-3-large",
        db_path="./lancedb_evaluation",
    )

    rag = AgenticRAG(config)

    # Add test documents about programming languages
    docs = [
        Document(
            content="""
            Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991.
            Python's design philosophy emphasizes code readability with its notable use of significant whitespace.
            Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small
            and large-scale projects. Python is dynamically typed and garbage-collected. It supports multiple programming
            paradigms, including structured, object-oriented, and functional programming. Python is often described as a
            "batteries included" language due to its comprehensive standard library.
            """,
            source="python_info.txt",
            doc_id="python_info",
        ),
        Document(
            content="""
            Java is a high-level, class-based, object-oriented programming language that is designed to have as few
            implementation dependencies as possible. It is a general-purpose programming language intended to let application
            developers write once, run anywhere (WORA), meaning that compiled Java code can run on all platforms that support
            Java without the need for recompilation. Java applications are typically compiled to bytecode that can run on any
            Java virtual machine (JVM) regardless of the underlying computer architecture. Java was originally developed by
            James Gosling at Sun Microsystems (which has since been acquired by Oracle) and released in 1995 as a core
            component of Sun Microsystems' Java platform.
            """,
            source="java_info.txt",
            doc_id="java_info",
        ),
        Document(
            content="""
            JavaScript, often abbreviated as JS, is a programming language that conforms to the ECMAScript specification.
            JavaScript is high-level, often just-in-time compiled, and multi-paradigm. It has curly-bracket syntax, dynamic
            typing, prototype-based object-orientation, and first-class functions. Alongside HTML and CSS, JavaScript is one
            of the core technologies of the World Wide Web. JavaScript enables interactive web pages and is an essential part
            of web applications. The vast majority of websites use it for client-side page behavior, and all major web browsers
            have a dedicated JavaScript engine to execute it. JavaScript was created by Brendan Eich in 1995 during his time at
            Netscape Communications. It was inspired by Java, Scheme and Self.
            """,
            source="javascript_info.txt",
            doc_id="javascript_info",
        ),
        Document(
            content="""
            C++ is a general-purpose programming language created by Bjarne Stroustrup as an extension of the C programming
            language, or "C with Classes". The language has expanded significantly over time, and modern C++ now has
            object-oriented, generic, and functional features in addition to facilities for low-level memory manipulation.
            It is almost always implemented as a compiled language, and many vendors provide C++ compilers, including the
            Free Software Foundation, LLVM, Microsoft, Intel, Oracle, and IBM, so it is available on many platforms.
            """,
            source="cpp_info.txt",
            doc_id="cpp_info",
        ),
    ]

    print(f"Adding {len(docs)} test documents...")
    rag.add_documents(docs)

    return rag


def create_test_questions() -> List[Dict[str, str]]:
    """Create test questions for evaluation."""
    return [
        {
            "question": "Who created Python and when was it released?",
            "answer": "Python was created by Guido van Rossum and was first released in 1991.",
        },
        {
            "question": "What are the key characteristics of JavaScript?",
            "answer": "JavaScript is a high-level, just-in-time compiled, multi-paradigm programming language. It features curly-bracket syntax, dynamic typing, prototype-based object-orientation, and first-class functions. It was created by Brendan Eich in 1995 and is one of the core technologies of the World Wide Web alongside HTML and CSS.",
        },
        {
            "question": "Compare Python and Java programming languages.",
            "answer": "Python and Java are both high-level programming languages. Python was created by Guido van Rossum in 1991 and emphasizes code readability with significant whitespace. It is dynamically typed and supports multiple programming paradigms. Java was developed by James Gosling at Sun Microsystems and released in 1995. It is class-based, object-oriented, and designed to have few implementation dependencies with a 'write once, run anywhere' approach. Java is typically compiled to bytecode that runs on a JVM, while Python is generally interpreted.",
        },
        {
            "question": "What are the main features of C++?",
            "answer": "C++ is a general-purpose programming language created by Bjarne Stroustrup as an extension of C ('C with Classes'). It features object-oriented, generic, and functional capabilities in addition to low-level memory manipulation. C++ is almost always implemented as a compiled language and is available on many platforms through various compilers provided by vendors like Free Software Foundation, LLVM, Microsoft, Intel, Oracle, and IBM.",
        },
    ]


def demo_rag_evaluation():
    """Demonstrate the RAG evaluation system."""
    print("\n=== RAG Evaluation Example ===")

    # Set up test data
    rag = setup_test_data()

    # Create test questions
    test_questions = create_test_questions()

    # Create evaluator
    evaluator = RAGEvaluator(rag)

    # Run evaluation
    print("\nRunning evaluation...")
    results_df = evaluator.evaluate_questions(test_questions)

    # Display results
    print("\nEvaluation Results:")
    print(results_df.to_string())

    # Calculate average scores
    avg_scores = {
        "ROUGE-1": results_df["rouge1"].mean(),
        "ROUGE-2": results_df["rouge2"].mean(),
        "ROUGE-L": results_df["rougeL"].mean(),
        "Semantic Similarity": results_df["semantic_similarity"].mean(),
        "Overall Score": results_df["overall_score"].mean(),
    }

    print("\nAverage Scores:")
    for metric, score in avg_scores.items():
        print(f"{metric}: {score:.4f}")

    # Save detailed results
    evaluator.save_results("./evaluation_results.json")
    print("\nDetailed results saved to evaluation_results.json")


if __name__ == "__main__":
    demo_rag_evaluation()
