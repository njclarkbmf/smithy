import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BenchmarkDataset:
    """Base class for benchmark datasets."""

    def __init__(self, name: str):
        self.name = name
        self.data = []

    def load(self) -> List[Dict[str, Any]]:
        """Load the dataset."""
        raise NotImplementedError("Subclasses must implement this method")

    def save_to_json(self, path: str):
        """Save the dataset to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)

    @staticmethod
    def load_from_json(path: str) -> List[Dict[str, Any]]:
        """Load a dataset from a JSON file."""
        with open(path, "r") as f:
            return json.load(f)


class QABenchmarkDataset(BenchmarkDataset):
    """Question answering benchmark dataset."""

    def __init__(self, name: str, data_path: str):
        super().__init__(name)
        self.data_path = data_path

    def load(self) -> List[Dict[str, Any]]:
        """Load the QA dataset."""
        # Check if path is a CSV file
        if self.data_path.endswith(".csv"):
            return self._load_from_csv()

        # Check if path is a JSON file
        elif self.data_path.endswith(".json"):
            return self._load_from_json()

        # If path is a directory, look for text files (one per QA pair)
        elif os.path.isdir(self.data_path):
            return self._load_from_directory()

        else:
            logger.error(f"Unsupported data path: {self.data_path}")
            return []

    def _load_from_csv(self) -> List[Dict[str, Any]]:
        """Load data from a CSV file."""
        df = pd.read_csv(self.data_path)

        # Convert DataFrame to list of dictionaries
        self.data = []
        for _, row in df.iterrows():
            question = row.get("question", row.get("Question", ""))
            answer = row.get("answer", row.get("Answer", ""))

            if not question or not answer:
                continue

            item = {"question": question, "answer": answer}

            # Add optional fields if they exist
            for field in ["relevant_docs", "category", "difficulty"]:
                if field in row:
                    item[field] = row[field]

            self.data.append(item)

        return self.data

    def _load_from_json(self) -> List[Dict[str, Any]]:
        """Load data from a JSON file."""
        self.data = self.load_from_json(self.data_path)
        return self.data

    def _load_from_directory(self) -> List[Dict[str, Any]]:
        """Load data from a directory of text files."""
        self.data = []
        question_files = [
            f for f in os.listdir(self.data_path) if f.endswith("_question.txt")
        ]

        for q_file in question_files:
            # Get the corresponding answer file
            a_file = q_file.replace("_question.txt", "_answer.txt")

            if not os.path.exists(os.path.join(self.data_path, a_file)):
                continue

            # Read question and answer
            with open(os.path.join(self.data_path, q_file), "r") as f:
                question = f.read().strip()

            with open(os.path.join(self.data_path, a_file), "r") as f:
                answer = f.read().strip()

            # Check for metadata file
            meta_file = q_file.replace("_question.txt", "_meta.json")
            metadata = {}

            if os.path.exists(os.path.join(self.data_path, meta_file)):
                with open(os.path.join(self.data_path, meta_file), "r") as f:
                    metadata = json.load(f)

            item = {"question": question, "answer": answer}
            item.update(metadata)

            self.data.append(item)

        return self.data


class CustomBenchmarkDataset(BenchmarkDataset):
    """Custom benchmark dataset for specialized evaluations."""

    def __init__(self, name: str, data: List[Dict[str, Any]]):
        super().__init__(name)
        self.data = data

    def load(self) -> List[Dict[str, Any]]:
        """Return the already loaded data."""
        return self.data


class DocumentRetrievalBenchmark(BenchmarkDataset):
    """Benchmark for document retrieval tasks."""

    def __init__(self, name: str, data_path: str):
        super().__init__(name)
        self.data_path = data_path

    def load(self) -> List[Dict[str, Any]]:
        """Load the document retrieval benchmark dataset."""
        # Check if path is a CSV file
        if self.data_path.endswith(".csv"):
            df = pd.read_csv(self.data_path)

            # Convert DataFrame to list of dictionaries
            self.data = []
            for _, row in df.iterrows():
                query = row.get("query", "")
                relevant_docs = row.get("relevant_docs", "")

                if not query or not relevant_docs:
                    continue

                # Parse relevant_docs if it's a string
                if isinstance(relevant_docs, str):
                    try:
                        relevant_docs = json.loads(relevant_docs)
                    except:
                        # Try comma-separated format
                        relevant_docs = [
                            doc.strip() for doc in relevant_docs.split(",")
                        ]

                self.data.append({"query": query, "relevant_docs": relevant_docs})

        # Check if path is a JSON file
        elif self.data_path.endswith(".json"):
            self.data = self.load_from_json(self.data_path)

        else:
            logger.error(f"Unsupported data path: {self.data_path}")
            return []

        return self.data


class TRECBenchmarkDataset(BenchmarkDataset):
    """
    Loads benchmark data from TREC format files.

    TREC (Text REtrieval Conference) is a standard format for information retrieval evaluation.
    """

    def __init__(self, name: str, queries_path: str, qrels_path: str):
        super().__init__(name)
        self.queries_path = queries_path
        self.qrels_path = qrels_path

    def load(self) -> List[Dict[str, Any]]:
        """Load the TREC format benchmark dataset."""
        # Load queries
        queries = {}
        with open(self.queries_path, "r") as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    qid, text = parts
                    queries[qid] = text

        # Load relevance judgments
        qrels = {}
        with open(self.qrels_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, _, docid, rel = parts[:4]
                    if int(rel) > 0:  # Only consider relevant documents
                        if qid not in qrels:
                            qrels[qid] = []
                        qrels[qid].append(docid)

        # Create dataset
        self.data = []
        for qid in queries:
            if qid in qrels:
                self.data.append(
                    {"question": queries[qid], "relevant_docs": qrels[qid]}
                )

        return self.data


def create_test_dataset(
    num_questions: int = 10, save_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Create a simple test dataset with predefined questions and answers.

    Args:
        num_questions: Number of questions to include
        save_path: Optional path to save the dataset

    Returns:
        List of dictionaries with questions and answers
    """
    test_data = [
        {
            "question": "What is artificial intelligence?",
            "answer": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. These systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.",
        },
        {
            "question": "Explain the difference between supervised and unsupervised learning.",
            "answer": "Supervised learning uses labeled training data with input-output pairs to learn a mapping function, while unsupervised learning works with unlabeled data to find patterns or structure within the data without explicit guidance. In supervised learning, the algorithm knows the 'right answers' during training, whereas in unsupervised learning, it must discover patterns independently.",
        },
        {
            "question": "What is a transformer model in machine learning?",
            "answer": "A transformer is a deep learning model architecture introduced in 2017 that uses self-attention mechanisms to process sequential data. Unlike previous sequence models like RNNs, transformers process entire sequences at once, allowing for parallelization. They're particularly effective for natural language processing tasks and form the foundation of models like BERT, GPT, and T5.",
        },
        {
            "question": "How does a neural network work?",
            "answer": "A neural network is a computational model inspired by the human brain. It consists of interconnected nodes (neurons) organized in layers. Each connection has a weight that adjusts during learning. Neurons receive inputs, apply an activation function to the weighted sum, and produce an output. Through backpropagation, the network adjusts weights to minimize prediction errors, enabling it to learn patterns in data.",
        },
        {
            "question": "What is transfer learning?",
            "answer": "Transfer learning is a machine learning technique where a model developed for one task is reused as the starting point for a model on a second task. This approach leverages knowledge gained from solving the first task to improve generalization in the second task, particularly when the second task has limited training data. It's commonly used with deep learning models like pre-trained language or vision models.",
        },
        {
            "question": "Explain the concept of embeddings in natural language processing.",
            "answer": "Embeddings in NLP are dense vector representations of words, phrases, or other textual elements in a continuous vector space. They capture semantic relationships by positioning similar words closer together in the vector space. Embeddings convert discrete textual data into numerical form that machine learning algorithms can process, while preserving meaningful relationships between words. Technologies like Word2Vec, GloVe, and contextual embeddings from transformers are common embedding methods.",
        },
        {
            "question": "What is reinforcement learning?",
            "answer": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative rewards. The agent learns through trial and error, receiving feedback in the form of rewards or penalties. Unlike supervised learning, there's no labeled training data; instead, the agent must discover which actions yield the highest rewards. It's used in robotics, game playing, and autonomous systems.",
        },
        {
            "question": "How does a recommendation system work?",
            "answer": "Recommendation systems suggest items to users based on patterns in data. They typically use collaborative filtering (finding users with similar preferences), content-based filtering (recommending items similar to what a user already likes), or hybrid approaches. Modern systems often employ matrix factorization, deep learning, or graph-based methods to identify patterns and make personalized recommendations for products, content, or services.",
        },
        {
            "question": "What is gradient descent?",
            "answer": "Gradient descent is an optimization algorithm used to minimize a function by iteratively moving toward the steepest descent as defined by the negative of the gradient. In machine learning, it's used to update model parameters to minimize the loss function. The algorithm calculates the gradient of the loss function with respect to each parameter, then updates parameters in the opposite direction of the gradient, scaled by a learning rate. Variants include batch, stochastic, and mini-batch gradient descent.",
        },
        {
            "question": "Explain the concept of overfitting in machine learning.",
            "answer": "Overfitting occurs when a machine learning model learns the training data too well, capturing noise and random fluctuations rather than the underlying pattern. An overfit model performs excellently on training data but poorly on new, unseen data. It's characterized by high variance and occurs especially with complex models trained on limited data. Techniques to prevent overfitting include regularization, cross-validation, early stopping, and using more training data.",
        },
        {
            "question": "What is deep learning?",
            "answer": "Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to progressively extract higher-level features from raw input. For example, in image processing, lower layers might identify edges, while higher layers recognize more complex patterns like faces. Deep learning has achieved breakthrough performance in tasks like computer vision, speech recognition, and natural language processing, largely due to increased computational power and data availability.",
        },
        {
            "question": "How does natural language processing work?",
            "answer": "Natural Language Processing (NLP) combines computational linguistics, machine learning, and deep learning to enable computers to process, understand, and generate human language. The process typically involves steps like tokenization (breaking text into words or subwords), part-of-speech tagging, syntactic parsing, named entity recognition, and semantic analysis. Modern NLP often uses transformer-based models pre-trained on vast text corpora, which can be fine-tuned for specific tasks like translation, summarization, or question answering.",
        },
        {
            "question": "What is computer vision?",
            "answer": "Computer vision is a field of artificial intelligence that enables computers to derive meaningful information from digital images, videos, and other visual inputs. It involves developing algorithms that can perform tasks humans do with their visual system, such as object recognition, scene understanding, and image classification. Modern computer vision heavily relies on deep learning, particularly convolutional neural networks (CNNs), and has applications in autonomous vehicles, medical imaging, surveillance, augmented reality, and more.",
        },
        {
            "question": "Explain the concept of clustering in machine learning.",
            "answer": "Clustering is an unsupervised learning technique that groups similar data points together based on their intrinsic properties without prior labels. The goal is to create clusters where data points within a cluster are more similar to each other than to those in other clusters. Common algorithms include K-means (partitions data into K clusters), hierarchical clustering (builds a tree of clusters), and DBSCAN (identifies clusters of varying shapes based on density). Clustering is used for customer segmentation, anomaly detection, document organization, and pattern recognition.",
        },
        {
            "question": "What is a convolutional neural network?",
            "answer": "A Convolutional Neural Network (CNN) is a specialized deep learning architecture designed primarily for processing grid-like data, such as images. CNNs use convolutional layers that apply filters to detect local patterns like edges, textures, and shapes, along with pooling layers that reduce dimensionality. This structure allows CNNs to automatically learn hierarchical features from data, making them highly effective for image classification, object detection, facial recognition, and other computer vision tasks, while requiring fewer parameters than fully connected networks.",
        },
    ]

    # Take the specified number of questions
    dataset = test_data[:num_questions]

    # Save the dataset if a path is provided
    if save_path:
        with open(save_path, "w") as f:
            json.dump(dataset, f, indent=2)

    return dataset
