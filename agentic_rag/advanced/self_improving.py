import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Collects and manages feedback on RAG responses."""

    def __init__(self, db_path: str = "./feedback_data.json"):
        self.db_path = db_path
        self.feedback_data = self._load_data()

    def _load_data(self) -> Dict:
        """Load existing feedback data or create new structure."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    return json.load(f)
            except:
                return {"feedback": [], "metrics": {}}
        else:
            return {"feedback": [], "metrics": {}}

    def _save_data(self):
        """Save feedback data to file."""
        with open(self.db_path, "w") as f:
            json.dump(self.feedback_data, f, indent=2)

    def add_feedback(
        self,
        query: str,
        response: str,
        retrieved_docs: List[Dict],
        feedback: str,
        rating: int,
        feedback_type: str = "user",
    ):
        """Add feedback entry to the database."""
        feedback_entry = {
            "id": len(self.feedback_data["feedback"]) + 1,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "response": response,
            "retrieved_docs": [
                {"id": doc["id"], "source": doc["source"]} for doc in retrieved_docs
            ],
            "feedback": feedback,
            "rating": rating,
            "feedback_type": feedback_type,
        }

        self.feedback_data["feedback"].append(feedback_entry)
        self._save_data()
        return feedback_entry["id"]

    def get_feedback_stats(self) -> Dict:
        """Calculate and return feedback statistics."""
        if not self.feedback_data["feedback"]:
            return {"count": 0, "avg_rating": 0}

        ratings = [entry["rating"] for entry in self.feedback_data["feedback"]]
        return {
            "count": len(ratings),
            "avg_rating": sum(ratings) / len(ratings),
            "rating_distribution": {str(r): ratings.count(r) for r in range(1, 6)},
        }

    def analyze_feedback(self, rag_system) -> Dict:
        """Analyze feedback to identify patterns and improvement areas."""
        if len(self.feedback_data["feedback"]) < 5:
            return {
                "status": "Not enough feedback for meaningful analysis",
                "suggestions": [],
            }

        # Get low-rated responses
        low_rated = [f for f in self.feedback_data["feedback"] if f["rating"] <= 2]

        if not low_rated:
            return {"status": "No low-rated responses to analyze", "suggestions": []}

        # Use the RAG system's LLM to analyze feedback
        analysis_prompt = f"""
        Analyze the following user feedback on RAG system responses, focusing on areas for improvement:
        
        {json.dumps(low_rated, indent=2)}
        
        Please identify:
        1. Common patterns in queries that received poor ratings
        2. Areas where the system seems to be struggling
        3. Specific suggestions for improvement
        
        Provide actionable recommendations for improving the RAG system based on this feedback.
        """

        try:
            if rag_system.config.provider == "openai":
                response = rag_system.config.client.chat.completions.create(
                    model=rag_system.config.model,
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.0,
                )
                analysis = response.choices[0].message.content
            elif rag_system.config.provider == "anthropic":
                response = rag_system.config.client.messages.create(
                    model=rag_system.config.model,
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.0,
                )
                analysis = response.content[0].text

            # Save analysis to metrics
            self.feedback_data["metrics"]["latest_analysis"] = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "analysis": analysis,
            }
            self._save_data()

            return {"status": "Analysis completed", "analysis": analysis}
        except Exception as e:
            logger.error(f"Error analyzing feedback: {str(e)}")
            return {"status": "Error", "error": str(e)}


class SelfImprovingRAG:
    """An extension of AgenticRAG with self-improvement capabilities."""

    def __init__(self, rag_system, feedback_db_path: str = "./feedback_data.json"):
        """
        Initialize the self-improving RAG system.

        Args:
            rag_system: An instance of AgenticRAG
            feedback_db_path: Path to the feedback database
        """
        self.rag_system = rag_system
        self.config = rag_system.config
        self.feedback_collector = FeedbackCollector(feedback_db_path)
        self.queries_log = []

    def query(self, user_query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a user query through the RAG pipeline with logging.
        """
        start_time = time.time()

        # Log the query start
        query_log = {
            "query": user_query,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_steps": [],
        }

        # 1. Plan the query
        subqueries = self.rag_system.query_planner.plan_query(user_query)
        query_log["processing_steps"].append(
            {
                "step": "query_planning",
                "subqueries": subqueries,
                "time": time.time() - start_time,
            }
        )

        # 2. Retrieve information
        step_start = time.time()
        retrieved_info = self.rag_system.info_retriever.retrieve(
            subqueries, self.config.top_k
        )
        query_log["processing_steps"].append(
            {
                "step": "retrieval",
                "retrieved_docs": len(retrieved_info),
                "time": time.time() - step_start,
            }
        )

        # 3. Generate response
        step_start = time.time()
        response = self.rag_system.response_generator.generate(
            user_query, retrieved_info
        )
        query_log["processing_steps"].append(
            {
                "step": "response_generation",
                "response_length": len(response),
                "time": time.time() - step_start,
            }
        )

        # Total processing time
        query_log["total_time"] = time.time() - start_time

        # Add to queries log
        self.queries_log.append(query_log)

        # 4. Return response and debug info
        debug_info = {
            "subqueries": subqueries,
            "retrieved_info": [
                {
                    "id": item["id"],
                    "source": item["source"],
                    "similarity": item.get("similarity"),
                }
                for item in retrieved_info
            ],
            "processing_time": query_log["total_time"],
        }

        return response, debug_info

    def collect_feedback(
        self,
        query: str,
        response: str,
        retrieved_docs: List[Dict],
        feedback: str,
        rating: int,
    ) -> int:
        """Collect user feedback on a response."""
        return self.feedback_collector.add_feedback(
            query, response, retrieved_docs, feedback, rating
        )

    def automatic_evaluation(
        self, query: str, response: str, retrieved_docs: List[Dict]
    ) -> int:
        """
        Automatically evaluate the quality of a response.
        Returns a rating from 1-5.
        """
        eval_prompt = f"""
        Evaluate the quality of the following RAG system response:
        
        Query: {query}
        
        Response: {response}
        
        Consider the following criteria:
        1. Relevance: How relevant is the response to the query?
        2. Completeness: Does the response fully address the query?
        3. Accuracy: Does the response contain factual errors?
        4. Coherence: How well-structured and logical is the response?
        
        Please provide a rating from 1 to 5 where:
        1 = Poor (significant issues, doesn't answer the query)
        5 = Excellent (comprehensive, accurate, well-structured)
        
        Return only a number from 1 to 5, without any explanation.
        """

        try:
            if self.config.provider == "openai":
                eval_response = self.config.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": eval_prompt}],
                    temperature=0.0,
                )
                rating_text = eval_response.choices[0].message.content.strip()

            elif self.config.provider == "anthropic":
                eval_response = self.config.client.messages.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": eval_prompt}],
                    temperature=0.0,
                )
                rating_text = eval_response.content[0].text.strip()

            # Extract the rating
            rating = int(rating_text[0]) if rating_text[0].isdigit() else 3
            rating = min(max(rating, 1), 5)  # Ensure it's between 1-5

            # Add automatic feedback
            self.feedback_collector.add_feedback(
                query,
                response,
                retrieved_docs,
                "Automatic evaluation",
                rating,
                "automatic",
            )

            return rating

        except Exception as e:
            logger.error(f"Error in automatic evaluation: {str(e)}")
            return 3  # Default to neutral rating

    def generate_improvement_report(self) -> Dict:
        """
        Generate a report with insights and recommendations for improvement.
        """
        # Analyze feedback
        feedback_analysis = self.feedback_collector.analyze_feedback(self.rag_system)

        # Analyze query patterns
        if len(self.queries_log) < 5:
            return {
                "status": "Not enough data for meaningful analysis",
                "feedback_analysis": feedback_analysis,
            }

        # Generate report
        report_prompt = f"""
        Generate an improvement report for a Retrieval-Augmented Generation (RAG) system based on the following data:
        
        Query Logs: {json.dumps(self.queries_log[-20:], indent=2)}
        
        Feedback Analysis: {feedback_analysis.get("analysis", "No analysis available")}
        
        Include the following sections:
        1. Performance Summary
        2. Common Query Patterns
        3. Areas for Improvement
        4. Specific Recommendations
        
        Be specific and actionable in your recommendations.
        """

        try:
            if self.config.provider == "openai":
                report_response = self.config.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": report_prompt}],
                    temperature=0.0,
                )
                report = report_response.choices[0].message.content

            elif self.config.provider == "anthropic":
                report_response = self.config.client.messages.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": report_prompt}],
                    temperature=0.0,
                )
                report = report_response.content[0].text

            return {
                "status": "Report generated",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "report": report,
                "feedback_stats": self.feedback_collector.get_feedback_stats(),
                "query_count": len(self.queries_log),
            }
        except Exception as e:
            logger.error(f"Error generating improvement report: {str(e)}")
            return {"status": "Error", "error": str(e)}
