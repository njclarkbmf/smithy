import logging
from typing import List

logger = logging.getLogger(__name__)


class QueryPlanner:
    """Plans queries based on user input."""

    def __init__(self, config):
        self.config = config

    def plan_query(self, user_query: str) -> List[str]:
        """
        Generate search queries to effectively retrieve relevant information.
        Returns a list of subqueries to execute.
        """
        system_prompt = """
        You are a Query Planning Agent. Your job is to analyze a user's question and create 
        effective search queries to find the most relevant information in a knowledge base.
        
        For complex queries, break them down into 2-4 specific search queries that will retrieve 
        different aspects of the information needed. For simple queries, a single search query may be sufficient.
        
        Return ONLY the search queries, one per line. Do not include any explanations or additional text.
        """

        try:
            if self.config.provider == "openai":
                response = self.config.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query},
                    ],
                    temperature=0.2,
                )
                subqueries = response.choices[0].message.content.strip().split("\n")

            elif self.config.provider == "anthropic":
                response = self.config.client.messages.create(
                    model=self.config.model,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_query}],
                    temperature=0.2,
                )
                subqueries = response.content[0].text.strip().split("\n")

            # Filter out empty queries and deduplicate
            subqueries = [q.strip() for q in subqueries if q.strip()]
            subqueries = list(dict.fromkeys(subqueries))

            return subqueries

        except Exception as e:
            logger.error(f"Error planning query: {str(e)}")
            # Fallback to using the original query
            return [user_query]
