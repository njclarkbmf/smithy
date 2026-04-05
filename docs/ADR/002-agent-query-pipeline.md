# ADR-002: Agent-Based Query/Retrieval/Generation Pipeline

| Field | Value |
|-------|-------|
| **Title** | Three-agent pipeline with QueryPlanner, InfoRetriever, and ResponseGenerator |
| **Status** | accepted |
| **Deciders** | Architecture team |
| **Date** | 2025-04-05 |
| **Supersedes** | None |
| **Tags** | agents, pipeline, rag, architecture |

## Context

The Smithy framework implements a Retrieval-Augmented Generation (RAG) pipeline. The core challenge is how to process a user query through multiple stages — decomposing complex queries, retrieving relevant information, and generating grounded responses — while maintaining modularity, testability, and extensibility.

Key requirements observed in the codebase:
- Complex queries should be broken into simpler subqueries for better retrieval
- Retrieved documents should be deduplicated and re-ranked by similarity
- Response generation must be grounded in retrieved context only (no hallucination)
- Each stage should be independently testable and replaceable
- Advanced techniques (HyDE, Multi-Query Fusion, Reranking, Self-Improving) should wrap the base pipeline without modifying it

## Decision

We will use a **three-agent pipeline** orchestrated by the `AgenticRAG` facade:

```
User Query → QueryPlanner → [Subqueries] → InfoRetriever → [Retrieved Docs] → ResponseGenerator → Response
```

### Agent Responsibilities

1. **QueryPlanner** (`agentic_rag/agents/query_planner.py`)
   - Uses an LLM to decompose a user query into 2-4 specific search queries
   - System prompt instructs the LLM to return one query per line
   - Falls back to the original query if the LLM call fails
   - Temperature: 0.2 (deterministic, focused)

2. **InfoRetriever** (`agentic_rag/agents/info_retriever.py`)
   - Searches the vector database for each subquery
   - Deduplicates results by document ID
   - Re-ranks by cosine similarity to the first (primary) subquery
   - Stateless — depends on `VectorDBManager` via dependency injection

3. **ResponseGenerator** (`agentic_rag/agents/response_generator.py`)
   - Constructs a system prompt with anti-hallucination guidelines
   - Formats retrieved context with source attribution
   - Calls the LLM with configured temperature
   - Returns error message on failure

### Orchestration (`agentic_rag/main.py`)

```python
def query(self, user_query: str) -> Tuple[str, Dict[str, Any]]:
    subqueries = self.query_planner.plan_query(user_query)
    retrieved_info = self.info_retriever.retrieve(subqueries, self.config.top_k)
    response = self.response_generator.generate(user_query, retrieved_info)
    return response, debug_info
```

### Advanced Technique Wrappers

Advanced techniques wrap the base pipeline rather than replacing it:

- **HyDE** generates a hypothetical document, uses it as the search query, then calls the base `ResponseGenerator`
- **MultiQueryFusion** generates alternative queries, searches for each, deduplicates, then calls `ResponseGenerator`
- **CrossEncoderReranker** replaces `InfoRetriever` with a two-stage retrieval + rerank pipeline
- **SelfImprovingRAG** wraps the full pipeline, adding feedback collection and automatic evaluation

```
                    ┌─────────────────────────────────────────────┐
                    │              AgenticRAG                     │
                    │                                             │
 User Query ───────►│  QueryPlanner ──► InfoRetriever ──► RespGen │
                    │                                             │
                    └─────────────────────────────────────────────┘
                           ▲              ▲               ▲
          HyDE ────────────┘              │               │
          MultiQueryFusion ───────────────┘               │
          CrossEncoderReranker ───────────────────────────┘
```

## Consequences

### Positive
- **Testability**: Each agent can be unit-tested in isolation with mocked dependencies
- **Extensibility**: New retrieval strategies (HyDE, reranking) wrap the base pipeline
- **Debuggability**: Each stage returns structured debug info
- **Composability**: Advanced techniques can be chained

### Negative
- **Latency**: QueryPlanner adds an extra LLM call before retrieval
- **Complexity**: Three agents vs. a single retrieve-then-generate loop
- **Error propagation**: A bad QueryPlanner output degrades the entire pipeline

### Risks
- QueryPlanner may generate irrelevant subqueries for simple queries, adding unnecessary API calls
- InfoRetriever's deduplication strategy assumes `id` uniqueness across searches, which may not hold if the DB is rebuilt

## Alternatives Considered

### Single-Stage Retrieve-Then-Generate
- **Description:** Use the user query directly for vector search, then generate
- **Pros:** Faster, simpler, fewer API calls
- **Cons:** Poor performance on multi-part or complex queries
- **Why rejected:** Does not meet the "agentic" requirement — the framework's value proposition is intelligent query decomposition

### LangGraph-Based State Machine
- **Description:** Use `langgraph` to define the pipeline as a state graph
- **Pros:** Declarative, supports conditional routing and loops
- **Cons:** Heavy dependency, steep learning curve, overkill for a linear pipeline
- **Why rejected:** `langgraph` is already a dependency but is not yet used. The team may adopt it in the future for self-improving RAG with feedback loops, but for now, the explicit Python orchestration is clearer.

### ReAct Agent Pattern
- **Description:** Use a single ReAct-style agent that iteratively retrieves and reasons
- **Pros:** More flexible, can adapt mid-query
- **Cons:** Unpredictable latency, harder to test, potential for infinite loops
- **Why rejected:** The current use case benefits from predictable, bounded-latency responses.

## Notes

- The pipeline currently executes synchronously. Future work could parallelize subquery retrieval using `asyncio.gather()`.
- The QueryPlanner's system prompt could be enhanced with few-shot examples for more reliable query decomposition.

## References

- [Query Decomation for RAG — HyDE Paper](https://arxiv.org/abs/2212.10496)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
