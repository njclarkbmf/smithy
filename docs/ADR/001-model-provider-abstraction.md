# ADR-001: Model Provider Abstraction & Qwen Integration

| Field | Value |
|-------|-------|
| **Title** | Unified ModelRouter abstraction for multi-provider LLM support including Qwen |
| **Status** | accepted |
| **Deciders** | Architecture team, ML engineering lead |
| **Date** | 2025-04-05 |
| **Supersedes** | None |
| **Tags** | llm, provider, qwen, architecture, refactoring |

## Context

The Smithy RAG framework currently supports two LLM providers: OpenAI and Anthropic. Provider-specific API calls are scattered across at least 8 files:

- `agentic_rag/config.py` — client initialization
- `agentic_rag/agents/query_planner.py` — chat completions
- `agentic_rag/agents/response_generator.py` — chat completions
- `agentic_rag/advanced/hyde.py` — chat completions
- `agentic_rag/advanced/multi_query.py` — chat completions
- `agentic_rag/advanced/self_improving.py` — chat completions + JSON mode
- `agentic_rag/evaluation/metrics.py` — chat completions + JSON mode
- `agentic_rag/vectordb.py` — embedding API calls

Each file contains duplicated `if provider == "openai" ... elif provider == "anthropic"` blocks. Adding a third provider (Qwen) would require modifying all 8 files, violating the Open-Closed Principle.

Additionally, the `Config` class in `config.py` hardcodes provider validation and throws `ValueError` for unknown providers, making it impossible to add Qwen without code changes.

Key requirements:
- Support Qwen models via DashScope SDK and OpenAI-compatible endpoints (vLLM/Ollama)
- Support Qwen embedding models (`text-embedding-v1/v2/v3`)
- Maintain backward compatibility with OpenAI and Anthropic
- Allow separate providers for chat vs. embeddings (since Anthropic has no embedding API)
- Enable graceful degradation and explicit error messages for unsupported features

## Decision

We will introduce a **ModelRouter** pattern with the following structure:

1. **`BaseProvider`** — Abstract base class defining the interface:
   - `chat_completion(messages, model, temperature, top_p, max_tokens, response_format) -> ChatResponse`
   - `get_embeddings(texts, model) -> EmbeddingResponse`

2. **Concrete providers**: `OpenAIProvider`, `AnthropicProvider`, `QwenProvider`
   - Each implements the `BaseProvider` interface
   - `QwenProvider` supports two modes:
     - **DashScope SDK mode** (`use_dashscope=true`): Uses the native `dashscope` Python package
     - **OpenAI-compatible mode** (`use_dashscope=false`): Uses the `openai` Python SDK pointing at a vLLM/Ollama endpoint serving Qwen models

3. **`ModelRouter`** — Single entry point that:
   - Instantiates the correct provider based on configuration
   - Routes all chat and embedding calls through a unified interface
   - Maintains separate chat and embedding providers (enabling, e.g., Anthropic chat + Qwen embeddings)
   - Exposes helper methods like `is_json_mode_supported()`

4. **`Config` class updated** to hold a `ModelRouter` instance instead of a raw OpenAI/Anthropic client.

5. **Environment variables extended**:
   - `DASHSCOPE_API_KEY`, `DASHSCOPE_BASE_URL`, `USE_DASHSCOPE`
   - `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`, `EMBEDDING_USE_DASHSCOPE`
   - `TOP_P`, `MAX_TOKENS`

6. **Qwen model registry** — A dictionary mapping Qwen model names to their capabilities (max tokens, JSON mode support, temperature range).

### File changes
- **New**: `agentic_rag/providers/__init__.py`
- **New**: `agentic_rag/providers/model_router.py`
- **Modified**: `agentic_rag/config.py` — uses `ModelRouter` instead of direct client
- **Modified**: `agentic_rag/utils/env_loader.py` — loads Qwen env vars
- **Modified**: All agent files to use `self.config.client.chat(...)` instead of `self.config.client.chat.completions.create(...)`
- **Modified**: `.env-example`, `requirements.txt`

## Consequences

### Positive
- Adding a new provider requires changes in **one file** (`model_router.py`) instead of 8+
- Qwen is a first-class citizen with full chat + embedding support
- Separate chat/embedding providers solve the Anthropic embedding gap
- Cleaner agent code — no more `if/elif` provider blocks
- OpenAI-compatible mode enables local Qwen serving via vLLM/Ollama

### Negative
- Migration requires updating all agent files to use the new `chat()` method signature
- Additional dependency: `dashscope` package (~50MB)
- Slightly more complex configuration (more env vars)

### Risks
- DashScope SDK API differences may require version pinning
- OpenAI-compatible mode depends on the serving framework's API compatibility
- If `ModelRouter` has bugs, it affects all providers simultaneously

## Alternatives Considered

### LangChain Provider Abstraction
- **Description:** Use LangChain's `ChatOpenAI`, `ChatAnthropic`, and `ChatTongyi` (Qwen) classes
- **Pros:** Mature, battle-tested, already a dependency via `langchain` and `langchain-openai`
- **Cons:** Heavy dependency graph, adds latency, forces LangChain patterns on the entire codebase, reduces control over API parameters
- **Why rejected:** Smithy already uses LangChain for text splitting but deliberately avoids it for the agent layer to maintain lightweight, explicit control over API calls.

### Provider-SConfig Subclasses
- **Description:** Create `OpenAIConfig`, `AnthropicConfig`, `QwenConfig` subclasses
- **Pros:** Type-safe, clean separation
- **Cons:** Duplicates config logic, makes polymorphism harder in agents
- **Why rejected:** Would require agents to handle config type dispatching, pushing the problem elsewhere.

### Direct SDK Patching
- **Description:** Monkey-patch the OpenAI client to route to Qwen when provider is "qwen"
- **Pros:** Zero code changes in agents
- **Cons:** Fragile, breaks on SDK updates, unmaintainable, debugging nightmare
- **Why rejected:** Obvious maintainability and reliability concerns.

## Notes

- The Qwen provider should be tested against both `qwen-turbo` (fast, cheap) and `qwen-max` (high quality) for RAG workloads.
- For local serving, vLLM provides better throughput than Ollama for production use cases.

## References

- [DashScope API Documentation](https://help.aliyun.com/zh/dashscope/)
- [Qwen Model Documentation](https://qwen.readthedocs.io/)
- [vLLM Qwen Support](https://docs.vllm.ai/en/latest/models/supported_models.html#qwen)
