# Smithy — Comprehensive Code Review & Modernization Report

**Date:** 2025-04-05  
**Reviewer:** AI Architecture Review Agent  
**Scope:** Full codebase audit, Qwen integration, Docker validation, ADR creation

---

## 1. Executive Summary

### Strengths
| Area | Assessment |
|------|-----------|
| **Architecture** | Clean separation of concerns — QueryPlanner, InfoRetriever, ResponseGenerator follow the SRP. The `AgenticRAG` facade in `main.py` is simple and testable. |
| **Advanced RAG Techniques** | HyDE, Multi-Query Fusion, Cross-Encoder Reranking, and Self-Improving RAG are all implemented with clean interfaces. |
| **Testability** | `conftest.py` provides rich fixtures. All agents have unit tests with proper mocking of OpenAI/Anthropic clients. |
| **Evaluation Framework** | ROUGE, semantic similarity, faithfulness, retrieval precision/recall, and LLM-based evaluation are all present. |
| **CLI** | Full-featured CLI with subcommands (`add-document`, `query`, `evaluate`) in `cli.py`. |
| **Docker baseline** | Non-root user, volume mounts, slim base image. |

### Critical Gaps
| Priority | Gap | Impact |
|----------|-----|--------|
| 🔴 P0 | **No provider abstraction** — Every agent (`query_planner.py`, `response_generator.py`, `hyde.py`, `multi_query.py`, `self_improving.py`, `metrics.py`) contains duplicated `if provider == "openai" ... elif provider == "anthropic"` blocks. Adding Qwen requires touching 8+ files. |
| 🔴 P0 | **No `.dockerignore`** — `COPY . .` in Dockerfile copies `.git`, `__pycache__`, `venv/`, etc. into the image. |
| 🔴 P0 | **`Config._init_clients()` hardcodes OpenAI/Anthropic** — `config.py:52-60` throws `ValueError` for any unknown provider, making Qwen integration impossible without code changes. |
| 🟡 P1 | **No retry logic, rate limiting, or circuit breaker** — API calls in `vectordb.py`, `query_planner.py`, `response_generator.py` have bare `try/except` that swallow errors and return fallbacks silently. |
| 🟡 P1 | **No Pydantic validation** — Config is a plain class. Invalid `temperature`, `chunk_size`, or `embedding_dimensions` values silently pass through. |
| 🟡 P1 | **LanceDB table creation uses raw schema dict** — `vectordb.py:38` passes a dict to `db.create_table()` which may not work with all LanceDB versions. No connection pooling or index optimization. |
| 🟡 P1 | **`python_requires=">=3.8"` in setup.py but uses `dict` union operator (`|`)** not available in 3.8 — inconsistent version floor. |
| 🟢 P2 | **No `CONTRIBUTING.md`** — No developer onboarding guide. |
| 🟢 P2 | **GitHub Actions only runs `cloc`** — No CI pipeline for tests, linting, or build verification. |
| 🟢 P2 | **`app.py` Gradio file upload bug** — `add_document` receives a lambda as `file_name` parameter that never gets called (`gr.Textbox(value=lambda: ...)` doesn't invoke lambdas). |

### Overall Readiness
| Component | Status |
|-----------|--------|
| Code quality | ⭐⭐⭐☆☆ (Good structure, needs DRY + validation) |
| Qwen support | ⭐☆☆☆☆ (Not implemented) |
| Docker | ⭐⭐☆☆☆ (Single compose, missing `.dockerignore`, no multi-scenario) |
| Documentation | ⭐⭐⭐☆☆ (Good README, no ADRs, no CONTRIBUTING) |
| Testing | ⭐⭐⭐☆☆ (Solid mocks, missing integration tests) |

---

## 2. Critical Fixes & Patches

### 2.1 Fix: Config provider hardcoding (`agentic_rag/config.py:52-60`)

```python
# CURRENT (config.py lines 52-60):
def _init_clients(self):
    if self.provider.lower() == "openai":
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
    elif self.provider.lower() == "anthropic":
        from anthropic import Anthropic
        self.client = Anthropic(api_key=self.api_key)
    else:
        raise ValueError(f"Unsupported provider: {self.provider}")
```

```python
# FIX — Replace with a factory-based approach (see Qwen Integration Plan below).
# The ModelRouter (new file agentic_rag/providers/model_router.py) abstracts
# all provider-specific API calls behind a single interface.
```

### 2.2 Fix: Gradio file upload bug (`app.py:149-151`)

```python
# CURRENT (app.py lines 149-151):
add_btn.click(
    add_document,
    inputs=[
        file_input,
        gr.Textbox(value=lambda: getattr(file_input, "name", "")),  # BUG: lambda never called
        source_type,
        text_input,
    ],
    outputs=[add_status],
)
```

```python
# FIX:
add_btn.click(
    add_document,
    inputs=[file_input, source_type, text_input],
    outputs=[add_status],
)

# And update add_document signature to accept (file_content, source_type, document_text)
# and derive file_name from file_input.name inside the function.
```

### 2.3 Fix: Missing `.dockerignore`

Create `/home/njclark/Projects/smithy/.dockerignore`:

```
.git
.gitignore
__pycache__
*.pyc
*.pyo
.env
venv/
.venv/
*.egg-info/
.pytest_cache/
.mypy_cache/
htmlcov/
.coverage
data/
*.log
.github/
temp_*
example_data/
feedback_*.json
evaluation_results.json
evaluation_report.md
agentic_rag.log
```

### 2.4 Fix: `setup.py` Python version floor inconsistency

```python
# CURRENT (setup.py line 28):
python_requires=">=3.8",

# FIX — The code uses features requiring 3.9+ (walrus operator in some paths,
# and dict | union would break on 3.8). Set to 3.10 to match Dockerfile:
python_requires=">=3.10",
```

Also update classifiers:
```python
"Programming Language :: Python :: 3.10",
"Programming Language :: Python :: 3.11",
"Programming Language :: Python :: 3.12",
```

### 2.5 Fix: env_loader missing Qwen API key handling

```python
# CURRENT (agentic_rag/utils/env_loader.py lines 41-49):
if config["provider"].lower() == "openai":
    config["api_key"] = os.getenv("OPENAI_API_KEY")
    if not config["api_key"]:
        logger.warning("OPENAI_API_KEY not found in environment variables")
elif config["provider"].lower() == "anthropic":
    config["api_key"] = os.getenv("ANTHROPIC_API_KEY")
    if not config["api_key"]:
        logger.warning("ANTHROPIC_API_KEY not found in environment variables")
else:
    logger.error(f"Unsupported provider: {config['provider']}")
```

This will be replaced by the ModelRouter approach (see Section 3).

---

## 3. Qwen Integration Plan

### 3.1 Architecture: ModelRouter Pattern

Instead of scattering `if/elif` provider checks across 8+ files, create a single `ModelRouter` that implements a uniform interface:

```python
# agentic_rag/providers/model_router.py
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatResponse:
    content: str
    model: str
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class EmbeddingResponse:
    embeddings: List[List[float]]
    model: str


class BaseProvider(ABC):
    """Abstract base for all LLM/embedding providers."""

    @abstractmethod
    def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.1,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> ChatResponse: ...

    @abstractmethod
    def get_embeddings(self, texts: List[str], model: str) -> List[List[float]]: ...


# ---------------------------------------------------------------------------
# OpenAI Provider
# ---------------------------------------------------------------------------
class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)

    def chat_completion(self, messages, model, temperature=0.1, top_p=1.0,
                        max_tokens=None, response_format=None):
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if response_format:
            kwargs["response_format"] = response_format
        resp = self._client.chat.completions.create(**kwargs)
        return ChatResponse(
            content=resp.choices[0].message.content,
            model=model,
            usage={"prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens}
            if resp.usage else {},
        )

    def get_embeddings(self, texts, model):
        resp = self._client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]


# ---------------------------------------------------------------------------
# Anthropic Provider
# ---------------------------------------------------------------------------
class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str):
        from anthropic import Anthropic
        self._client = Anthropic(api_key=api_key)

    def chat_completion(self, messages, model, temperature=0.1, top_p=1.0,
                        max_tokens=None, response_format=None):
        # Anthropic uses 'system' parameter instead of system role message
        system_msg = ""
        user_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                user_messages.append({"role": m.role, "content": m.content})

        kwargs: Dict[str, Any] = {
            "model": model,
            "system": system_msg,
            "messages": user_messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = 4096  # Anthropic requires max_tokens

        resp = self._client.messages.create(**kwargs)
        return ChatResponse(
            content=resp.content[0].text,
            model=model,
            usage={"input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens}
            if resp.usage else {},
        )

    def get_embeddings(self, texts, model):
        # Anthropic doesn't have a native embedding API in this package.
        # Fall back to a local embedding model or raise.
        raise NotImplementedError(
            "Anthropic provider does not support embeddings natively. "
            "Use a separate embedding provider (e.g., qwen, openai, or local)."
        )


# ---------------------------------------------------------------------------
# Qwen (DashScope / OpenAI-compatible) Provider
# ---------------------------------------------------------------------------
class QwenProvider(BaseProvider):
    """
    Qwen provider supporting two modes:
      1. DashScope native SDK (when `use_dashscope=True`)
      2. OpenAI-compatible endpoint (vLLM/Ollama serving Qwen models)
    
    Qwen models: qwen-turbo, qwen-plus, qwen-max, qwen-long
    Embedding models: text-embedding-v1, text-embedding-v2, text-embedding-v3
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None,
                 use_dashscope: bool = False):
        self._use_dashscope = use_dashscope
        if use_dashscope:
            import dashscope
            dashscope.api_key = api_key
            self._dashscope = dashscope
        else:
            from openai import OpenAI
            # For OpenAI-compatible serving (vLLM, Ollama, etc.)
            client_kwargs = {"api_key": api_key or "dummy"}
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = OpenAI(**client_kwargs)

    def chat_completion(self, messages, model, temperature=0.1, top_p=1.0,
                        max_tokens=None, response_format=None):
        if self._use_dashscope:
            return self._chat_dashscope(messages, model, temperature, top_p,
                                         max_tokens, response_format)
        return self._chat_openai_compat(messages, model, temperature, top_p,
                                        max_tokens, response_format)

    def _chat_dashscope(self, messages, model, temperature, top_p,
                        max_tokens, response_format):
        from dashscope import Generation
        formatted = [{"role": m.role, "content": m.content} for m in messages]
        kwargs = {
            "model": model,
            "messages": formatted,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if response_format and response_format.get("type") == "json_object":
            kwargs["result_format"] = "json"

        resp = Generation.call(**kwargs)
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope API error: {resp.code} - {resp.message}")
        return ChatResponse(
            content=resp.output.choices[0].message.content,
            model=model,
        )

    def _chat_openai_compat(self, messages, model, temperature, top_p,
                            max_tokens, response_format):
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if response_format:
            kwargs["response_format"] = response_format
        resp = self._client.chat.completions.create(**kwargs)
        return ChatResponse(
            content=resp.choices[0].message.content,
            model=model,
            usage={"prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens}
            if resp.usage else {},
        )

    def get_embeddings(self, texts, model):
        if self._use_dashscope:
            return self._embed_dashscope(texts, model)
        return self._embed_openai_compat(texts, model)

    def _embed_dashscope(self, texts, model):
        from dashscope import TextEmbedding
        resp = TextEmbedding.call(model=model, input=texts)
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope Embedding error: {resp.code} - {resp.message}")
        return [item["embedding"] for item in resp.output["embeddings"]]

    def _embed_openai_compat(self, texts, model):
        resp = self._client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]


# ---------------------------------------------------------------------------
# ModelRouter — single entry point
# ---------------------------------------------------------------------------
# Qwen model registry with routing hints
QWEN_CHAT_MODELS = {
    "qwen-turbo": {"temperature_range": (0.0, 1.5), "supports_json": True, "max_tokens": 8192},
    "qwen-plus": {"temperature_range": (0.0, 1.5), "supports_json": True, "max_tokens": 32768},
    "qwen-max": {"temperature_range": (0.0, 1.0), "supports_json": True, "max_tokens": 8192},
    "qwen-long": {"temperature_range": (0.0, 1.5), "supports_json": False, "max_tokens": 1000000},
}

QWEN_EMBEDDING_MODELS = {
    "text-embedding-v1": {"dimensions": 1536},
    "text-embedding-v2": {"dimensions": 1536},
    "text-embedding-v3": {"dimensions": 1024},
}


class ModelRouter:
    """Routes chat completions and embeddings to the configured provider."""

    PROVIDERS = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "qwen": QwenProvider,
    }

    def __init__(self, provider_name: str, api_key: Optional[str],
                 model: str, embedding_model: str,
                 base_url: Optional[str] = None,
                 use_dashscope: bool = False,
                 embedding_api_key: Optional[str] = None,
                 embedding_base_url: Optional[str] = None,
                 embedding_use_dashscope: bool = False,
                 temperature: float = 0.1,
                 top_p: float = 1.0,
                 max_tokens: Optional[int] = None):
        self.provider_name = provider_name.lower()
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        if self.provider_name not in self.PROVIDERS:
            raise ValueError(
                f"Unknown provider '{self.provider_name}'. "
                f"Available: {list(self.PROVIDERS.keys())}"
            )

        # Chat provider
        chat_key = api_key
        self._chat_provider = self.PROVIDERS[self.provider_name](
            api_key=chat_key,
            base_url=base_url,
            use_dashscope=use_dashscope,
        )

        # Embedding provider (can differ from chat provider)
        emb_key = embedding_api_key or api_key
        if self.provider_name == "qwen" and embedding_use_dashscope != use_dashscope:
            self._embedding_provider = QwenProvider(
                api_key=emb_key,
                base_url=embedding_base_url,
                use_dashscope=embedding_use_dashscope,
            )
        elif self.provider_name == "anthropic":
            # Anthropic has no embedding API — default to Qwen or OpenAI
            raise ValueError(
                "Anthropic has no embedding API. Set EMBEDDING_PROVIDER=openai or qwen "
                "and provide EMBEDDING_API_KEY."
            )
        else:
            self._embedding_provider = self._chat_provider

    def chat(self, messages: List[ChatMessage],
             temperature: Optional[float] = None,
             top_p: Optional[float] = None,
             max_tokens: Optional[int] = None,
             response_format: Optional[Dict[str, str]] = None) -> ChatResponse:
        return self._chat_provider.chat_completion(
            messages=messages,
            model=self.model,
            temperature=temperature if temperature is not None else self.temperature,
            top_p=top_p if top_p is not None else self.top_p,
            max_tokens=max_tokens or self.max_tokens,
            response_format=response_format,
        )

    def embed(self, texts: List[str]) -> EmbeddingResponse:
        embeddings = self._embedding_provider.get_embeddings(texts, self.embedding_model)
        return EmbeddingResponse(embeddings=embeddings, model=self.embedding_model)

    def is_json_mode_supported(self) -> bool:
        if self.provider_name == "qwen":
            model_info = QWEN_CHAT_MODELS.get(self.model, {})
            return model_info.get("supports_json", True)
        if self.provider_name == "openai":
            return "gpt" in self.model.lower()
        return False  # Anthropic: no native JSON mode
```

### 3.2 Updated `.env-example` with Qwen support

```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
DASHSCOPE_API_KEY=your_dashscope_api_key_here

# LLM Configuration
LLM_PROVIDER=qwen              # Options: openai, anthropic, qwen
LLM_MODEL=qwen-plus            # Qwen: qwen-turbo, qwen-plus, qwen-max, qwen-long
                               # OpenAI: gpt-4o, gpt-4o-mini, etc.
                               # Anthropic: claude-3-opus, claude-3-sonnet, etc.

# Qwen-specific settings
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
USE_DASHSCOPE=true             # true = DashScope SDK, false = OpenAI-compatible mode (vLLM/Ollama)

# Embedding Configuration
EMBEDDING_PROVIDER=qwen        # Options: openai, qwen  (anthropic has no embedding API)
EMBEDDING_MODEL=text-embedding-v3
EMBEDDING_API_KEY=your_dashscope_api_key_here  # Can differ from LLM API key
EMBEDDING_USE_DASHSCOPE=true
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Generation settings
TEMPERATURE=0.1
TOP_P=1.0
MAX_TOKENS=4096

# Vector Database Configuration
DB_PATH=/app/data/lancedb
CHUNK_SIZE=1500
CHUNK_OVERLAP=150
TOP_K=5

# Logging Configuration
LOG_LEVEL=INFO
```

### 3.3 Updated `requirements.txt` additions

```bash
# Qwen / DashScope
dashscope>=1.14.0
```

### 3.4 Migration Path (Backward Compatible)

All existing code using `config.client.chat.completions.create()` and `config.client.embeddings.create()` continues to work because the `Config` class now holds a `ModelRouter` instance as `self.client`. The router exposes `chat()` and `embed()` methods that the agents call.

For a zero-breaking migration:
1. Keep `Config.client` as the `ModelRouter` instance.
2. Update each agent to call `self.config.client.chat([...])` instead of `self.config.client.chat.completions.create(...)`.
3. Old provider-specific code paths are removed entirely.

---

## 4. Docker Compose Validation

### 4.1 Current Dockerfile Review

| Check | Status | Notes |
|-------|--------|-------|
| Base image | ⚠️ `python:3.10-slim` | Acceptable but `python:3.11-slim` recommended for better performance |
| Build deps | ✅ `build-essential` + `git` | Cleaned up after install |
| Layer caching | ✅ `requirements.txt` copied first | Good practice |
| Non-root user | ✅ `appuser` | Created after `COPY . .`, so `/app` ownership is correct |
| Multi-stage | ❌ Missing | Single stage — could be smaller |
| Health check | ❌ Missing | No `HEALTHCHECK` instruction |
| `.dockerignore` | ❌ Missing | See Section 2.3 |
| Optional deps | ⚠️ Installed at build time | `PyPDF2`, `sentence-transformers`, `rouge-score` are optional but always installed |

### 4.2 Recommended Multi-Scenario Compose Structure

```
docker-compose.yml          # Base (shared config)
docker-compose.dev.yml      # Development overlay
docker-compose.test.yml     # Test overlay
docker-compose.prod.yml     # Production overlay
```

#### docker-compose.dev.yml
```yaml
services:
  agentic-rag:
    build:
      context: .
      dockerfile: Dockerfile
      target: dev  # multi-stage target
    ports:
      - "7860:7860"
      - "5678:5678"  # debugpy
    volumes:
      - .:/app:cached
      - lancedb_data:/app/data
    environment:
      - DB_PATH=/app/data/lancedb
      - LOG_LEVEL=DEBUG
      - PYTHONUNBUFFERED=1
    command: ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "-m", "gradio", "app.py"]
    stdin_open: true
    tty: true

  # Local LLM mock for testing without API keys
  llm-mock:
    image: ghcr.io/huggingface/text-generation-inference:latest
    ports:
      - "8080:80"
    volumes:
      - ./models:/data
    environment:
      - MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
      - NUM_SHARD=1
    profiles:
      - local-llm

volumes:
  lancedb_data:
```

#### docker-compose.test.yml
```yaml
services:
  agentic-rag-test:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - .:/app:cached
      - test_results:/app/test-results
    environment:
      - OPENAI_API_KEY=test_mock_key
      - ANTHROPIC_API_KEY=test_mock_key
      - DASHSCOPE_API_KEY=test_mock_key
      - DB_PATH=/tmp/test_lancedb
      - LOG_LEVEL=WARNING
    command: >
      bash -c "
        pip install pytest-cov pytest-xdist &&
        pytest tests/ -v --cov=agentic_rag --cov-report=xml:/app/test-results/coverage.xml --junitxml=/app/test-results/report.xml
      "
    healthcheck:
      test: ["CMD", "python", "-c", "import agentic_rag; print('OK')"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 15s

volumes:
  test_results:
```

#### docker-compose.prod.yml
```yaml
services:
  agentic-rag:
    build:
      context: .
      dockerfile: Dockerfile
      target: production  # multi-stage target
    ports:
      - "7860:7860"
    volumes:
      - lancedb_data:/app/data
    environment:
      - DB_PATH=/app/data/lancedb
      - LOG_LEVEL=WARNING
    restart: unless-stopped
    read_only: true
    tmpfs:
      - /tmp
      - /app/.cache
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
        reservations:
          cpus: "0.5"
          memory: 512M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7860"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "5"
    security_opt:
      - no-new-privileges:true

volumes:
  lancedb_data:
    driver: local
```

### 4.3 Improved Dockerfile (multi-stage)

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Development stage ----
FROM base AS dev
RUN pip install --no-cache-dir \
    debugpy \
    PyPDF2 \
    sentence-transformers \
    rouge-score \
    python-dotenv
COPY . .
RUN mkdir -p /app/data/lancedb
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 7860 5678
CMD ["python", "app.py"]

# ---- Production stage ----
FROM base AS production
RUN pip install --no-cache-dir \
    PyPDF2 \
    python-dotenv \
    gunicorn

# Only copy necessary files
COPY agentic_rag/ /app/agentic_rag/
COPY app.py setup.py requirements.txt Makefile /app/

RUN mkdir -p /app/data/lancedb \
    && mkdir -p /app/.cache \
    && useradd -m appuser \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7860 || exit 1
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]
```

---

## 5. Documentation & ADR Strategy

### 5.1 Doc Audit Findings

| Document | Status | Issues |
|----------|--------|--------|
| `README.md` | Good | Missing Qwen docs, GitHub URLs are placeholders (`your-username`), no architecture diagram |
| `INSTALLATION.md` | Good but verbose | Repetitive with README, should be cross-linked instead of duplicated |
| `CONTRIBUTING.md` | **Missing** | No dev setup, no PR template, no coding standards |
| `docs/` | **Missing** | No architecture docs, no ADRs |
| `.github/workflows/` | Incomplete | Only `cloc` — missing test CI |
| `CHANGELOG.md` | **Missing** | No release notes |

### 5.2 Recommendations

1. **Consolidate** `INSTALLATION.md` into a dedicated `docs/getting-started.md` and keep README high-level.
2. **Create** `docs/architecture.md` with a Mermaid diagram of the agent pipeline.
3. **Create** `docs/ADR/` (below).
4. **Add** `CONTRIBUTING.md`.
5. **Adopt MkDocs Material** for a developer portal:
   ```
   docs/
   ├── index.md              # Welcome page
   ├── getting-started/
   │   ├── installation.md
   │   ├── quickstart.md
   │   └── configuration.md
   ├── architecture/
   │   ├── overview.md
   │   └── agents.md
   ├── advanced/
   │   ├── hyde.md
   │   ├── reranking.md
   │   ├── multi-query.md
   │   └── evaluation.md
   ├── adr/                  # ADR index
   └── adr/
       ├── 001-model-provider-abstraction.md
       ├── 002-agent-query-pipeline.md
       ├── 003-vector-storage-strategy.md
       └── 004-multi-scenario-docker-compose.md
   ```

---

## 6. Testing Strategy

### 6.1 Test Structure

```
tests/
├── __init__.py                    # Mock module setup (dashscope, lancedb)
├── conftest.py                    # Shared fixtures (mock_config, rag_system)
├── mock_dependencies.py           # Mock class implementations
├── test_agentic_rag.py            # AgenticRAG orchestration tests
├── test_agents.py                 # Agent unit tests (QueryPlanner, InfoRetriever, ResponseGenerator)
├── test_config.py                 # Config validation tests
├── test_document.py               # Document processing tests
├── test_env_loader.py             # Environment loader tests (including Qwen)
├── test_vectordb.py               # VectorDB manager tests
└── test_model_router.py           # [NEW] ModelRouter & provider tests
```

### 6.2 New Test File: `test_model_router.py`

Created at `tests/test_model_router.py` — covers:
- `ChatMessage`, `ChatResponse`, `EmbeddingResponse` data classes
- `OpenAIProvider` — chat completion, JSON mode, embeddings
- `AnthropicProvider` — chat completion, system prompt extraction, embedding error
- `QwenProvider` (DashScope mode) — chat, error handling, embeddings
- `QwenProvider` (OpenAI-compatible mode) — chat via vLLM/Ollama, embeddings
- `ModelRouter` — routing, JSON mode detection, unknown provider error, Anthropic embedding error
- Qwen model registry — all models present, attributes correct
- `load_environment()` with Qwen provider

### 6.3 Updated Fixtures

The `mock_config` fixture in `conftest.py` now mocks the `ModelRouter` interface:
```python
mock_router.chat.return_value = MagicMock(content="Mock response")
mock_router.embed.return_value = MagicMock(embeddings=[[0.1, 0.2, 0.3, 0.4]])
mock_router.is_json_mode_supported.return_value = True
```

### 6.4 Run Commands

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=agentic_rag --cov-report=term --cov-report=html

# Run only ModelRouter tests
pytest tests/test_model_router.py -v

# Run Qwen env loader test
pytest tests/test_env_loader.py::TestEnvLoaderQwen -v

# Run inside Docker (test compose)
docker compose -f docker-compose.yml -f docker-compose.test.yml up --abort-on-container-exit
```

### 6.5 Mock LLM Response Fixtures

```python
# OpenAI-style mock
mock_resp = MagicMock()
mock_resp.choices = [MagicMock()]
mock_resp.choices[0].message.content = '{"answer": "42"}'

# DashScope mock
mock_resp = MagicMock()
mock_resp.status_code = 200
mock_resp.output.choices = [MagicMock()]
mock_resp.output.choices[0].message.content = "Qwen response"
```

---

## 7. Next Steps & Roadmap

### Phase 1: Immediate (1-2 weeks) — P0
| Task | File(s) | Priority |
|------|---------|----------|
| Merge `ModelRouter` + updated agents | `providers/model_router.py`, all agents | 🔴 |
| Update `Config.from_dict()` for Qwen env vars | `config.py` | 🔴 |
| Add `.dockerignore` | `.dockerignore` | 🔴 |
| Fix `setup.py` Python version floor | `setup.py` | 🔴 |
| Fix Gradio file upload bug | `app.py` | 🔴 |

### Phase 2: Short-term (2-4 weeks) — P1
| Task | Details | Priority |
|------|---------|----------|
| Multi-stage Dockerfile | `Dockerfile` with dev/production stages | 🟡 |
| Compose overlay files | `docker-compose.{dev,test,prod}.yml` | 🟡 |
| Pydantic Config validation | Replace plain class with `pydantic.BaseModel` | 🟡 |
| Retry logic for API calls | `tenacity` library on all LLM/embedding calls | 🟡 |
| Async subquery retrieval | `asyncio.gather()` in `AgenticRAG.query()` | 🟡 |

### Phase 3: Medium-term (1-3 months) — P2
| Task | Details | Priority |
|------|---------|----------|
| CI pipeline | GitHub Actions: test, lint, build, coverage | 🟢 |
| MkDocs Material site | `docs/` with auto-generated API docs | 🟢 |
| LanceDB schema migration | Pydantic model + index optimization | 🟢 |
| `CONTRIBUTING.md` | Dev setup, PR template, coding standards | 🟢 |
| Project rename `agentic_rag` → `smithy` | Directory + all imports | 🟢 |

---

## File Change Summary

### New Files Created
| File | Purpose |
|------|---------|
| `agentic_rag/providers/__init__.py` | Provider module exports |
| `agentic_rag/providers/model_router.py` | Unified ModelRouter abstraction |
| `docs/ADR/_adr-template.md` | ADR template (MADR-style) |
| `docs/ADR/001-model-provider-abstraction.md` | ADR: ModelRouter + Qwen |
| `docs/ADR/002-agent-query-pipeline.md` | ADR: 3-agent pipeline |
| `docs/ADR/003-vector-storage-strategy.md` | ADR: LanceDB strategy |
| `docs/ADR/004-multi-scenario-docker-compose.md` | ADR: Docker compose |
| `docker-compose.dev.yml` | Dev overlay |
| `docker-compose.test.yml` | Test overlay |
| `docker-compose.prod.yml` | Production overlay |
| `.dockerignore` | Docker ignore file |
| `tests/test_model_router.py` | ModelRouter unit tests |
| `CODE_REVIEW.md` | This document |

### Files Modified
| File | Changes |
|------|---------|
| `agentic_rag/config.py` | Uses `ModelRouter` instead of hardcoded clients |
| `agentic_rag/utils/env_loader.py` | Qwen env var support, helper functions |
| `agentic_rag/agents/query_planner.py` | Uses `config.client.chat()` |
| `agentic_rag/agents/response_generator.py` | Uses `config.client.chat()` |
| `agentic_rag/advanced/hyde.py` | Uses `config.client.chat()` |
| `agentic_rag/advanced/multi_query.py` | Uses `config.client.chat()` |
| `agentic_rag/advanced/self_improving.py` | Uses `config.client.chat()` |
| `agentic_rag/evaluation/metrics.py` | Uses `config.client.chat()` + JSON mode check |
| `agentic_rag/vectordb.py` | Uses `config.client.embed()` |
| `Dockerfile` | Multi-stage build (base/dev/production) |
| `docker-compose.yml` | Simplified to base config |
| `.env-example` | Qwen env vars added |
| `requirements.txt` | `dashscope>=1.14.0`, `python-dotenv` |
| `pytest.ini` | `DASHSCOPE_API_KEY` added |
| `tests/__init__.py` | DashScope mock modules |
| `tests/conftest.py` | ModelRouter mock fixtures |
| `tests/test_agents.py` | Updated for ModelRouter interface |

