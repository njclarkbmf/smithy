"""
ModelRouter — Unified provider abstraction for LLM chat completions and embeddings.

Supports OpenAI, Anthropic, and Qwen (via DashScope SDK or OpenAI-compatible endpoints).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Qwen model registry
# ---------------------------------------------------------------------------
QWEN_CHAT_MODELS: Dict[str, Dict[str, Any]] = {
    "qwen-turbo": {
        "temperature_range": (0.0, 1.5),
        "supports_json": True,
        "max_tokens": 8192,
    },
    "qwen-plus": {
        "temperature_range": (0.0, 1.5),
        "supports_json": True,
        "max_tokens": 32768,
    },
    "qwen-max": {
        "temperature_range": (0.0, 1.0),
        "supports_json": True,
        "max_tokens": 8192,
    },
    "qwen-long": {
        "temperature_range": (0.0, 1.5),
        "supports_json": False,
        "max_tokens": 1_000_000,
    },
}

QWEN_EMBEDDING_MODELS: Dict[str, Dict[str, Any]] = {
    "text-embedding-v1": {"dimensions": 1536},
    "text-embedding-v2": {"dimensions": 1536},
    "text-embedding-v3": {"dimensions": 1024},
}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class BaseProvider(ABC):
    """Abstract base for all LLM / embedding providers."""

    @abstractmethod
    def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.1,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> ChatResponse:
        """Generate a chat completion."""
        ...

    @abstractmethod
    def get_embeddings(
        self, texts: List[str], model: str
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        ...


# ---------------------------------------------------------------------------
# OpenAI Provider
# ---------------------------------------------------------------------------
class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        from openai import OpenAI

        kwargs: Dict[str, str] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.1,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> ChatResponse:
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
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
        return ChatResponse(
            content=resp.choices[0].message.content,
            model=model,
            usage=usage,
        )

    def get_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        resp = self._client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]


# ---------------------------------------------------------------------------
# Anthropic Provider
# ---------------------------------------------------------------------------
class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str):
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)

    def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.1,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> ChatResponse:
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
            kwargs["max_tokens"] = 4096  # Anthropic requires this

        resp = self._client.messages.create(**kwargs)
        usage = {}
        if resp.usage:
            usage = {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            }
        return ChatResponse(
            content=resp.content[0].text,
            model=model,
            usage=usage,
        )

    def get_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        raise NotImplementedError(
            "Anthropic does not provide an embedding API. "
            "Configure a separate embedding provider (openai or qwen)."
        )


# ---------------------------------------------------------------------------
# Qwen (DashScope / OpenAI-compatible) Provider
# ---------------------------------------------------------------------------
class QwenProvider(BaseProvider):
    """
    Qwen provider supporting two modes:

    1. **DashScope native SDK** — when ``use_dashscope=True``
    2. **OpenAI-compatible endpoint** — when ``use_dashscope=False``
       (vLLM, Ollama, or any OpenAI-compatible server serving Qwen models)
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        use_dashscope: bool = False,
    ):
        self._use_dashscope = use_dashscope
        self._api_key = api_key
        self._base_url = base_url

        if use_dashscope:
            import dashscope

            dashscope.api_key = api_key
            self._dashscope = dashscope
        else:
            from openai import OpenAI

            client_kwargs: Dict[str, str] = {
                "api_key": api_key or "dummy",
            }
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = OpenAI(**client_kwargs)

    # ---- Chat completion ----

    def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.1,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> ChatResponse:
        if self._use_dashscope:
            return self._chat_dashscope(
                messages, model, temperature, top_p, max_tokens, response_format
            )
        return self._chat_openai_compat(
            messages, model, temperature, top_p, max_tokens, response_format
        )

    def _chat_dashscope(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, str]],
    ) -> ChatResponse:
        from dashscope import Generation

        formatted = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        kwargs: Dict[str, Any] = {
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
            raise RuntimeError(
                f"DashScope API error: {resp.code} — {resp.message}"
            )
        content = resp.output.choices[0].message.content
        return ChatResponse(content=content, model=model)

    def _chat_openai_compat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, str]],
    ) -> ChatResponse:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if response_format:
            kwargs["response_format"] = response_format

        resp = self._client.chat.completions.create(**kwargs)
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
        return ChatResponse(
            content=resp.choices[0].message.content,
            model=model,
            usage=usage,
        )

    # ---- Embeddings ----

    def get_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        if self._use_dashscope:
            return self._embed_dashscope(texts, model)
        return self._embed_openai_compat(texts, model)

    def _embed_dashscope(
        self, texts: List[str], model: str
    ) -> List[List[float]]:
        from dashscope import TextEmbedding

        resp = TextEmbedding.call(model=model, input=texts)
        if resp.status_code != 200:
            raise RuntimeError(
                f"DashScope Embedding error: {resp.code} — {resp.message}"
            )
        return [item["embedding"] for item in resp.output["embeddings"]]

    def _embed_openai_compat(
        self, texts: List[str], model: str
    ) -> List[List[float]]:
        resp = self._client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]


# ---------------------------------------------------------------------------
# ModelRouter — single entry point
# ---------------------------------------------------------------------------
class ModelRouter:
    """
    Routes chat completions and embeddings to the configured provider.

    Supports separate providers for chat vs. embeddings (e.g. Anthropic chat
    + OpenAI embeddings).
    """

    _PROVIDER_MAP = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "qwen": QwenProvider,
    }

    def __init__(
        self,
        provider_name: str,
        api_key: Optional[str],
        model: str,
        embedding_model: str,
        base_url: Optional[str] = None,
        use_dashscope: bool = False,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_use_dashscope: bool = False,
        temperature: float = 0.1,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
    ):
        self.provider_name = provider_name.lower()
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        if self.provider_name not in self._PROVIDER_MAP:
            raise ValueError(
                f"Unknown provider '{self.provider_name}'. "
                f"Available: {list(self._PROVIDER_MAP.keys())}"
            )

        chat_key = api_key or ""
        provider_cls = self._PROVIDER_MAP[self.provider_name]

        # Chat provider
        if self.provider_name == "qwen":
            self._chat_provider = provider_cls(
                api_key=chat_key,
                base_url=base_url,
                use_dashscope=use_dashscope,
            )
        elif self.provider_name == "openai":
            self._chat_provider = provider_cls(
                api_key=chat_key,
                base_url=base_url,
            )
        else:
            self._chat_provider = provider_cls(api_key=chat_key)

        # Embedding provider — may differ from chat provider
        emb_key = embedding_api_key or api_key or ""
        emb_provider_name = self.provider_name
        if emb_provider_name == "anthropic":
            raise ValueError(
                "Anthropic has no embedding API. Set EMBEDDING_PROVIDER=openai "
                "or EMBEDDING_PROVIDER=qwen and provide EMBEDDING_API_KEY."
            )

        emb_cls = self._PROVIDER_MAP.get(emb_provider_name)
        if emb_cls is None:
            raise ValueError(f"Unknown embedding provider: {emb_provider_name}")

        if emb_provider_name == "qwen":
            self._embedding_provider = emb_cls(
                api_key=emb_key,
                base_url=embedding_base_url or base_url,
                use_dashscope=embedding_use_dashscope or use_dashscope,
            )
        elif emb_provider_name == "openai":
            self._embedding_provider = emb_cls(
                api_key=emb_key,
                base_url=embedding_base_url,
            )
        else:
            self._embedding_provider = emb_cls(api_key=emb_key)

    # ---- Public API ----

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> ChatResponse:
        """Generate a chat completion."""
        return self._chat_provider.chat_completion(
            messages=messages,
            model=self.model,
            temperature=temperature if temperature is not None else self.temperature,
            top_p=top_p if top_p is not None else self.top_p,
            max_tokens=max_tokens or self.max_tokens,
            response_format=response_format,
        )

    def embed(self, texts: List[str]) -> EmbeddingResponse:
        """Generate embeddings for a list of texts."""
        embeddings = self._embedding_provider.get_embeddings(
            texts, self.embedding_model
        )
        return EmbeddingResponse(embeddings=embeddings, model=self.embedding_model)

    def is_json_mode_supported(self) -> bool:
        """Return True if the current model supports JSON-formatted output."""
        if self.provider_name == "qwen":
            info = QWEN_CHAT_MODELS.get(self.model, {})
            return info.get("supports_json", True)
        if self.provider_name == "openai":
            return "gpt" in self.model.lower()
        return False

    @property
    def provider(self) -> str:
        """Alias for provider_name (for compatibility with existing code)."""
        return self.provider_name
