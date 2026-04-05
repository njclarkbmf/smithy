"""
Tests for the ModelRouter and provider abstraction.
"""
import os
import json
import pytest
from unittest.mock import MagicMock, patch

from agentic_rag.providers.model_router import (
    BaseProvider,
    ChatMessage,
    ChatResponse,
    EmbeddingResponse,
    ModelRouter,
    OpenAIProvider,
    AnthropicProvider,
    QwenProvider,
    QWEN_CHAT_MODELS,
    QWEN_EMBEDDING_MODELS,
)


# ===================================================================
# ChatMessage / ChatResponse / EmbeddingResponse
# ===================================================================
class TestDataClasses:
    def test_chat_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_response(self):
        resp = ChatResponse(content="Hi", model="gpt-4o")
        assert resp.content == "Hi"
        assert resp.model == "gpt-4o"
        assert resp.finish_reason == "stop"
        assert resp.usage == {}

    def test_embedding_response(self):
        resp = EmbeddingResponse(embeddings=[[0.1, 0.2]], model="text-embedding-v3")
        assert len(resp.embeddings) == 1
        assert resp.model == "text-embedding-v3"


# ===================================================================
# OpenAIProvider
# ===================================================================
class TestOpenAIProvider:
    @pytest.fixture
    def provider(self):
        with patch("openai.OpenAI") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # Mock chat completion
            mock_chat_resp = MagicMock()
            mock_chat_resp.choices = [MagicMock()]
            mock_chat_resp.choices[0].message.content = "OpenAI says hi"
            mock_chat_resp.usage = MagicMock(
                prompt_tokens=10, completion_tokens=20
            )
            mock_instance.chat.completions.create.return_value = mock_chat_resp

            # Mock embeddings
            mock_emb_resp = MagicMock()
            mock_emb_resp.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
            mock_instance.embeddings.create.return_value = mock_emb_resp

            prov = OpenAIProvider(api_key="test_key")
            prov._client = mock_instance
            yield prov

    def test_chat_completion(self, provider):
        messages = [
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="Hello"),
        ]
        resp = provider.chat_completion(messages, model="gpt-4o", temperature=0.5)
        assert resp.content == "OpenAI says hi"
        assert resp.model == "gpt-4o"
        assert resp.usage["prompt_tokens"] == 10
        provider._client.chat.completions.create.assert_called_once()
        call_kwargs = provider._client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert len(call_kwargs["messages"]) == 2

    def test_chat_completion_with_response_format(self, provider):
        messages = [ChatMessage(role="user", content="Give JSON")]
        provider.chat_completion(
            messages,
            model="gpt-4o",
            response_format={"type": "json_object"},
        )
        call_kwargs = provider._client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_get_embeddings(self, provider):
        embeddings = provider.get_embeddings(["hello"], model="text-embedding-3-small")
        assert embeddings == [[0.1, 0.2, 0.3]]
        provider._client.embeddings.create.assert_called_once()


# ===================================================================
# AnthropicProvider
# ===================================================================
class TestAnthropicProvider:
    @pytest.fixture
    def provider(self):
        with patch("anthropic.Anthropic") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            mock_resp = MagicMock()
            mock_content = MagicMock()
            mock_content.text = "Claude says hi"
            mock_resp.content = [mock_content]
            mock_resp.usage = MagicMock(input_tokens=5, output_tokens=15)
            mock_instance.messages.create.return_value = mock_resp

            prov = AnthropicProvider(api_key="test_key")
            prov._client = mock_instance
            yield prov

    def test_chat_completion(self, provider):
        messages = [
            ChatMessage(role="system", content="Be helpful"),
            ChatMessage(role="user", content="Hi"),
        ]
        resp = provider.chat_completion(messages, model="claude-3-opus")
        assert resp.content == "Claude says hi"
        call_kwargs = provider._client.messages.create.call_args[1]
        assert call_kwargs["system"] == "Be helpful"
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    def test_chat_completion_defaults_max_tokens(self, provider):
        messages = [ChatMessage(role="user", content="Hi")]
        provider.chat_completion(messages, model="claude-3-opus")
        call_kwargs = provider._client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 4096  # default when None

    def test_get_embeddings_raises(self, provider):
        with pytest.raises(NotImplementedError, match="embedding"):
            provider.get_embeddings(["hello"], model="n/a")


# ===================================================================
# QwenProvider
# ===================================================================
class TestQwenProviderDashscope:
    """Qwen provider using DashScope SDK."""

    @pytest.fixture
    def provider(self):
        with patch("dashscope.api_key"), \
             patch("dashscope.Generation") as mock_gen, \
             patch("dashscope.TextEmbedding") as mock_emb:

            # Mock chat
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_choice = MagicMock()
            mock_choice.message.content = "Qwen says hi"
            mock_resp.output.choices = [mock_choice]
            mock_gen.call.return_value = mock_resp

            # Mock embeddings
            mock_emb_resp = MagicMock()
            mock_emb_resp.status_code = 200
            mock_emb_resp.output = {
                "embeddings": [{"embedding": [0.1, 0.2, 0.3]}]
            }
            mock_emb.call.return_value = mock_emb_resp

            prov = QwenProvider(
                api_key="test_key", use_dashscope=True
            )
            yield prov

    def test_chat_completion_dashscope(self, provider):
        messages = [ChatMessage(role="user", content="Hello")]
        resp = provider.chat_completion(messages, model="qwen-plus")
        assert resp.content == "Qwen says hi"

    def test_chat_completion_error(self, provider):
        from dashscope import Generation
        mock_err = MagicMock()
        mock_err.status_code = 400
        mock_err.code = "InvalidParameter"
        mock_err.message = "Bad request"
        Generation.call.return_value = mock_err
        messages = [ChatMessage(role="user", content="Hello")]
        with pytest.raises(RuntimeError, match="DashScope API error"):
            provider.chat_completion(messages, model="qwen-plus")

    def test_get_embeddings_dashscope(self, provider):
        embeddings = provider.get_embeddings(["hello"], model="text-embedding-v3")
        assert embeddings == [[0.1, 0.2, 0.3]]


class TestQwenProviderOpenAICompat:
    """Qwen provider using OpenAI-compatible endpoint (vLLM/Ollama)."""

    @pytest.fixture
    def provider(self):
        with patch("openai.OpenAI") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            mock_chat_resp = MagicMock()
            mock_chat_resp.choices = [MagicMock()]
            mock_chat_resp.choices[0].message.content = "vLLM Qwen says hi"
            mock_chat_resp.usage = None
            mock_instance.chat.completions.create.return_value = mock_chat_resp

            mock_emb_resp = MagicMock()
            mock_emb_resp.data = [MagicMock(embedding=[0.4, 0.5, 0.6])]
            mock_instance.embeddings.create.return_value = mock_emb_resp

            prov = QwenProvider(
                api_key="dummy",
                base_url="http://localhost:8000/v1",
                use_dashscope=False,
            )
            prov._client = mock_instance
            yield prov

    def test_chat_completion_openai_compat(self, provider):
        messages = [ChatMessage(role="user", content="Hello")]
        resp = provider.chat_completion(messages, model="qwen-plus")
        assert resp.content == "vLLM Qwen says hi"

    def test_get_embeddings_openai_compat(self, provider):
        embeddings = provider.get_embeddings(["hello"], model="text-embedding-v3")
        assert embeddings == [[0.4, 0.5, 0.6]]


# ===================================================================
# ModelRouter
# ===================================================================
class TestModelRouter:
    @pytest.fixture
    def openai_router(self):
        with patch("openai.OpenAI") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "Routed response"
            mock_resp.usage = None
            mock_instance.chat.completions.create.return_value = mock_resp
            mock_instance.embeddings.create.return_value = MagicMock(
                data=[MagicMock(embedding=[0.1] * 4)]
            )
            router = ModelRouter(
                provider_name="openai",
                api_key="test_key",
                model="gpt-4o",
                embedding_model="text-embedding-3-small",
            )
            router._chat_provider._client = mock_instance
            router._embedding_provider._client = mock_instance
            yield router

    def test_router_chat(self, openai_router):
        messages = [ChatMessage(role="user", content="Hi")]
        resp = openai_router.chat(messages, temperature=0.3)
        assert resp.content == "Routed response"

    def test_router_embed(self, openai_router):
        resp = openai_router.embed(["hello world"])
        assert resp.model == "text-embedding-3-small"
        assert resp.embeddings == [[0.1] * 4]

    def test_router_json_mode(self, openai_router):
        assert openai_router.is_json_mode_supported() is True

    def test_router_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            ModelRouter(
                provider_name="unknown",
                api_key="x",
                model="x",
                embedding_model="x",
            )

    def test_router_anthropic_no_embed(self):
        with pytest.raises(ValueError, match="Anthropic.*embedding"):
            ModelRouter(
                provider_name="anthropic",
                api_key="test_key",
                model="claude-3-opus",
                embedding_model="n/a",
            )


# ===================================================================
# Qwen Model Registry
# ===================================================================
class TestQwenModelRegistry:
    def test_qwen_chat_models_exist(self):
        assert "qwen-turbo" in QWEN_CHAT_MODELS
        assert "qwen-plus" in QWEN_CHAT_MODELS
        assert "qwen-max" in QWEN_CHAT_MODELS
        assert "qwen-long" in QWEN_CHAT_MODELS

    def test_qwen_chat_model_attrs(self):
        for name, info in QWEN_CHAT_MODELS.items():
            assert "temperature_range" in info
            assert "supports_json" in info
            assert "max_tokens" in info

    def test_qwen_embedding_models_exist(self):
        assert "text-embedding-v1" in QWEN_EMBEDDING_MODELS
        assert "text-embedding-v2" in QWEN_EMBEDDING_MODELS
        assert "text-embedding-v3" in QWEN_EMBEDDING_MODELS

    def test_qwen_embedding_model_dimensions(self):
        for name, info in QWEN_EMBEDDING_MODELS.items():
            assert "dimensions" in info
            assert isinstance(info["dimensions"], int)


# ===================================================================
# Integration: env_loader loads Qwen config
# ===================================================================
class TestEnvLoaderQwen:
    def test_load_environment_qwen(self):
        mock_env = {
            "DASHSCOPE_API_KEY": "qwen_test_key",
            "LLM_PROVIDER": "qwen",
            "LLM_MODEL": "qwen-plus",
            "EMBEDDING_PROVIDER": "qwen",
            "EMBEDDING_MODEL": "text-embedding-v3",
            "USE_DASHSCOPE": "true",
            "TEMPERATURE": "0.2",
            "DB_PATH": "./test_db",
            "CHUNK_SIZE": "1000",
            "CHUNK_OVERLAP": "100",
            "TOP_K": "3",
        }
        with patch("dotenv.load_dotenv"), \
             patch.dict(os.environ, mock_env, clear=True):
            from agentic_rag.utils.env_loader import load_environment
            config = load_environment()
            assert config["provider"] == "qwen"
            assert config["model"] == "qwen-plus"
            assert config["embedding_model"] == "text-embedding-v3"
