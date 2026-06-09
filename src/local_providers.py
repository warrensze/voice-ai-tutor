"""Local LLM and embedding provider adapters."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama, OllamaEmbeddings

from settings_store import UserSettings

LLAMACPP_CHAT_TIMEOUT_SECONDS = float(
    os.getenv("VOICE_TUTOR_LLM_TIMEOUT_SECONDS", "30")
)


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str = "local",
    timeout: float = 120.0,
    stream: bool = False,
):
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, data=data, headers=headers, method="POST")
    return urlopen(request, timeout=timeout)


class LocalOpenAIChat(Runnable[Any, str]):
    """Minimal OpenAI-compatible chat runnable for llama.cpp."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "local",
        temperature: float = 0.7,
        max_tokens: int = 220,
        timeout: float = 120.0,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def invoke(self, input: Any, config: Any | None = None, **kwargs: Any) -> str:
        payload = self._payload(input, stream=False)
        try:
            with _post_json(
                _join_url(self.base_url, "/chat/completions"),
                payload,
                api_key=self.api_key,
                timeout=self.timeout,
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except URLError as error:
            raise RuntimeError(f"llama.cpp chat endpoint is unavailable: {error}") from error

        choices = body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")

    def stream(self, input: Any, config: Any | None = None, **kwargs: Any):
        payload = self._payload(input, stream=True)
        try:
            with _post_json(
                _join_url(self.base_url, "/chat/completions"),
                payload,
                api_key=self.api_key,
                timeout=self.timeout,
                stream=True,
            ) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            yield str(content)
        except URLError as error:
            raise RuntimeError(f"llama.cpp chat endpoint is unavailable: {error}") from error

    def _payload(self, input: Any, *, stream: bool) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": self._prepare_messages(input),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def _prepare_messages(self, input: Any) -> list[dict[str, str]]:
        messages = _messages_from_input(input)
        if "qwen3" not in self.model.lower():
            return messages

        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = str(message.get("content") or "")
            lower = content.lower()
            if "/no_think" not in lower and "/think" not in lower:
                message["content"] = f"{content}\n/no_think"
            break
        return messages


class LocalOpenAIEmbeddings(Embeddings):
    """OpenAI-compatible embedding adapter for llama.cpp embedding servers."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "local",
        timeout: float = 120.0,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model, "input": texts}
        try:
            with _post_json(
                _join_url(self.base_url, "/embeddings"),
                payload,
                api_key=self.api_key,
                timeout=self.timeout,
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except URLError as error:
            raise RuntimeError(
                f"llama.cpp embedding endpoint is unavailable: {error}"
            ) from error

        rows = sorted(body.get("data") or [], key=lambda item: int(item.get("index", 0)))
        return [list(row.get("embedding") or []) for row in rows]

    def embed_query(self, text: str) -> list[float]:
        results = self.embed_documents([text])
        return results[0] if results else []


def _messages_from_input(input: Any) -> list[dict[str, str]]:
    if hasattr(input, "to_messages"):
        messages = input.to_messages()
    elif isinstance(input, list):
        messages = input
    else:
        messages = [HumanMessage(content=str(input))]

    converted = []
    for message in messages:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, BaseMessage):
            role = getattr(message, "type", "user")
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
        elif isinstance(message, dict):
            converted.append(
                {
                    "role": str(message.get("role") or "user"),
                    "content": str(message.get("content") or ""),
                }
            )
            continue
        else:
            role = "user"

        converted.append({"role": role, "content": str(getattr(message, "content", ""))})

    return converted


def create_chat_model(settings: UserSettings, *, num_predict: int = 220):
    """Create the configured local chat model."""
    if settings.llm_provider == "ollama":
        return ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            streaming=True,
            temperature=0.7,
            num_predict=num_predict,
        )

    return LocalOpenAIChat(
        base_url=settings.llamacpp_chat_base_url,
        model=settings.llamacpp_chat_model,
        api_key=settings.llamacpp_api_key,
        temperature=0.7,
        max_tokens=num_predict,
        timeout=LLAMACPP_CHAT_TIMEOUT_SECONDS,
    )


def create_embedding_model(settings: UserSettings):
    """Create embeddings for the configured local embedding provider."""
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )

    return LocalOpenAIEmbeddings(
        base_url=settings.llamacpp_embedding_base_url,
        model=settings.llamacpp_embedding_model,
        api_key=settings.llamacpp_api_key,
    )


def provider_status(settings: UserSettings) -> dict[str, Any]:
    """Return configured provider labels and endpoints for UI diagnostics."""
    if settings.llm_provider == "ollama":
        chat_endpoint = settings.ollama_base_url
        chat_model = settings.ollama_chat_model
    else:
        chat_endpoint = settings.llamacpp_chat_base_url
        chat_model = settings.llamacpp_chat_model

    if settings.embedding_provider == "ollama":
        embedding_endpoint = settings.ollama_base_url
        embedding_model = settings.ollama_embedding_model
    else:
        embedding_endpoint = settings.llamacpp_embedding_base_url
        embedding_model = settings.llamacpp_embedding_model

    return {
        "llm_provider": settings.llm_provider,
        "chat_endpoint": chat_endpoint,
        "chat_model": chat_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_endpoint": embedding_endpoint,
        "embedding_model": embedding_model,
        "tts_backend": settings.tts_backend,
    }
