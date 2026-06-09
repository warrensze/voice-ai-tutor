"""Self-starting llama.cpp runtime for local chat and embeddings."""

from __future__ import annotations

import atexit
from dataclasses import dataclass, asdict
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from settings_store import DATA_DIR, UserSettings

QWEN3_CHAT_MODEL = "Qwen/Qwen3-8B-GGUF:Q4_K_M"
NOMIC_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5-GGUF:Q4_K_M"

CHAT_PORT = 8080
EMBED_PORT = 8081

LLAMACPP_AUTO_BOOTSTRAP_ENV = "LLAMACPP_AUTO_BOOTSTRAP"
LLAMACPP_AUTO_INSTALL_ENV = "LLAMACPP_AUTO_INSTALL"

LOG_DIR = DATA_DIR / "logs"
_state_lock = threading.Lock()
_bootstrap_thread: threading.Thread | None = None
_managed_processes: dict[str, subprocess.Popen] = {}


@dataclass
class ServerState:
    role: str
    model: str
    endpoint: str
    status: str = "unknown"
    message: str = ""
    pid: int | None = None
    log_path: str = ""


@dataclass
class BootstrapState:
    enabled: bool = True
    status: str = "idle"
    message: str = ""
    llama_server_path: str = ""
    auto_install: bool = True
    chat: ServerState | None = None
    embedding: ServerState | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.chat is None:
            payload["chat"] = None
        if self.embedding is None:
            payload["embedding"] = None
        return payload


_state = BootstrapState(
    chat=ServerState(
        role="chat",
        model=QWEN3_CHAT_MODEL,
        endpoint=f"http://127.0.0.1:{CHAT_PORT}/v1",
    ),
    embedding=ServerState(
        role="embedding",
        model=NOMIC_EMBED_MODEL,
        endpoint=f"http://127.0.0.1:{EMBED_PORT}/v1",
    ),
)


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _base_health_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}/health"


def _server_ready(base_url: str, *, timeout: float = 1.0) -> bool:
    health_url = _base_health_url(base_url)
    if not health_url:
        return False
    try:
        with urlopen(health_url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def _find_llama_server() -> str:
    candidates = [
        os.getenv("LLAMA_SERVER_BIN", ""),
        shutil.which("llama-server") or "",
        "/opt/homebrew/bin/llama-server",
        "/usr/local/bin/llama-server",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return ""


def _find_brew() -> str:
    return shutil.which("brew") or "/opt/homebrew/bin/brew"


def _install_llamacpp_with_brew() -> str:
    brew = _find_brew()
    if not brew or not Path(brew).exists():
        raise RuntimeError(
            "llama-server is missing and Homebrew was not found for automatic install."
        )
    result = subprocess.run(
        [brew, "install", "llama.cpp"],
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Homebrew could not install llama.cpp. {details}")
    server = _find_llama_server()
    if not server:
        raise RuntimeError("Homebrew finished, but llama-server was still not found.")
    return server


def _start_server(
    *,
    role: str,
    server_path: str,
    model: str,
    endpoint: str,
    port: int,
    embedding: bool = False,
) -> ServerState:
    state = ServerState(role=role, model=model, endpoint=endpoint)
    if _server_ready(endpoint):
        state.status = "ready"
        state.message = "Already running."
        return state

    existing = _managed_processes.get(role)
    if existing is not None and existing.poll() is None:
        state.status = "starting"
        state.message = "Managed process is still loading."
        state.pid = existing.pid
        return state

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"llamacpp-{role}.log"
    command = [
        server_path,
        "-hf",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        "2048" if embedding else "8192",
    ]
    if embedding:
        command.extend(["--embedding", "--pooling", "cls"])
    else:
        command.extend(
            [
                "--jinja",
                "--reasoning-format",
                "deepseek",
                "-ngl",
                "99",
                "--temp",
                "0.6",
                "--top-k",
                "20",
                "--top-p",
                "0.95",
                "--min-p",
                "0",
            ]
        )

    env = dict(os.environ)
    # The app is offline for HF-backed Python libraries, but llama.cpp needs
    # network access once to fetch GGUF files if they are not cached locally.
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    env.pop("HF_DATASETS_OFFLINE", None)

    handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(DATA_DIR.parent),
    )
    _managed_processes[role] = process
    state.status = "starting"
    state.message = "Starting llama-server; first launch may download the GGUF model."
    state.pid = process.pid
    state.log_path = str(log_path)
    return state


def _wait_until_ready(state: ServerState, *, timeout: float = 10.0) -> ServerState:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _server_ready(state.endpoint):
            state.status = "ready"
            state.message = "Ready."
            return state
        process = _managed_processes.get(state.role)
        if process is not None and process.poll() is not None:
            state.status = "error"
            state.message = f"llama-server exited with code {process.returncode}."
            return state
        time.sleep(0.5)
    return state


def _bootstrap(settings: UserSettings) -> None:
    global _state
    enabled = _env_enabled(LLAMACPP_AUTO_BOOTSTRAP_ENV, True)
    auto_install = _env_enabled(LLAMACPP_AUTO_INSTALL_ENV, True)
    with _state_lock:
        _state.enabled = enabled
        _state.auto_install = auto_install
        _state.status = "starting" if enabled else "disabled"
        _state.message = "Preparing local llama.cpp runtime." if enabled else ""
        _state.chat = ServerState(
            role="chat",
            model=settings.llamacpp_chat_model,
            endpoint=settings.llamacpp_chat_base_url,
        )
        _state.embedding = ServerState(
            role="embedding",
            model=settings.llamacpp_embedding_model,
            endpoint=settings.llamacpp_embedding_base_url,
        )

    if not enabled:
        return

    try:
        server_path = _find_llama_server()
        if not server_path and auto_install:
            with _state_lock:
                _state.status = "installing"
                _state.message = "Installing llama.cpp with Homebrew."
            server_path = _install_llamacpp_with_brew()
        if not server_path:
            raise RuntimeError("llama-server was not found.")

        chat_state = _start_server(
            role="chat",
            server_path=server_path,
            model=settings.llamacpp_chat_model,
            endpoint=settings.llamacpp_chat_base_url,
            port=CHAT_PORT,
        )
        embed_state = _start_server(
            role="embedding",
            server_path=server_path,
            model=settings.llamacpp_embedding_model,
            endpoint=settings.llamacpp_embedding_base_url,
            port=EMBED_PORT,
            embedding=True,
        )

        with _state_lock:
            _state.llama_server_path = server_path
            _state.chat = chat_state
            _state.embedding = embed_state
            _state.status = "starting"
            _state.message = "Loading local chat and embedding models."

        chat_state = _wait_until_ready(chat_state)
        embed_state = _wait_until_ready(embed_state)
        overall_ready = chat_state.status == "ready" and embed_state.status == "ready"

        with _state_lock:
            _state.chat = chat_state
            _state.embedding = embed_state
            _state.status = "ready" if overall_ready else "starting"
            _state.message = (
                "Local Qwen chat and Nomic embeddings are ready."
                if overall_ready
                else "Local models are still loading or downloading."
            )
    except Exception as error:
        with _state_lock:
            _state.status = "error"
            _state.message = str(error)


def ensure_llamacpp_bootstrap(settings: UserSettings) -> dict[str, Any]:
    """Start local llama.cpp bootstrap in the background when needed."""
    global _bootstrap_thread
    if settings.llm_provider != "llamacpp" and settings.embedding_provider != "llamacpp":
        return llamacpp_bootstrap_status()

    with _state_lock:
        already_ready = _state.status == "ready"
        thread_alive = _bootstrap_thread is not None and _bootstrap_thread.is_alive()
    if already_ready:
        return llamacpp_bootstrap_status()
    if not thread_alive:
        _bootstrap_thread = threading.Thread(
            target=_bootstrap,
            args=(settings,),
            daemon=True,
            name="LlamaCppBootstrap",
        )
        _bootstrap_thread.start()
    return llamacpp_bootstrap_status()


def refresh_llamacpp_status() -> dict[str, Any]:
    """Refresh readiness flags for any already running endpoints."""
    with _state_lock:
        chat = _state.chat
        embedding = _state.embedding
        if chat and _server_ready(chat.endpoint, timeout=0.4):
            chat.status = "ready"
            chat.message = "Ready."
        if embedding and _server_ready(embedding.endpoint, timeout=0.4):
            embedding.status = "ready"
            embedding.message = "Ready."
        if chat and embedding and chat.status == "ready" and embedding.status == "ready":
            _state.status = "ready"
            _state.message = "Local Qwen chat and Nomic embeddings are ready."
    return llamacpp_bootstrap_status()


def llamacpp_bootstrap_status() -> dict[str, Any]:
    with _state_lock:
        return _state.to_dict()


def stop_managed_servers() -> None:
    for process in list(_managed_processes.values()):
        if process.poll() is None:
            process.terminate()


atexit.register(stop_managed_servers)
