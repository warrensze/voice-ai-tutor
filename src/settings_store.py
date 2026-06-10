"""Local user settings for provider and UI preferences."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any

SUPPORTED_SUBJECTS = ("history", "chemistry", "math", "english")
LLM_PROVIDERS = ("llamacpp", "ollama")
TTS_BACKENDS = ("piper", "kokoro", "pyttsx3")
KOKORO_DEVICES = ("auto", "cpu", "cuda")
STT_PROVIDERS = ("faster-whisper", "whispercpp")
STT_DEVICES = ("auto", "cpu", "cuda")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_SETTINGS_PATH = DATA_DIR / "user_settings.json"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _clean_choice(value: Any, allowed: tuple[str, ...], default: str) -> str:
    cleaned = str(value or "").strip().lower()
    return cleaned if cleaned in allowed else default


def _clean_stt_provider(value: Any, default: str = "faster-whisper") -> str:
    cleaned = str(value or "").strip().lower().replace("_", "-")
    if cleaned in {"fasterwhisper", "faster-whisper"}:
        return "faster-whisper"
    if cleaned in {"whisper.cpp", "whisper-cpp", "whispercpp"}:
        return "whispercpp"
    return default


def _clean_subject(value: Any, default: str = "english") -> str:
    return _clean_choice(value, SUPPORTED_SUBJECTS, default)


def _clean_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class UserSettings:
    """Serializable local settings used by the web UI and runtime."""

    llm_provider: str = "llamacpp"
    embedding_provider: str = "llamacpp"
    tts_backend: str = "piper"
    stt_provider: str = "faster-whisper"
    current_subject: str = "english"
    speak_responses: bool = True

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "llama3.1:8b"
    ollama_embedding_model: str = "mxbai-embed-large"

    llamacpp_chat_base_url: str = "http://127.0.0.1:8080/v1"
    llamacpp_chat_model: str = "Qwen/Qwen3-8B-GGUF:Q4_K_M"
    llamacpp_embedding_base_url: str = "http://127.0.0.1:8081/v1"
    llamacpp_embedding_model: str = "nomic-ai/nomic-embed-text-v1.5-GGUF:Q4_K_M"
    llamacpp_api_key: str = "local"

    kokoro_voice: str = "af_heart"
    kokoro_device: str = "auto"
    kokoro_allow_cpu: bool = True
    piper_voice: str = "en_US-lessac-medium"
    piper_data_dir: str = "models/piper"
    piper_use_cuda: bool = False
    piper_length_scale: float = 1.0
    piper_noise_scale: float = 0.667
    piper_noise_w_scale: float = 0.8
    piper_volume: float = 1.0
    pyttsx3_voice: str = ""

    stt_language: str = "en"
    faster_whisper_model: str = "base.en"
    faster_whisper_device: str = "auto"
    faster_whisper_compute_type: str = "auto"
    whispercpp_binary_path: str = "whisper-cli"
    whispercpp_model_path: str = "models/stt/whisper.cpp/ggml-base.en.bin"
    whispercpp_language: str = "en"

    subject_voices: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            "kokoro": {
                "history": "am_adam",
                "chemistry": "bf_alice",
                "math": "af_sky",
                "english": "af_heart",
            },
            "piper": {
                "history": "en_US-lessac-medium",
                "chemistry": "en_US-lessac-medium",
                "math": "en_US-lessac-medium",
                "english": "en_US-lessac-medium",
            },
            "pyttsx3": {
                "history": "",
                "chemistry": "",
                "math": "",
                "english": "",
            },
        }
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "UserSettings":
        defaults = cls()
        if not isinstance(payload, dict):
            payload = {}

        merged = asdict(defaults)
        for key, value in payload.items():
            if key in merged:
                merged[key] = value

        settings = cls(**merged)
        settings.llm_provider = _clean_choice(
            settings.llm_provider, LLM_PROVIDERS, defaults.llm_provider
        )
        settings.embedding_provider = _clean_choice(
            settings.embedding_provider, LLM_PROVIDERS, settings.llm_provider
        )
        settings.tts_backend = _clean_choice(
            settings.tts_backend, TTS_BACKENDS, defaults.tts_backend
        )
        settings.stt_provider = _clean_stt_provider(
            settings.stt_provider, defaults.stt_provider
        )
        settings.current_subject = _clean_subject(settings.current_subject)
        if settings.llamacpp_chat_model == "local-model":
            settings.llamacpp_chat_model = defaults.llamacpp_chat_model
        if settings.llamacpp_embedding_model == "local-embed":
            settings.llamacpp_embedding_model = defaults.llamacpp_embedding_model
        settings.speak_responses = bool(settings.speak_responses)
        settings.kokoro_device = _clean_choice(
            settings.kokoro_device, KOKORO_DEVICES, defaults.kokoro_device
        )
        settings.faster_whisper_device = _clean_choice(
            settings.faster_whisper_device,
            STT_DEVICES,
            defaults.faster_whisper_device,
        )
        settings.stt_language = str(settings.stt_language or "").strip() or "en"
        settings.faster_whisper_model = (
            str(settings.faster_whisper_model or "").strip()
            or defaults.faster_whisper_model
        )
        settings.faster_whisper_compute_type = (
            str(settings.faster_whisper_compute_type or "").strip()
            or defaults.faster_whisper_compute_type
        )
        settings.whispercpp_binary_path = (
            str(settings.whispercpp_binary_path or "").strip()
            or defaults.whispercpp_binary_path
        )
        settings.whispercpp_model_path = (
            str(settings.whispercpp_model_path or "").strip()
            or defaults.whispercpp_model_path
        )
        settings.whispercpp_language = (
            str(settings.whispercpp_language or "").strip()
            or settings.stt_language
            or "en"
        )
        settings.kokoro_allow_cpu = bool(settings.kokoro_allow_cpu)
        settings.piper_use_cuda = bool(settings.piper_use_cuda)
        settings.piper_length_scale = _clean_float(settings.piper_length_scale, 1.0)
        settings.piper_noise_scale = _clean_float(settings.piper_noise_scale, 0.667)
        settings.piper_noise_w_scale = _clean_float(settings.piper_noise_w_scale, 0.8)
        settings.piper_volume = max(0.0, _clean_float(settings.piper_volume, 1.0))

        if not isinstance(settings.subject_voices, dict):
            settings.subject_voices = defaults.subject_voices
        else:
            settings.subject_voices = _merge_subject_voices(
                defaults.subject_voices,
                settings.subject_voices,
            )

        return settings

    @classmethod
    def from_env(cls) -> "UserSettings":
        settings = cls()
        settings.llm_provider = _clean_choice(
            os.getenv("LLM_PROVIDER"), LLM_PROVIDERS, settings.llm_provider
        )
        settings.embedding_provider = _clean_choice(
            os.getenv("EMBEDDING_PROVIDER"),
            LLM_PROVIDERS,
            settings.llm_provider,
        )
        settings.tts_backend = _clean_choice(
            os.getenv("TTS_BACKEND"), TTS_BACKENDS, settings.tts_backend
        )
        settings.stt_provider = _clean_stt_provider(
            os.getenv("STT_PROVIDER"), settings.stt_provider
        )
        settings.ollama_chat_model = os.getenv(
            "OLLAMA_CHAT_MODEL", settings.ollama_chat_model
        )
        settings.ollama_embedding_model = os.getenv(
            "OLLAMA_EMBEDDING_MODEL", settings.ollama_embedding_model
        )
        settings.llamacpp_chat_base_url = os.getenv(
            "LLAMACPP_CHAT_BASE_URL", settings.llamacpp_chat_base_url
        )
        settings.llamacpp_chat_model = os.getenv(
            "LLAMACPP_CHAT_MODEL", settings.llamacpp_chat_model
        )
        settings.llamacpp_embedding_base_url = os.getenv(
            "LLAMACPP_EMBED_BASE_URL", settings.llamacpp_embedding_base_url
        )
        settings.llamacpp_embedding_model = os.getenv(
            "LLAMACPP_EMBED_MODEL", settings.llamacpp_embedding_model
        )
        settings.piper_voice = os.getenv("PIPER_VOICE", settings.piper_voice)
        settings.kokoro_voice = os.getenv("KOKORO_VOICE", settings.kokoro_voice)
        settings.kokoro_device = _clean_choice(
            os.getenv("KOKORO_DEVICE"), KOKORO_DEVICES, settings.kokoro_device
        )
        settings.kokoro_allow_cpu = _env_bool(
            "KOKORO_ALLOW_CPU", settings.kokoro_allow_cpu
        )
        settings.piper_data_dir = os.getenv("PIPER_DATA_DIR", settings.piper_data_dir)
        settings.piper_use_cuda = _env_bool("PIPER_USE_CUDA", settings.piper_use_cuda)
        settings.stt_language = os.getenv("STT_LANGUAGE", settings.stt_language)
        settings.faster_whisper_model = os.getenv(
            "FASTER_WHISPER_MODEL", settings.faster_whisper_model
        )
        settings.faster_whisper_device = _clean_choice(
            os.getenv("FASTER_WHISPER_DEVICE"),
            STT_DEVICES,
            settings.faster_whisper_device,
        )
        settings.faster_whisper_compute_type = os.getenv(
            "FASTER_WHISPER_COMPUTE_TYPE",
            settings.faster_whisper_compute_type,
        )
        settings.whispercpp_binary_path = os.getenv(
            "WHISPERCPP_BINARY_PATH", settings.whispercpp_binary_path
        )
        settings.whispercpp_model_path = os.getenv(
            "WHISPERCPP_MODEL_PATH", settings.whispercpp_model_path
        )
        settings.whispercpp_language = os.getenv(
            "WHISPERCPP_LANGUAGE", settings.whispercpp_language
        )
        return settings

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def provider_signature(self) -> str:
        return "|".join(
            [
                self.llm_provider,
                self.embedding_provider,
                self.tts_backend,
                self.stt_provider,
                self.ollama_chat_model,
                self.ollama_embedding_model,
                self.llamacpp_chat_base_url,
                self.llamacpp_chat_model,
                self.llamacpp_embedding_base_url,
                self.llamacpp_embedding_model,
                self.current_subject,
                self.piper_voice,
                self.kokoro_voice,
                self.kokoro_device,
                str(self.kokoro_allow_cpu),
                self.pyttsx3_voice,
                self.stt_language,
                self.faster_whisper_model,
                self.faster_whisper_device,
                self.faster_whisper_compute_type,
                self.whispercpp_binary_path,
                self.whispercpp_model_path,
                self.whispercpp_language,
                json.dumps(self.subject_voices, sort_keys=True),
            ]
        )

    def selected_voice(self, subject: str | None = None) -> str:
        backend = _clean_choice(self.tts_backend, TTS_BACKENDS, "kokoro")
        subject_name = _clean_subject(subject or self.current_subject)
        subject_map = self.subject_voices.get(backend, {})
        voice = str(subject_map.get(subject_name) or "").strip()
        if voice:
            return voice
        if backend == "piper":
            return self.piper_voice
        if backend == "pyttsx3":
            return self.pyttsx3_voice
        return self.kokoro_voice


def _merge_subject_voices(
    defaults: dict[str, dict[str, str]], loaded: dict[str, Any]
) -> dict[str, dict[str, str]]:
    merged = {backend: dict(voices) for backend, voices in defaults.items()}
    for backend, voice_map in loaded.items():
        if backend not in merged or not isinstance(voice_map, dict):
            continue
        for subject in SUPPORTED_SUBJECTS:
            value = voice_map.get(subject)
            if isinstance(value, str):
                merged[backend][subject] = value.strip()
    return merged


def load_user_settings(path: str | Path | None = None) -> UserSettings:
    """Load settings from disk, falling back to env-backed defaults."""
    settings_path = Path(path) if path else DEFAULT_SETTINGS_PATH
    env_settings = UserSettings.from_env()
    if not settings_path.exists():
        return env_settings
    try:
        with settings_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return env_settings
    merged = env_settings.to_dict()
    if isinstance(payload, dict):
        merged.update(payload)
    return UserSettings.from_dict(merged)


def save_user_settings(
    settings: UserSettings, path: str | Path | None = None
) -> UserSettings:
    """Persist settings atomically and return the normalized settings."""
    settings_path = Path(path) if path else DEFAULT_SETTINGS_PATH
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = UserSettings.from_dict(settings.to_dict())
    tmp_path = settings_path.with_suffix(settings_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(normalized.to_dict(), handle, indent=2, ensure_ascii=True)
    tmp_path.replace(settings_path)
    return normalized


def update_user_settings(
    updates: dict[str, Any], path: str | Path | None = None
) -> UserSettings:
    """Merge and persist partial settings from the UI."""
    current = load_user_settings(path)
    payload = current.to_dict()
    for key, value in updates.items():
        if key in payload:
            payload[key] = value
    settings = UserSettings.from_dict(payload)
    return save_user_settings(settings, path)


def settings_status_payload(settings: UserSettings) -> dict[str, Any]:
    """Return a UI-friendly settings payload without derived internals."""
    return {
        "settings": settings.to_dict(),
        "supported_subjects": list(SUPPORTED_SUBJECTS),
        "llm_providers": list(LLM_PROVIDERS),
        "tts_backends": list(TTS_BACKENDS),
        "stt_providers": list(STT_PROVIDERS),
    }
