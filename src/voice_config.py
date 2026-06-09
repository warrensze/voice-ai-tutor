import json
import os
from pathlib import Path

DEFAULT_SUBJECT_VOICE_MAP = {
    "history": "af_bella",
    "chemistry": "am_adam",
    "math": "af_heart",
    "english": "af_sarah",
}

VOICE_CONFIG_ENV_VAR = "VOICE_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "agent_voices.json"


def _read_voice_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Voice config must be a JSON object.")
    return loaded


def load_subject_voice_map(
    config_path: str | Path | None = None, *, backend: str = "kokoro"
) -> dict[str, str]:
    """Load per-subject voices from JSON, with safe defaults."""
    if config_path is None:
        env_value = os.getenv(VOICE_CONFIG_ENV_VAR)
        config_path = Path(env_value) if env_value else DEFAULT_CONFIG_PATH
    else:
        config_path = Path(config_path)

    voices = dict(DEFAULT_SUBJECT_VOICE_MAP)
    if not config_path.exists():
        return voices

    try:
        loaded = _read_voice_config(config_path)
    except Exception as error:
        print(f"[VoiceConfig] Failed to load {config_path}: {error}")
        return voices

    backend_config = loaded.get(backend)
    if isinstance(backend_config, dict):
        loaded_values = backend_config
    else:
        loaded_values = loaded

    for subject in voices:
        value = loaded_values.get(subject)
        if isinstance(value, str) and value.strip():
            voices[subject] = value.strip()

    return voices
