import re
import threading
import os
import importlib.util
import json
import platform
import shutil
import subprocess
import time
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pyttsx3

try:
    from kokoro import KPipeline  # type: ignore[import-not-found]

    KOKORO_IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - optional dependency
    KPipeline = None
    KOKORO_IMPORT_ERROR = error

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional dependency
    sd = None

try:
    from waveglow_tts import WaveGlowConfig, WaveGlowSynthesizer
except Exception:  # pragma: no cover - optional dependency
    WaveGlowConfig = None
    WaveGlowSynthesizer = None

try:
    from piper import PiperVoice, SynthesisConfig as PiperSynthesisConfig

    PIPER_IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - optional dependency
    PiperVoice = None
    PiperSynthesisConfig = None
    PIPER_IMPORT_ERROR = error

try:
    from settings_store import UserSettings
except Exception:  # pragma: no cover - settings are optional for legacy use
    UserSettings = None


KOKORO_VOICE_IDS = (
    "af_heart",
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "em_santa",
    "ff_siwis",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex",
    "pm_santa",
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
)


_TTS_REGISTRY_LOCK = threading.RLock()
_TTS_PLAYBACK_LOCK = threading.RLock()
_TTS_INSTANCES = weakref.WeakSet()
_ACTIVE_TTS_REF = None


class TTSBackendUnavailable(RuntimeError):
    """Raised when the selected TTS backend is not ready to speak."""


def stop_all_tts(*, except_instance=None, wait: bool = False):
    """Stop every TTS instance except the optional current owner."""
    global _ACTIVE_TTS_REF
    with _TTS_REGISTRY_LOCK:
        targets = [
            instance
            for instance in list(_TTS_INSTANCES)
            if instance is not except_instance
        ]
        _ACTIVE_TTS_REF = (
            weakref.ref(except_instance) if except_instance is not None else None
        )
    for instance in targets:
        try:
            instance.stop(wait=wait, release_owner=False)
        except Exception:
            pass


def _piper_data_dir_for(settings=None) -> Path:
    data_dir = Path(
        getattr(
            settings,
            "piper_data_dir",
            os.getenv("PIPER_DATA_DIR", "models/piper"),
        )
    )
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parents[1] / data_dir
    return data_dir


def _resolve_piper_voice_path(voice: str, settings=None) -> Path | None:
    """Find a Piper ONNX voice by explicit path or configured voice id."""
    if not voice:
        return None

    candidate = Path(voice).expanduser()
    if candidate.suffix.lower() == ".onnx" and candidate.exists():
        return candidate

    data_dir = _piper_data_dir_for(settings)
    candidates = [
        data_dir / f"{voice}.onnx",
        data_dir / voice / f"{voice}.onnx",
        data_dir / voice,
    ]
    for path in candidates:
        if path.suffix.lower() == ".onnx" and path.exists():
            return path
    return None


def _format_piper_voice_label(voice_id: str) -> str:
    parts = voice_id.split("-")
    if len(parts) < 3:
        return voice_id.replace("_", " ")

    locale = parts[0]
    name = " ".join(parts[1:-1]).replace("_", " ").title()
    quality = parts[-1].replace("_", " ").title()
    locale_labels = {
        "en_US": "US English",
        "en_GB": "UK English",
        "zh_CN": "Mandarin Chinese (Mainland)",
    }
    language = locale_labels.get(locale, locale.replace("_", "-"))
    return f"{language} · {name} · {quality}"


def _format_kokoro_voice_label(voice_id: str) -> str:
    if len(voice_id) < 3 or "_" not in voice_id:
        return voice_id.replace("_", " ")

    prefix, name = voice_id.split("_", 1)
    language_code = prefix[0]
    gender_code = prefix[1] if len(prefix) > 1 else ""
    language_labels = {
        "a": "American English",
        "b": "British English",
        "e": "Spanish",
        "f": "French",
        "h": "Hindi",
        "i": "Italian",
        "j": "Japanese",
        "p": "Brazilian Portuguese",
        "z": "Mandarin Chinese",
    }
    gender_labels = {
        "f": "Female",
        "m": "Male",
    }
    language = language_labels.get(language_code, language_code.upper())
    gender = gender_labels.get(gender_code, gender_code.upper())
    display_name = name.replace("_", " ").title()
    return f"{language} · {gender} · {display_name}"


def _piper_voice_runtime_error(voice_id: str, settings=None) -> str:
    """Return a clear local-runtime error for a Piper voice, or empty if usable."""
    voice_path = _resolve_piper_voice_path(voice_id, settings)
    if voice_path is None:
        return f"voice '{voice_id}' was not found in {_piper_data_dir_for(settings)}."

    config_path = voice_path.with_suffix(voice_path.suffix + ".json")
    if not config_path.exists():
        return f"voice '{voice_id}' is missing config file {config_path.name}."

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as error:
        return f"voice '{voice_id}' config could not be read: {error}"

    if (
        config.get("phoneme_type") == "pinyin"
        and importlib.util.find_spec("g2pw") is None
    ):
        return f"voice '{voice_id}' requires local Python package g2pw for Chinese pinyin phonemization."

    return ""


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _kokoro_runtime_options(settings=None) -> tuple[str, bool]:
    requested_device = str(
        getattr(
            settings,
            "kokoro_device",
            os.getenv("KOKORO_DEVICE", "auto"),
        )
        or "auto"
    ).strip().lower()
    if requested_device not in {"auto", "cpu", "cuda"}:
        requested_device = "auto"

    if hasattr(settings, "kokoro_allow_cpu"):
        allow_cpu = bool(getattr(settings, "kokoro_allow_cpu"))
    else:
        allow_cpu = _env_bool("KOKORO_ALLOW_CPU", True)
    return requested_device, allow_cpu


def _resolve_kokoro_device(torch_module, settings=None) -> tuple[str, str]:
    requested_device, allow_cpu = _kokoro_runtime_options(settings)
    cuda_available = bool(torch_module.cuda.is_available())

    if requested_device == "auto":
        resolved_device = "cuda" if cuda_available else "cpu"
    else:
        resolved_device = requested_device

    if requested_device == "cuda" and not cuda_available:
        return (
            "",
            "Kokoro is selected for CUDA, but CUDA is unavailable. Choose auto/cpu or enable CPU Kokoro.",
        )

    if resolved_device == "cpu" and not allow_cpu:
        return (
            "",
            "Kokoro is selected for CPU, but CPU Kokoro is not enabled.",
        )

    return resolved_device, ""


def tts_backend_status(settings=None) -> dict[str, Any]:
    """Return strict health for the selected local TTS backend."""
    backend = str(getattr(settings, "tts_backend", "pyttsx3") or "pyttsx3").strip().lower()
    voice = ""
    if settings is not None and hasattr(settings, "selected_voice"):
        voice = str(settings.selected_voice(getattr(settings, "current_subject", None)) or "")
    else:
        voice = str(getattr(settings, f"{backend}_voice", "") or "")

    result: dict[str, Any] = {
        "ok": False,
        "backend": backend,
        "voice": voice,
        "device": "",
        "strict": True,
        "error": "",
    }

    if backend == "piper":
        if sd is None:
            result["error"] = "Piper is selected, but sounddevice is not available."
        elif PiperVoice is None:
            result["error"] = f"Piper is selected, but the Piper package failed to import: {PIPER_IMPORT_ERROR!r}"
        elif _resolve_piper_voice_path(voice, settings) is None:
            result["error"] = f"Piper is selected, but voice '{voice}' was not found in {_piper_data_dir_for(settings)}."
        elif runtime_error := _piper_voice_runtime_error(voice, settings):
            result["error"] = f"Piper is selected, but {runtime_error}"
        else:
            result["ok"] = True
        return result

    if backend == "kokoro":
        if sd is None:
            result["error"] = "Kokoro is selected, but sounddevice is not available."
            return result
        if KPipeline is None:
            result["error"] = f"Kokoro is selected, but the Kokoro package failed to import: {KOKORO_IMPORT_ERROR!r}"
            return result
        try:
            import torch
        except Exception as error:
            result["error"] = f"Kokoro is selected, but PyTorch failed to import: {error}"
            return result

        resolved_device, device_error = _resolve_kokoro_device(torch, settings)
        if device_error:
            result["error"] = device_error
            return result
        result["device"] = resolved_device
        result["ok"] = True
        return result

    if backend == "pyttsx3":
        use_macos_say = os.getenv("TTS_USE_MACOS_SAY", "1").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if platform.system() == "Darwin" and use_macos_say:
            if shutil.which("say"):
                result["ok"] = True
            else:
                result["error"] = "pyttsx3 is selected, but the macOS say command was not found."
            return result
        engine = None
        try:
            engine = pyttsx3.init()
            result["ok"] = True
        except Exception as error:
            result["error"] = f"pyttsx3 is selected, but initialization failed: {error}"
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
        return result

    result["error"] = f"Unsupported TTS backend selected: {backend}"
    return result


def list_piper_voices(settings=None) -> list[dict[str, Any]]:
    """Return locally installed Piper voices from the configured model folder."""
    data_dir = _piper_data_dir_for(settings)
    if not data_dir.exists():
        return []

    voices: dict[str, dict[str, Any]] = {}
    for path in sorted(data_dir.rglob("*.onnx")):
        voice_id = path.stem
        if not voice_id or voice_id in voices:
            continue
        runtime_error = _piper_voice_runtime_error(voice_id, settings)
        voices[voice_id] = {
            "id": voice_id,
            "label": _format_piper_voice_label(voice_id),
            "path": str(path.relative_to(data_dir)),
            "available": not runtime_error,
        }
        if runtime_error:
            voices[voice_id]["error"] = runtime_error
    return list(voices.values())


def list_kokoro_voices(settings=None) -> list[dict[str, Any]]:
    """Return selectable Kokoro voice ids."""
    configured = str(getattr(settings, "kokoro_voice", "") or "").strip()
    configured_and_known = (
        [configured, *KOKORO_VOICE_IDS] if configured else KOKORO_VOICE_IDS
    )
    voice_ids = list(dict.fromkeys(configured_and_known))
    return [
        {
            "id": voice_id,
            "label": _format_kokoro_voice_label(voice_id),
            "available": True,
        }
        for voice_id in voice_ids
        if voice_id
    ]


def list_pyttsx3_voices(settings=None) -> list[dict[str, Any]]:
    """Return installed system voices exposed through pyttsx3."""
    options: list[dict[str, Any]] = []
    engine = None
    try:
        engine = pyttsx3.init()
        for voice in engine.getProperty("voices") or []:
            voice_id = str(getattr(voice, "id", "") or "").strip()
            if not voice_id:
                continue
            name = str(getattr(voice, "name", "") or voice_id).strip()
            options.append(
                {
                    "id": voice_id,
                    "label": name,
                    "available": True,
                }
            )
    except Exception as error:
        configured = str(getattr(settings, "pyttsx3_voice", "") or "").strip()
        if configured:
            options.append(
                {
                    "id": configured,
                    "label": configured,
                    "available": False,
                    "error": str(error),
                }
            )
    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
    return options


def list_tts_voices(settings=None) -> dict[str, list[dict[str, Any]]]:
    """Return voice choices grouped by local TTS backend."""
    voices = {
        "kokoro": list_kokoro_voices(settings),
        "piper": list_piper_voices(settings),
        "pyttsx3": list_pyttsx3_voices(settings),
    }
    voices["pyttsx3"].insert(
        0,
        {
            "id": "",
            "label": "System default",
            "available": True,
        },
    )

    current_piper = str(getattr(settings, "piper_voice", "") or "").strip()
    if current_piper and all(voice["id"] != current_piper for voice in voices["piper"]):
        voices["piper"].insert(
            0,
            {
                "id": current_piper,
                "label": f"{current_piper} (missing)",
                "available": False,
            },
        )
    return voices


class TextToSpeech:
    def __init__(
        self,
        voice: str = "af_heart",
        speed: float = 1.0,
        *,
        backend: str | None = None,
        backend_order: str | list[str] | None = None,
        settings: "UserSettings | None" = None,
    ):
        """Initialize the selected local TTS backend without provider fallback."""
        self.rate = 175
        configured_volume = float(os.getenv("TTS_OUTPUT_VOLUME", "1.0"))
        self.volume = min(1.0, max(0.0, configured_volume))
        if settings is not None:
            backend = backend or getattr(settings, "tts_backend", None)
            voice = settings.selected_voice(getattr(settings, "current_subject", None))
        self.voice = voice
        self.speed = speed
        self.settings = settings
        self.output_gain = max(0.1, float(os.getenv("TTS_OUTPUT_GAIN", "1.0")))
        self._lock = threading.Lock()
        self._kokoro_lock = threading.Lock()
        self._piper_lock = threading.Lock()
        self._speak_async_lock = threading.Lock()
        self._active_engine = None
        self._active_process = None
        self._stop_requested = threading.Event()
        self._speak_thread = None
        self._async_queue: list[str] = []
        self._async_queue_lock = threading.Lock()
        self._async_queue_condition = threading.Condition(self._async_queue_lock)
        self._warmup_thread = None
        self._kokoro_pipeline = None
        self._kokoro_assets_ready = False
        self._blend_warning_shown = False
        self._piper_voice = None
        self._piper_voice_path = None
        self._waveglow_synth = None
        self._waveglow_failed = False  # Track if WaveGlow had a failure
        self._is_playing = threading.Event()  # Track when audio is being played
        self._last_playback_end = 0.0
        self._max_async_queue = max(1, int(os.getenv("TTS_ASYNC_QUEUE_MAX", "8")))
        self._max_async_chars = max(40, int(os.getenv("TTS_ASYNC_MAX_CHARS", "320")))
        self._async_thread_daemon = os.getenv(
            "TTS_ASYNC_DAEMON", "0"
        ).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self._kokoro_repo_id = os.getenv("KOKORO_REPO_ID", "hexgrad/Kokoro-82M")
        self._kokoro_enable_voice_blend = os.getenv(
            "KOKORO_ENABLE_VOICE_BLEND", "0"
        ).strip().lower() in {"1", "true", "yes"}
        self._kokoro_offline_after_preload = os.getenv(
            "KOKORO_OFFLINE_AFTER_PRELOAD", "1"
        ).strip().lower() in {"1", "true", "yes"}
        self._piper_data_dir = Path(
            getattr(settings, "piper_data_dir", os.getenv("PIPER_DATA_DIR", "models/piper"))
        )
        self._piper_use_cuda = bool(
            getattr(
                settings,
                "piper_use_cuda",
                os.getenv("PIPER_USE_CUDA", "0").strip().lower()
                in {"1", "true", "yes"},
            )
        )
        self._piper_length_scale = float(
            getattr(settings, "piper_length_scale", os.getenv("PIPER_LENGTH_SCALE", "1.0"))
        )
        self._piper_noise_scale = float(
            getattr(settings, "piper_noise_scale", os.getenv("PIPER_NOISE_SCALE", "0.667"))
        )
        self._piper_noise_w_scale = float(
            getattr(settings, "piper_noise_w_scale", os.getenv("PIPER_NOISE_W_SCALE", "0.8"))
        )
        self._piper_volume = float(
            getattr(settings, "piper_volume", os.getenv("PIPER_VOLUME", "1.0"))
        )
        self._use_macos_say = os.getenv(
            "TTS_USE_MACOS_SAY", "1"
        ).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self._kokoro_marker_path = (
            Path(__file__).resolve().parents[1]
            / f".kokoro_assets_{self._kokoro_repo_id.replace('/', '_')}.ready"
        )
        if self._kokoro_offline_after_preload and self._kokoro_marker_path.exists():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.output_device_override = self._resolve_output_override()
        selected_backend = str(backend or os.getenv("TTS_BACKEND", "")).strip().lower()
        if not selected_backend:
            if isinstance(backend_order, str):
                selected_backend = next(
                    (
                        name.strip().lower()
                        for name in backend_order.split(",")
                        if name.strip()
                    ),
                    "pyttsx3",
                )
            elif backend_order:
                selected_backend = next(
                    (
                        str(name).strip().lower()
                        for name in backend_order
                        if str(name).strip()
                    ),
                    "pyttsx3",
                )
            else:
                selected_backend = "pyttsx3"
        self.backend_order = [selected_backend]
        self.backend = selected_backend
        self.backend_available = False
        self.backend_error = ""
        print(f"[Mouth] Initializing selected TTS backend: {self.backend}")
        self._configure_backend()
        if self.backend_available:
            print(f"[Mouth] Using {self.backend.upper()} backend for speech output")
        else:
            print(f"[Mouth] {self.backend.upper()} unavailable: {self.backend_error}")
        with _TTS_REGISTRY_LOCK:
            _TTS_INSTANCES.add(self)
        self._start_background_warmup()

    def _claim_audio_owner(self):
        """Ensure this instance is the only TTS instance allowed to speak."""
        global _ACTIVE_TTS_REF
        with _TTS_REGISTRY_LOCK:
            active = _ACTIVE_TTS_REF() if _ACTIVE_TTS_REF is not None else None
            if active is self:
                return
        stop_all_tts(except_instance=self, wait=False)
        with _TTS_REGISTRY_LOCK:
            _ACTIVE_TTS_REF = weakref.ref(self)

    def _release_audio_owner(self):
        global _ACTIVE_TTS_REF
        with _TTS_REGISTRY_LOCK:
            active = _ACTIVE_TTS_REF() if _ACTIVE_TTS_REF is not None else None
            if active is self:
                _ACTIVE_TTS_REF = None

    def _start_background_warmup(self):
        """Preload heavy TTS backend components to reduce first-response silence."""
        if self.backend != "waveglow" or self._waveglow_synth is None:
            return

        def warmup_runner():
            print("[Mouth] WaveGlow warmup starting...")
            try:
                self._waveglow_synth.load()
                print("[Mouth] WaveGlow warmup completed successfully")
            except Exception as e:
                print(f"[Mouth] WaveGlow warmup failed: {e}")
                # Mark as failed but DO NOT switch backends - let playback handle it
                self._waveglow_failed = True

        self._warmup_thread = threading.Thread(target=warmup_runner, daemon=True)
        self._warmup_thread.start()

    def _ensure_waveglow_loaded(self):
        """Block briefly until warm-up completes, then load on demand if needed."""
        if self._waveglow_synth is None:
            return

        thread = self._warmup_thread
        if thread and thread.is_alive():
            thread.join(timeout=0.8)

        if not self._waveglow_synth.is_ready:
            self._waveglow_synth.load()

    def _resolve_output_override(self):
        """Resolve optional fixed output override from environment variable."""
        if sd is None:
            return None

        override = os.getenv("TTS_OUTPUT_DEVICE", "").strip()
        if override:
            return int(override) if override.isdigit() else override
        return None

    def _current_output_device(self):
        """Return active playback output device, following system defaults by default."""
        if sd is None:
            return None

        if self.output_device_override is not None:
            return self.output_device_override

        # None means: always use current OS default output device at playback time.
        return None

    def _play_audio(self, audio: Any, sample_rate: int):
        """Play audio on the configured output device."""
        if sd is None:
            return

        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.size == 0:
            return
        waveform = np.clip(waveform * self.output_gain, -1.0, 1.0)

        device = self._current_output_device()
        with _TTS_PLAYBACK_LOCK:
            try:
                if self._stop_requested.is_set():
                    return
                self._is_playing.set()
                sd.stop()
                sd.play(waveform, sample_rate, device=device)
                sd.wait()
            finally:
                self._is_playing.clear()
                self._last_playback_end = time.monotonic()

    def _play_audio_exclusive(self, render_and_play):
        """Serialize non-sounddevice engines with the same process-wide speaker lock."""
        with _TTS_PLAYBACK_LOCK:
            if self._stop_requested.is_set():
                return
            try:
                if sd is not None:
                    sd.stop()
            except Exception:
                pass
            render_and_play()

    def _backend_unavailable(self, message: str) -> bool:
        self.backend_available = False
        self.backend_error = message
        print(f"[Mouth] {message}")
        return False

    def _backend_ready(self) -> bool:
        self.backend_available = True
        self.backend_error = ""
        return True

    def _configure_backend(self):
        """Initialize only the selected backend."""
        initializers = {
            "waveglow": self._init_waveglow_backend,
            "kokoro": self._init_kokoro_backend,
            "piper": self._init_piper_backend,
            "pyttsx3": self._init_pyttsx3_backend,
        }
        initializer = initializers.get(self.backend)
        if initializer is None:
            self._backend_unavailable(f"Unsupported TTS backend selected: {self.backend}")
            return

        if initializer():
            self._backend_ready()
            print(f"[Mouth] {self.backend} backend initialized successfully")
            return

        if not self.backend_error:
            self._backend_unavailable(
                f"{self.backend} backend failed to initialize."
            )

    def _init_kokoro_backend(self) -> bool:
        """Initialize Kokoro backend if dependencies are available.

        Kokoro-82M supports voice blending with the syntax:
        - Single voice: 'af_heart'
        - Blended voices: 'af_heart+af_bella' (50/50 blend)
        - Weighted blend: 'af_heart+af_bella@0.7' (70% af_heart, 30% af_bella)
        """
        if sd is None:
            return self._backend_unavailable(
                "Kokoro is selected, but sounddevice is not available."
            )

        if KPipeline is None:
            return self._backend_unavailable(
                "Kokoro is selected, but the Kokoro package failed to import. "
                f"Original error: {repr(KOKORO_IMPORT_ERROR)}"
            )

        try:
            import torch
        except Exception as error:
            return self._backend_unavailable(
                f"Kokoro is selected, but PyTorch failed to import: {error}"
            )

        requested_device, _ = _kokoro_runtime_options(self.settings)
        resolved_device, device_error = _resolve_kokoro_device(torch, self.settings)
        if device_error:
            return self._backend_unavailable(device_error)
        if requested_device == "auto" and resolved_device == "cpu":
            print("[Mouth] CUDA unavailable; using CPU for Kokoro because Kokoro device is auto")

        try:
            self._kokoro_pipeline = KPipeline(
                lang_code="a",
                repo_id=self._kokoro_repo_id,
                device=resolved_device,
            )
            if resolved_device.startswith("cuda") and torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[Mouth] Kokoro-82M initialized on CUDA GPU: {gpu_name}")
            else:
                print(f"[Mouth] Kokoro-82M initialized on {resolved_device}")

            self._preload_kokoro_assets()
            return True
        except Exception as e:
            self._kokoro_pipeline = None
            return self._backend_unavailable(f"Failed to initialize Kokoro-82M: {e}")

    def _init_waveglow_backend(self) -> bool:
        """Prepare WaveGlow backend with lazy model loading."""
        if WaveGlowSynthesizer is None or WaveGlowConfig is None or sd is None:
            return self._backend_unavailable(
                "WaveGlow is selected, but WaveGlowSynthesizer, WaveGlowConfig, or sounddevice is not available."
            )

        try:
            import torch
        except Exception:
            return self._backend_unavailable(
                "WaveGlow is selected, but PyTorch is not available."
            )

        requested_device = os.getenv("WAVEGLOW_DEVICE", "cuda").strip().lower()
        if not requested_device:
            requested_device = "cuda"

        allow_cpu = os.getenv("WAVEGLOW_ALLOW_CPU", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }

        if requested_device in {"auto", ""}:
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"

        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            if not allow_cpu:
                return self._backend_unavailable(
                    "WaveGlow is selected, but CUDA is unavailable. "
                    "Set WAVEGLOW_ALLOW_CPU=1 if you intentionally want CPU WaveGlow."
                )
            requested_device = "cpu"
            print("[Mouth] CUDA unavailable; using CPU for WaveGlow because WAVEGLOW_ALLOW_CPU=1")

        if requested_device == "cpu" and not allow_cpu:
            return self._backend_unavailable(
                "WaveGlow is selected for CPU, but WAVEGLOW_ALLOW_CPU is not enabled."
            )

        try:
            sigma = float(os.getenv("WAVEGLOW_SIGMA", "0.8"))
        except ValueError:
            sigma = 0.8

        try:
            config = WaveGlowConfig(
                device=requested_device,
                sample_rate=int(os.getenv("WAVEGLOW_SAMPLE_RATE", "22050")),
                sigma=sigma,
            )
            self._waveglow_synth = WaveGlowSynthesizer(config=config)
            print(f"[Mouth] WaveGlow initialized on {requested_device}")
            self._waveglow_failed = False
            return True
        except Exception as e:
            self._waveglow_synth = None
            return self._backend_unavailable(
                f"Failed to initialize WaveGlowSynthesizer: {e}"
            )

    def _init_piper_backend(self) -> bool:
        """Initialize Piper with a local ONNX voice file."""
        if sd is None:
            return self._backend_unavailable(
                "Piper is selected, but sounddevice is not available."
            )
        if PiperVoice is None:
            return self._backend_unavailable(
                "Piper is selected, but the Piper package failed to import. "
                f"Original error: {repr(PIPER_IMPORT_ERROR)}"
            )

        voice_path = self._resolve_piper_voice_path(self.voice)
        if voice_path is None:
            return self._backend_unavailable(
                f"Piper is selected, but voice '{self.voice}' was not found in "
                f"{_piper_data_dir_for(self.settings)}."
            )
        if runtime_error := _piper_voice_runtime_error(self.voice, self.settings):
            return self._backend_unavailable(f"Piper is selected, but {runtime_error}")

        try:
            self._piper_voice = PiperVoice.load(
                str(voice_path),
                use_cuda=self._piper_use_cuda,
            )
            self._piper_voice_path = voice_path
            return True
        except Exception as error:
            self._piper_voice = None
            self._piper_voice_path = None
            return self._backend_unavailable(
                f"Failed to load Piper voice {voice_path}: {error}"
            )

    def _init_pyttsx3_backend(self) -> bool:
        """Accept pyttsx3 as selected; actual engine errors surface at playback."""
        return True

    def _resolve_piper_voice_path(self, voice: str) -> Path | None:
        """Find a Piper ONNX voice by explicit path or voice id."""
        return _resolve_piper_voice_path(voice, self.settings)

    def _piper_synthesis_config(self):
        """Build Piper synthesis config when the installed version supports it."""
        if PiperSynthesisConfig is None:
            return None
        try:
            return PiperSynthesisConfig(
                volume=self._piper_volume,
                length_scale=self._piper_length_scale,
                noise_scale=self._piper_noise_scale,
                noise_w_scale=self._piper_noise_w_scale,
            )
        except Exception:
            return None

    def is_audio_playing(self) -> bool:
        """Return True if audio is currently being played through speakers."""
        return self._is_playing.is_set()

    def recently_played(self, window_seconds: float = 0.8) -> bool:
        """Return True if audio playback ended recently."""
        if self._is_playing.is_set():
            return True
        if self._last_playback_end <= 0:
            return False
        return (time.monotonic() - self._last_playback_end) <= max(0.0, window_seconds)

    def set_voice(self, new_voice: str):
        """Update the active voice preset used by subsequent speech calls."""
        if not new_voice:
            return
        if new_voice == self.voice:
            return
        self.voice = new_voice
        if self.backend == "piper":
            self.stop(wait=False, release_owner=False)
            self._piper_voice = None
            self._piper_voice_path = None
            self.backend_available = False
            self.backend_error = ""
            self._configure_backend()

    def _parse_blended_voice(self, voice: str) -> tuple[str, str, float] | None:
        """Parse blended voice syntax: voice_a+voice_b@ratio."""
        pattern = re.compile(
            r"^\s*([a-z0-9_]+)\+([a-z0-9_]+)(?:@([0-9]*\.?[0-9]+))?\s*$",
            re.IGNORECASE,
        )
        match = pattern.match(voice or "")
        if not match:
            return None

        voice_a, voice_b, ratio_text = match.groups()
        ratio = float(ratio_text) if ratio_text else 0.5
        ratio = max(0.0, min(1.0, ratio))
        return voice_a, voice_b, ratio

    def _render_kokoro_audio(self, text: str, voice: str) -> np.ndarray:
        """Render a full audio waveform for text using a single Kokoro voice."""
        if self._kokoro_pipeline is None:
            return np.array([], dtype=np.float32)

        rendered: list[np.ndarray] = []
        with self._kokoro_lock:
            generator = self._kokoro_pipeline(
                text,
                voice=voice,
                speed=self.speed,
                split_pattern=r"\n+",
            )
            for _, _, audio in generator:
                if self._stop_requested.is_set():
                    return np.array([], dtype=np.float32)
                audio_np = np.asarray(audio, dtype=np.float32).flatten()
                if audio_np.size:
                    rendered.append(audio_np)

        if not rendered:
            return np.array([], dtype=np.float32)
        return np.concatenate(rendered)

    def _mix_kokoro_voices(
        self, text: str, voice_a: str, voice_b: str, ratio_a: float
    ) -> np.ndarray:
        """Generate and blend two Kokoro voices into one waveform."""
        wave_a = self._render_kokoro_audio(text, voice_a)
        wave_b = self._render_kokoro_audio(text, voice_b)

        if wave_a.size == 0 and wave_b.size == 0:
            return np.array([], dtype=np.float32)
        if wave_a.size == 0:
            return wave_b
        if wave_b.size == 0:
            return wave_a

        target_len = max(wave_a.shape[0], wave_b.shape[0])
        if wave_a.shape[0] < target_len:
            wave_a = np.pad(wave_a, (0, target_len - wave_a.shape[0]))
        if wave_b.shape[0] < target_len:
            wave_b = np.pad(wave_b, (0, target_len - wave_b.shape[0]))

        mixed = (ratio_a * wave_a) + ((1.0 - ratio_a) * wave_b)
        peak = float(np.max(np.abs(mixed)))
        if peak > 1.0:
            mixed = mixed / peak
        return mixed.astype(np.float32)

    def _preload_kokoro_assets(self):
        """Download model and all voice files to local HF cache before speaking."""

        def _set_hf_offline_mode():
            if self._kokoro_offline_after_preload:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        if self._kokoro_marker_path.exists():
            self._kokoro_assets_ready = True
            print("[Mouth] Kokoro assets already preloaded (marker found)")
            _set_hf_offline_mode()
            return

        preload_enabled = os.getenv("KOKORO_PRELOAD_ON_STARTUP", "1").strip().lower()
        if preload_enabled not in {"1", "true", "yes"}:
            return
        if self._kokoro_assets_ready:
            return

        try:
            from huggingface_hub import hf_hub_download

            repo_id = self._kokoro_repo_id
            required_files = [
                "config.json",
                "kokoro-v1_0.pth",
                *[f"voices/{voice_id}.pt" for voice_id in KOKORO_VOICE_IDS],
            ]

            # Fast local-only check: if all assets are already cached, avoid network.
            all_cached_locally = True
            for required_file in required_files:
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=required_file,
                        local_files_only=True,
                    )
                except Exception:
                    all_cached_locally = False
                    break

            if all_cached_locally:
                self._kokoro_marker_path.write_text("ready\n", encoding="utf-8")
                self._kokoro_assets_ready = True
                print("[Mouth] Kokoro assets found in local cache")
                _set_hf_offline_mode()
                return

            print(
                f"[Mouth] Preloading Kokoro assets from {repo_id} ({len(KOKORO_VOICE_IDS)} voices)..."
            )

            for required_file in required_files:
                hf_hub_download(repo_id=repo_id, filename=required_file)

            self._kokoro_marker_path.write_text("ready\n", encoding="utf-8")
            self._kokoro_assets_ready = True
            print("[Mouth] Kokoro assets preloaded successfully")
            _set_hf_offline_mode()
        except Exception as error:
            print(f"[Mouth] Warning: Kokoro asset preload failed: {error}")

    def _create_engine(self):
        """Create and configure a pyttsx3 engine instance."""
        engine = pyttsx3.init()
        # Tune speech to a natural conversational pace.
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)

        # Map Kokoro-like presets to common Windows SAPI voice names.
        voice_aliases = {
            "af_heart": ["zira", "aria", "female"],
            "af_bella": ["zira", "female"],
            "af_sarah": ["zira", "aria", "female"],
            "af_nicole": ["zira", "female"],
            "am_adam": ["david", "mark", "male"],
            "am_michael": ["david", "mark", "male"],
        }
        requested = (self.voice or "").lower()
        preferred_tokens = voice_aliases.get(requested, [requested])
        for voice in engine.getProperty("voices"):
            voice_text = f"{voice.id} {voice.name}".lower()
            if any(token and token in voice_text for token in preferred_tokens):
                engine.setProperty("voice", voice.id)
                break

        return engine

    def _iter_chunks(self, text: str):
        """Split long text into sentence-like chunks for smoother playback."""
        for part in re.split(r"(?<=[.!?])\s+|\n+", text):
            chunk = part.strip()
            if chunk:
                yield chunk

    def _split_for_async_queue(self, text: str) -> list[str]:
        """Split queued text into bounded segments without dropping content."""
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []

        max_chars = self._max_async_chars
        if len(normalized) <= max_chars:
            return [normalized]

        segments: list[str] = []
        sentence_parts = re.split(r"(?<=[.!?])\s+", normalized)
        if not sentence_parts:
            sentence_parts = [normalized]

        current = ""

        def flush_current() -> None:
            nonlocal current
            if current:
                segments.append(current)
                current = ""

        def add_word(word: str) -> None:
            nonlocal current
            if not word:
                return

            if not current:
                current = word
                return

            candidate = f"{current} {word}"
            if len(candidate) <= max_chars:
                current = candidate
                return

            flush_current()
            current = word

        for part in sentence_parts:
            sentence = part.strip()
            if not sentence:
                continue

            words = sentence.split()
            if not words:
                continue

            for word in words:
                if len(word) > max_chars:
                    flush_current()
                    start = 0
                    while start < len(word):
                        token = word[start : start + max_chars]
                        if len(token) == max_chars:
                            segments.append(token)
                        else:
                            current = token
                        start += max_chars
                    continue

                add_word(word)

            flush_current()

        flush_current()
        return [segment for segment in segments if segment]

    def _speak_with_engine(self, text: str):
        """Speak a single chunk using the pyttsx3 backend."""
        if self._stop_requested.is_set():
            return
        if self._speak_with_macos_say_chunks([text]):
            return

        def render_and_play():
            engine = self._create_engine()
            with self._lock:
                self._active_engine = engine
            try:
                self._is_playing.set()
                engine.say(text)
                engine.runAndWait()
            finally:
                self._is_playing.clear()
                self._last_playback_end = time.monotonic()
                try:
                    engine.stop()
                except Exception:
                    pass
                with self._lock:
                    if self._active_engine is engine:
                        self._active_engine = None

        self._play_audio_exclusive(render_and_play)

    def is_available(self) -> bool:
        """Return whether the selected backend initialized successfully."""
        return bool(self.backend_available)

    def status_payload(self) -> dict[str, Any]:
        """Return the active runtime TTS status."""
        return {
            "ok": bool(self.backend_available),
            "backend": self.backend,
            "voice": self.voice,
            "strict": True,
            "error": self.backend_error,
        }

    def _require_backend_ready(self):
        if self.backend_available:
            return
        message = self.backend_error or f"{self.backend} is not available."
        raise TTSBackendUnavailable(message)

    def _mark_runtime_error(self, message: str):
        self.backend_available = False
        self.backend_error = message
        print(f"[Mouth] {message}")

    def _speak_chunks(self, text: str):
        """Play chunks through the active backend until completed or stopped."""
        self._require_backend_ready()
        if self.backend == "waveglow":
            self._speak_with_waveglow(text)
            return

        if self.backend == "kokoro":
            self._speak_with_kokoro(text)
            return

        if self.backend == "piper":
            self._speak_with_piper(text)
            return

        if self.backend == "pyttsx3":
            self._speak_with_engine_chunks(text)
            return

        self._mark_runtime_error(f"Unsupported TTS backend selected: {self.backend}")
        raise TTSBackendUnavailable(self.backend_error)

    def _speak_with_waveglow(self, text: str):
        """Speak text with WaveGlow and stream generated audio through sounddevice."""
        if self._waveglow_synth is None or sd is None:
            self._waveglow_failed = True
            self._mark_runtime_error("WaveGlow is selected, but it is not initialized.")
            raise TTSBackendUnavailable(self.backend_error)

        for chunk in self._iter_chunks(text):
            if self._stop_requested.is_set():
                return

            try:
                self._ensure_waveglow_loaded()
                waveform = self._waveglow_synth.synthesize(chunk)
                if waveform.size == 0:
                    continue
                self._play_audio(waveform, self._waveglow_synth.sample_rate)
            except Exception as e:
                self._waveglow_failed = True
                self._mark_runtime_error(
                    f"WaveGlow playback failed for the selected backend: {e}"
                )
                raise TTSBackendUnavailable(self.backend_error) from e

    def _speak_with_engine_chunks(self, text: str):
        """Speak all chunks with one pyttsx3 engine run."""
        chunks = [chunk for chunk in self._iter_chunks(text)]
        if not chunks or self._stop_requested.is_set():
            return
        if self._speak_with_macos_say_chunks(chunks):
            return

        def render_and_play():
            engine = self._create_engine()
            with self._lock:
                self._active_engine = engine
            try:
                self._is_playing.set()
                for chunk in chunks:
                    if self._stop_requested.is_set():
                        break
                    engine.say(chunk)
                if not self._stop_requested.is_set():
                    engine.runAndWait()
            finally:
                self._is_playing.clear()
                self._last_playback_end = time.monotonic()
                try:
                    engine.stop()
                except Exception:
                    pass
                with self._lock:
                    if self._active_engine is engine:
                        self._active_engine = None

        try:
            self._play_audio_exclusive(render_and_play)
        except Exception as e:
            if self._stop_requested.is_set():
                return
            print(f"[Mouth] pyttsx3 error: {e}, retrying")
            try:
                self._play_audio_exclusive(render_and_play)
            except Exception as retry_error:
                print(f"[Mouth] pyttsx3 retry failed: {retry_error}")

    def _speak_with_macos_say_chunks(self, chunks: list[str]) -> bool:
        """Use the blocking macOS say command to avoid pyttsx3 overlap."""
        say_path = shutil.which("say")
        if (
            not self._use_macos_say
            or platform.system() != "Darwin"
            or say_path is None
        ):
            return False

        voice_name = self._resolve_macos_say_voice()

        def render_and_play():
            self._is_playing.set()
            try:
                for chunk in chunks:
                    if self._stop_requested.is_set():
                        break
                    command = [say_path]
                    if voice_name:
                        command.extend(["-v", voice_name])
                    command.append(chunk)
                    process = subprocess.Popen(command)
                    with self._lock:
                        self._active_process = process
                    try:
                        while process.poll() is None:
                            if self._stop_requested.is_set():
                                process.terminate()
                                try:
                                    process.wait(timeout=0.4)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                                    process.wait(timeout=0.4)
                                break
                            time.sleep(0.02)
                    finally:
                        with self._lock:
                            if self._active_process is process:
                                self._active_process = None
            finally:
                self._is_playing.clear()
                self._last_playback_end = time.monotonic()

        try:
            self._play_audio_exclusive(render_and_play)
            return True
        except Exception as error:
            if self._stop_requested.is_set():
                return True
            print(f"[Mouth] macOS say playback error: {error}")
            return False

    def _resolve_macos_say_voice(self) -> str | None:
        """Translate an installed pyttsx3 voice id to a macOS say voice name."""
        requested = str(self.voice or "").strip()
        if not requested or "_" in requested:
            return None

        if not requested.startswith("com.apple."):
            return requested

        engine = None
        try:
            engine = pyttsx3.init()
            for voice in engine.getProperty("voices") or []:
                voice_id = str(getattr(voice, "id", "") or "")
                if voice_id == requested:
                    name = str(getattr(voice, "name", "") or "").strip()
                    return name or None
        except Exception:
            return None
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
        return None

    def _speak_with_piper(self, text: str):
        """Speak text with Piper and stream audio chunks through sounddevice."""
        if self._piper_voice is None:
            self._mark_runtime_error("Piper is selected, but no Piper voice is loaded.")
            raise TTSBackendUnavailable(self.backend_error)

        syn_config = self._piper_synthesis_config()
        try:
            with self._piper_lock:
                for text_chunk in self._iter_chunks(text):
                    if self._stop_requested.is_set():
                        return

                    kwargs = {"syn_config": syn_config} if syn_config is not None else {}
                    for audio_chunk in self._piper_voice.synthesize(
                        text_chunk, **kwargs
                    ):
                        if self._stop_requested.is_set():
                            return
                        audio = np.frombuffer(
                            audio_chunk.audio_int16_bytes, dtype=np.int16
                        ).astype(np.float32)
                        if audio.size == 0:
                            continue
                        audio = audio / 32768.0
                        channels = int(getattr(audio_chunk, "sample_channels", 1) or 1)
                        if channels > 1 and audio.size % channels == 0:
                            audio = audio.reshape((-1, channels))
                        sample_rate = int(getattr(audio_chunk, "sample_rate", 22050))
                        self._play_audio(audio, sample_rate)
        except Exception as error:
            self._mark_runtime_error(
                f"Piper playback failed for the selected backend: {error}"
            )
            raise TTSBackendUnavailable(self.backend_error) from error

    def _speak_with_kokoro(self, text: str):
        """Speak text with Kokoro-82M and stream generated audio through sounddevice.

        Supports voice blending for natural voice variations:
        - af_heart+af_bella (50/50 blend)
        - af_heart+af_bella@0.7 (70% af_heart, 30% af_bella)
        """
        if self._kokoro_pipeline is None:
            self._mark_runtime_error("Kokoro is selected, but it is not initialized.")
            raise TTSBackendUnavailable(self.backend_error)

        blend = self._parse_blended_voice(self.voice)
        if blend is not None and self._kokoro_enable_voice_blend:
            voice_a, voice_b, ratio_a = blend
            mixed = self._mix_kokoro_voices(text, voice_a, voice_b, ratio_a)
            if mixed.size:
                self._play_audio(mixed, 24000)
            return

        selected_voice = self.voice
        if blend is not None and not self._kokoro_enable_voice_blend:
            selected_voice = blend[0]
            if not self._blend_warning_shown:
                print(
                    f"[Mouth] Voice blending is disabled; using primary voice '{selected_voice}'."
                )
                self._blend_warning_shown = True

        try:
            with self._kokoro_lock:
                generator = self._kokoro_pipeline(
                    text,
                    voice=selected_voice,
                    speed=self.speed,
                    split_pattern=r"\n+",
                )
                for _, _, audio in generator:
                    if self._stop_requested.is_set():
                        return
                    self._play_audio(audio, 24000)
        except Exception as error:
            self._mark_runtime_error(
                f"Kokoro playback failed for the selected backend: {error}"
            )
            raise TTSBackendUnavailable(self.backend_error) from error

    def speak(self, text):
        """Speak text synchronously, replacing any in-flight playback."""
        if not text:
            return
        self._require_backend_ready()

        self._claim_audio_owner()
        self.stop(release_owner=False)
        self._stop_requested.clear()
        try:
            self._speak_chunks(text)
        finally:
            self._release_audio_owner()

    def _consume_async_queue(self):
        """Drain queued async speech chunks in order without dropping content."""
        try:
            while not self._stop_requested.is_set():
                with self._async_queue_condition:
                    if not self._async_queue:
                        break
                    next_text = self._async_queue.pop(0)
                    self._async_queue_condition.notify_all()

                try:
                    self._speak_chunks(next_text)
                except Exception as error:
                    if self._stop_requested.is_set():
                        break
                    print(f"[Mouth] Async speech error: {error}")
        finally:
            self._release_audio_owner()
            with self._async_queue_condition:
                self._speak_thread = None
                self._async_queue_condition.notify_all()

    def speak_async(self, text):
        """Speak text in a background thread so main flow can continue."""
        if not text:
            return False
        if not self.backend_available:
            message = self.backend_error or f"{self.backend} is not available."
            print(f"[Mouth] Cannot speak: {message}")
            return False

        cleaned = str(text).strip()
        if not cleaned:
            return False

        segments = self._split_for_async_queue(cleaned)
        if not segments:
            return False

        with self._speak_async_lock:
            self._claim_audio_owner()
            with self._async_queue_condition:
                self._stop_requested.clear()
                for segment in segments:
                    while (
                        len(self._async_queue) >= self._max_async_queue
                        and not self._stop_requested.is_set()
                    ):
                        self._async_queue_condition.wait(timeout=0.05)

                    if self._stop_requested.is_set():
                        self._release_audio_owner()
                        return False

                    self._async_queue.append(segment)
                    self._async_queue_condition.notify_all()

                should_start_worker = (
                    self._speak_thread is None or not self._speak_thread.is_alive()
                )
                if should_start_worker:
                    self._speak_thread = threading.Thread(
                        target=self._consume_async_queue,
                        daemon=self._async_thread_daemon,
                        name="TTSAsyncWorker",
                    )
                    self._speak_thread.start()
        return True

    def wait_until_done(self):
        """Block until the current asynchronous speech thread completes."""
        while True:
            thread = self._speak_thread
            if not thread or not thread.is_alive():
                break
            if thread is threading.current_thread():
                break
            thread.join()

    def has_pending_audio(self) -> bool:
        """Return True while audio is playing or queued for playback."""
        thread = self._speak_thread
        thread_alive = bool(thread and thread.is_alive())
        with self._async_queue_condition:
            queue_has_items = bool(self._async_queue)
        return self._is_playing.is_set() or thread_alive or queue_has_items

    def stop(self, wait: bool = True, *, release_owner: bool = True):
        """Stop any active speech output across both supported backends."""
        self._stop_requested.set()

        with self._async_queue_condition:
            self._async_queue.clear()
            self._async_queue_condition.notify_all()

        with self._lock:
            engine = self._active_engine
            process = self._active_process

        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
        if process is not None:
            try:
                process.terminate()
            except Exception:
                pass

        if self.backend in {"kokoro", "piper", "waveglow"} and sd is not None:
            try:
                sd.stop()
            except Exception:
                pass

        if wait:
            self.wait_until_done()

        if release_owner:
            self._release_audio_owner()
