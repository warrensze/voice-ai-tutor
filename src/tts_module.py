import re
import threading
import os
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
        voices[voice_id] = {
            "id": voice_id,
            "label": voice_id.replace("_", " "),
            "path": str(path.relative_to(data_dir)),
            "available": True,
        }
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
            "label": voice_id.replace("_", " "),
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
        """Initialize voice settings and select the best available TTS backend."""
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
        self._kokoro_marker_path = (
            Path(__file__).resolve().parents[1]
            / f".kokoro_assets_{self._kokoro_repo_id.replace('/', '_')}.ready"
        )
        if self._kokoro_offline_after_preload and self._kokoro_marker_path.exists():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.output_device_override = self._resolve_output_override()
        self.require_kokoro = os.getenv("TTS_REQUIRE_KOKORO", "1").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        default_order = "kokoro,piper,waveglow,pyttsx3"
        configured_order = backend_order or os.getenv("TTS_BACKEND_ORDER", default_order)
        if isinstance(configured_order, str):
            requested_order = [
                name.strip().lower()
                for name in configured_order.split(",")
                if name.strip()
            ]
        else:
            requested_order = [
                str(name).strip().lower() for name in configured_order if str(name).strip()
            ]
        if backend:
            selected_backend = str(backend).strip().lower()
            requested_order = [selected_backend] + [
                name for name in requested_order if name != selected_backend
            ]
        self.backend_order = list(dict.fromkeys(requested_order or ["pyttsx3"]))
        self.backend = "pyttsx3"
        print(
            f"[Mouth] Initializing TTS backend (order: {', '.join(self.backend_order)})"
        )
        self._configure_backend()
        kokoro_was_primary = self.backend_order and self.backend_order[0] == "kokoro"
        if self.backend != "kokoro" and kokoro_was_primary:
            message = (
                f"[Mouth] Kokoro was requested but backend resolved to '{self.backend}'. "
                "Set TTS_REQUIRE_KOKORO=0 to allow fallback backends."
            )
            if self.require_kokoro:
                raise RuntimeError(message)
            print(message)
        print(f"[Mouth] Using {self.backend.upper()} backend for speech output")
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
        """Play audio on configured output device, then retry on system default if needed."""
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
                self._is_playing.clear()
                self._last_playback_end = time.monotonic()
                return
            except Exception:
                # Retry on system default in case selected output was unavailable.
                try:
                    sd.stop()
                    sd.play(waveform, sample_rate)
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

    def _configure_backend(self):
        """Select the first available backend from configured backend order."""
        self.backend = "pyttsx3"

        for candidate in self.backend_order:
            if candidate == "waveglow":
                if self._init_waveglow_backend():
                    print("[Mouth] WaveGlow backend initialized successfully")
                    self.backend = "waveglow"
                    return
                else:
                    print(
                        "[Mouth] WaveGlow backend initialization failed, trying next backend"
                    )
            elif candidate == "kokoro":
                if self._init_kokoro_backend():
                    print("[Mouth] Kokoro backend initialized successfully")
                    self.backend = "kokoro"
                    return
                else:
                    print(
                        "[Mouth] Kokoro backend initialization failed, trying next backend"
                    )
            elif candidate == "piper":
                if self._init_piper_backend():
                    print("[Mouth] Piper backend initialized successfully")
                    self.backend = "piper"
                    return
                else:
                    print(
                        "[Mouth] Piper backend initialization failed, trying next backend"
                    )
            elif candidate == "pyttsx3":
                print("[Mouth] Falling back to pyttsx3 backend")
                self.backend = "pyttsx3"
                return

    def _init_kokoro_backend(self) -> bool:
        """Initialize Kokoro backend if dependencies are available.

        Kokoro-82M supports voice blending with the syntax:
        - Single voice: 'af_heart'
        - Blended voices: 'af_heart+af_bella' (50/50 blend)
        - Weighted blend: 'af_heart+af_bella@0.7' (70% af_heart, 30% af_bella)
        """
        if sd is None:
            print("[Mouth] Kokoro dependency not available: sounddevice")
            return False

        if KPipeline is None:
            print(
                "[Mouth] Kokoro import failed. "
                f"Original error: {repr(KOKORO_IMPORT_ERROR)}"
            )
            print(
                "[Mouth] If you are on Python 3.14, use Python 3.12/3.13 for kokoro compatibility."
            )
            return False

        try:
            import torch
        except Exception as error:
            print(f"[Mouth] PyTorch is required for kokoro. Import failed: {error}")
            return False

        requested_device = os.getenv("KOKORO_DEVICE", "cuda").strip().lower()
        if not requested_device:
            requested_device = "cuda"

        allow_cpu = os.getenv("KOKORO_ALLOW_CPU", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }

        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            if not allow_cpu:
                print(
                    "[Mouth] Kokoro requires CUDA but torch.cuda.is_available() is False. "
                    "Set KOKORO_ALLOW_CPU=1 to permit CPU fallback."
                )
                return False
            requested_device = "cpu"
            print(
                "[Mouth] CUDA unavailable; using CPU for kokoro due KOKORO_ALLOW_CPU=1"
            )

        if requested_device == "cpu" and not allow_cpu:
            print(
                "[Mouth] KOKORO_DEVICE is set to CPU but KOKORO_ALLOW_CPU is not enabled."
            )
            return False

        try:
            self._kokoro_pipeline = KPipeline(
                lang_code="a",
                repo_id=self._kokoro_repo_id,
                device=requested_device,
            )
            if requested_device.startswith("cuda") and torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[Mouth] Kokoro-82M initialized on CUDA GPU: {gpu_name}")
            else:
                print(f"[Mouth] Kokoro-82M initialized on {requested_device}")

            self._preload_kokoro_assets()
            return True
        except Exception as e:
            print(f"[Mouth] Failed to initialize Kokoro-82M: {e}")
            self._kokoro_pipeline = None
            return False

    def _init_waveglow_backend(self) -> bool:
        """Prepare WaveGlow backend with lazy model loading."""
        if WaveGlowSynthesizer is None or WaveGlowConfig is None or sd is None:
            print(
                "[Mouth] WaveGlow dependencies not available (WaveGlowSynthesizer, WaveGlowConfig, or sounddevice)"
            )
            return False

        try:
            import torch
        except Exception:
            print("[Mouth] PyTorch not available, cannot use WaveGlow")
            return False

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
                print(
                    "[Mouth] WaveGlow unavailable: CUDA is not available in torch. "
                    "Set WAVEGLOW_ALLOW_CPU=1 to use CPU or install CUDA-enabled torch."
                )
                return False
            requested_device = "cpu"
            print("[Mouth] CUDA not available, falling back to CPU for WaveGlow")

        if requested_device == "cpu" and not allow_cpu:
            print("[Mouth] CPU requested but WAVEGLOW_ALLOW_CPU not enabled")
            return False

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
            print(f"[Mouth] Failed to initialize WaveGlowSynthesizer: {e}")
            self._waveglow_synth = None
            return False

    def _init_piper_backend(self) -> bool:
        """Initialize Piper with a local ONNX voice file."""
        if sd is None:
            print("[Mouth] Piper dependency not available: sounddevice")
            return False
        if PiperVoice is None:
            print(
                "[Mouth] Piper import failed. "
                f"Original error: {repr(PIPER_IMPORT_ERROR)}"
            )
            return False

        voice_path = self._resolve_piper_voice_path(self.voice)
        if voice_path is None:
            print(
                f"[Mouth] Piper voice '{self.voice}' was not found in "
                f"{self._piper_data_dir}."
            )
            return False

        try:
            self._piper_voice = PiperVoice.load(
                str(voice_path),
                use_cuda=self._piper_use_cuda,
            )
            self._piper_voice_path = voice_path
            return True
        except Exception as error:
            print(f"[Mouth] Failed to load Piper voice {voice_path}: {error}")
            self._piper_voice = None
            self._piper_voice_path = None
            return False

    def _resolve_piper_voice_path(self, voice: str) -> Path | None:
        """Find a Piper ONNX voice by explicit path or voice id."""
        if not voice:
            return None

        candidate = Path(voice).expanduser()
        if candidate.suffix.lower() == ".onnx" and candidate.exists():
            return candidate

        data_dir = self._piper_data_dir
        if not data_dir.is_absolute():
            data_dir = Path(__file__).resolve().parents[1] / data_dir

        candidates = [
            data_dir / f"{voice}.onnx",
            data_dir / voice / f"{voice}.onnx",
            data_dir / voice,
        ]
        for path in candidates:
            if path.suffix.lower() == ".onnx" and path.exists():
                return path
        return None

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
        self.voice = new_voice

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

    def _speak_chunks(self, text: str):
        """Play chunks through the active backend until completed or stopped."""
        # Use the selected backend
        if (
            self.backend == "waveglow"
            and self._waveglow_synth is not None
            and sd is not None
        ):
            self._speak_with_waveglow(text)
            return

        if (
            self.backend == "kokoro"
            and self._kokoro_pipeline is not None
            and sd is not None
        ):
            self._speak_with_kokoro(text)
            return

        if self.backend == "piper" and self._piper_voice is not None and sd is not None:
            self._speak_with_piper(text)
            return

        self._speak_with_engine_chunks(text)

    def _speak_with_waveglow(self, text: str):
        """Speak text with WaveGlow and stream generated audio through sounddevice."""
        if self._waveglow_synth is None or sd is None:
            print("[Mouth] WaveGlow synthesizer not initialized, using fallback")
            self._waveglow_failed = True
            # Fall through to kokoro or pyttsx3
            if self._kokoro_pipeline is not None and sd is not None:
                self._speak_with_kokoro(text)
            else:
                self._speak_with_engine_chunks(text)
            return

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
                # WaveGlow failed, mark it and fall through to next backend
                print(f"[Mouth] WaveGlow playback error: {e}")
                self._waveglow_failed = True
                print(
                    "[Mouth] Switching to fallback backend for current and remaining text"
                )
                # Fall back for this chunk and any remaining using kokoro or pyttsx3
                if self._kokoro_pipeline is not None and sd is not None:
                    self._speak_with_kokoro(chunk)
                else:
                    self._speak_with_engine(chunk)
                break

    def _speak_with_engine_chunks(self, text: str):
        """Speak all chunks with one pyttsx3 engine run."""
        chunks = [chunk for chunk in self._iter_chunks(text)]
        if not chunks or self._stop_requested.is_set():
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

    def _speak_with_piper(self, text: str):
        """Speak text with Piper and stream audio chunks through sounddevice."""
        if self._piper_voice is None:
            self._speak_with_engine_chunks(text)
            return

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
            print(f"[Mouth] Piper playback error: {error}")
            print("[Mouth] Falling back to pyttsx3 for this response")
            self._speak_with_engine_chunks(text)

    def _speak_with_kokoro(self, text: str):
        """Speak text with Kokoro-82M and stream generated audio through sounddevice.

        Supports voice blending for natural voice variations:
        - af_heart+af_bella (50/50 blend)
        - af_heart+af_bella@0.7 (70% af_heart, 30% af_bella)
        """
        if self._kokoro_pipeline is None:
            return

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
            print(f"[Mouth] Kokoro playback error: {error}")
            print("[Mouth] Falling back to pyttsx3 for this response")
            self._speak_with_engine_chunks(text)

    def speak(self, text):
        """Speak text synchronously, replacing any in-flight playback."""
        if not text:
            return

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

                self._speak_chunks(next_text)
        finally:
            self._release_audio_owner()

        with self._async_queue_condition:
            self._speak_thread = None
            self._async_queue_condition.notify_all()

    def speak_async(self, text):
        """Speak text in a background thread so main flow can continue."""
        if not text:
            return

        cleaned = str(text).strip()
        if not cleaned:
            return

        segments = self._split_for_async_queue(cleaned)
        if not segments:
            return

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
                        return

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

        if engine is not None:
            try:
                engine.stop()
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
