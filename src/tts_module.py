import re
import threading

import pyttsx3

try:
    from kokoro import KPipeline  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    KPipeline = None

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional dependency
    sd = None


class TextToSpeech:
    def __init__(self, voice: str = "af_heart", speed: float = 1.0):
        """Initialize voice settings and select the best available TTS backend."""
        self.rate = 175
        self.volume = 1.0
        self.voice = voice
        self.speed = speed
        self._lock = threading.Lock()
        self._active_engine = None
        self._stop_requested = threading.Event()
        self._speak_thread = None
        self._kokoro_pipeline = None
        self.backend = "pyttsx3"
        self._configure_backend()

    def _configure_backend(self):
        """Prefer Kokoro when available, otherwise fall back to pyttsx3."""
        if KPipeline is None or sd is None:
            self.backend = "pyttsx3"
            return

        try:
            self._kokoro_pipeline = KPipeline(lang_code="a", device="cuda")
            self.backend = "kokoro"
        except Exception:
            self._kokoro_pipeline = None
            self.backend = "pyttsx3"

    def set_voice(self, new_voice: str):
        """Update the active voice preset used by subsequent speech calls."""
        if not new_voice:
            return
        self.voice = new_voice

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

    def _speak_with_engine(self, text: str):
        """Speak a single chunk using the pyttsx3 backend."""
        engine = self._create_engine()
        try:
            with self._lock:
                self._active_engine = engine
            engine.say(text)
            engine.runAndWait()
        finally:
            with self._lock:
                self._active_engine = None
            engine.stop()

    def _speak_chunks(self, text: str):
        """Play chunks through the active backend until completed or stopped."""
        if (
            self.backend == "kokoro"
            and self._kokoro_pipeline is not None
            and sd is not None
        ):
            self._speak_with_kokoro(text)
            return

        for chunk in self._iter_chunks(text):
            if self._stop_requested.is_set():
                return

            try:
                self._speak_with_engine(chunk)
            except Exception:
                if self._stop_requested.is_set():
                    return
                self._speak_with_engine(chunk)

    def _speak_with_kokoro(self, text: str):
        """Speak text with Kokoro and stream generated audio through sounddevice."""
        generator = self._kokoro_pipeline(
            text,
            voice=self.voice,
            speed=self.speed,
            split_pattern=r"\n+",
        )
        for _, _, audio in generator:
            if self._stop_requested.is_set():
                return
            sd.play(audio, 24000)
            sd.wait()

    def speak(self, text):
        """Speak text synchronously, replacing any in-flight playback."""
        if not text:
            return

        self.stop()
        self._stop_requested.clear()
        self._speak_chunks(text)

    def speak_async(self, text):
        """Speak text in a background thread so main flow can continue."""
        if not text:
            return

        self.stop()
        self._stop_requested.clear()

        def runner():
            self._speak_chunks(text)

        self._speak_thread = threading.Thread(target=runner, daemon=True)
        self._speak_thread.start()

    def wait_until_done(self):
        """Block until the current asynchronous speech thread completes."""
        thread = self._speak_thread
        if thread and thread.is_alive():
            thread.join()

    def stop(self, wait: bool = True):
        """Stop any active speech output across both supported backends."""
        self._stop_requested.set()

        with self._lock:
            engine = self._active_engine

        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

        if self.backend == "kokoro" and sd is not None:
            try:
                sd.stop()
            except Exception:
                pass

        if wait:
            self.wait_until_done()
