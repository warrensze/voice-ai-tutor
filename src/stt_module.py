import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import os
import threading
from pathlib import Path

# Configuration
MODEL_SIZE = "base.en"  # 'base.en' is instant; 'distil-large-v3' is higher quality
DEVICE = "cuda"  # Force RTX 5070 usage
COMPUTE_TYPE = "float16"
CPU_COMPUTE_TYPE = "int8"
ENERGY_THRESHOLD = float(os.getenv("STT_ENERGY_THRESHOLD", "0.006"))
MAX_IDLE_SECONDS = 5
MIN_DYNAMIC_THRESHOLD = 0.002
MAX_DYNAMIC_THRESHOLD = 0.015
CALIBRATION_CHUNKS = 8
MIN_SPEECH_CHUNKS = 6
SPEAKER_ECHO_THRESHOLD = 0.12  # Very aggressive echo filtering - speaker output is typically 3-5x louder than user speech
ENABLE_ECHO_CANCELLATION = os.getenv("STT_ENABLE_ECHO_CANCELLATION", "1").lower() in {
    "1",
    "true",
    "yes",
}
ENABLE_NOISE_REDUCTION = os.getenv("STT_ENABLE_NOISE_REDUCTION", "1").lower() in {
    "1",
    "true",
    "yes",
}
ENABLE_AUTO_GAIN = os.getenv("STT_ENABLE_AUTO_GAIN", "1").lower() in {
    "1",
    "true",
    "yes",
}
NOISE_REDUCTION_STRENGTH = float(os.getenv("STT_NOISE_REDUCTION_STRENGTH", "0.7"))
NOISE_GATE_MULTIPLIER = float(os.getenv("STT_NOISE_GATE_MULTIPLIER", "1.6"))
TARGET_INPUT_RMS = float(os.getenv("STT_TARGET_INPUT_RMS", "0.08"))
MAX_AUTO_GAIN = float(os.getenv("STT_MAX_AUTO_GAIN", "6.0"))
ECHO_GUARD_SECONDS = float(os.getenv("STT_ECHO_GUARD_SECONDS", "0.8"))
ECHO_START_SUPPRESS_SECONDS = float(
    os.getenv("STT_ECHO_START_SUPPRESS_SECONDS", "0.35")
)
ECHO_START_SUPPRESS_MIN_GAIN = float(
    os.getenv("STT_ECHO_START_SUPPRESS_MIN_GAIN", "0.35")
)
DEBUG_AUDIO = os.getenv("DEBUG_AUDIO", "").lower() in {
    "1",
    "true",
    "yes",
}  # Enable audio debugging


class SpeechToText:
    def __init__(self, tts_instance=None):
        """Initialize Whisper model and audio capture configuration.

        Args:
            tts_instance: Optional TextToSpeech instance to prevent microphone
                         pickup of speaker output during playback.
        """
        self.active_device = DEVICE
        self.model = self._load_model()
        self.tts_instance = tts_instance  # Reference to TTS for playback detection

        # Audio Setup
        self.chunk_size = 512
        self.channels = 1
        self.rate = 16000
        self.input_device = self._resolve_input_device()
        self._last_threshold = ENERGY_THRESHOLD
        self.enable_echo_cancellation = ENABLE_ECHO_CANCELLATION
        self.enable_noise_reduction = ENABLE_NOISE_REDUCTION
        self.enable_auto_gain = ENABLE_AUTO_GAIN

    def _is_tts_recently_active(self) -> bool:
        """Return True if speaker playback is active or just recently ended."""
        if not self.tts_instance:
            return False

        try:
            if self.tts_instance.is_audio_playing():
                return True
        except Exception:
            return False

        recently_played = getattr(self.tts_instance, "recently_played", None)
        if callable(recently_played):
            try:
                return bool(recently_played(ECHO_GUARD_SECONDS))
            except Exception:
                return False

        return False

    def _build_noise_reference(self, audio_buffer: list[np.ndarray]) -> np.ndarray:
        """Build a noise reference from initial calibration chunks."""
        if not audio_buffer:
            return np.array([], dtype=np.float32)

        frames = audio_buffer[:CALIBRATION_CHUNKS]
        if not frames:
            return np.array([], dtype=np.float32)

        return np.concatenate([frame.flatten() for frame in frames]).astype(np.float32)

    def _pre_emphasis(self, audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
        """Apply a light high-pass filter to reduce low-frequency rumble."""
        if audio.size == 0:
            return audio

        emphasized = np.empty_like(audio, dtype=np.float32)
        emphasized[0] = audio[0]
        emphasized[1:] = audio[1:] - (coeff * audio[:-1])
        return emphasized

    def _noise_gate(self, audio: np.ndarray, noise_reference: np.ndarray) -> np.ndarray:
        """Attenuate samples that are likely below useful speech level."""
        if audio.size == 0 or noise_reference.size == 0:
            return audio

        noise_rms = float(np.sqrt(np.mean(noise_reference * noise_reference)))
        if noise_rms <= 0:
            return audio

        gate_threshold = noise_rms * NOISE_GATE_MULTIPLIER
        gated = audio.copy()
        gated[np.abs(gated) < gate_threshold] *= 0.15
        return gated.astype(np.float32)

    def _spectral_noise_reduction(
        self, audio: np.ndarray, noise_reference: np.ndarray
    ) -> np.ndarray:
        """Apply spectral subtraction to reduce steady background noise."""
        if audio.size == 0 or noise_reference.size == 0:
            return audio

        n_fft = int(2 ** np.ceil(np.log2(max(audio.size, 512))))
        signal_spec = np.fft.rfft(audio, n=n_fft)
        noise_spec = np.fft.rfft(noise_reference, n=n_fft)

        signal_mag = np.abs(signal_spec)
        noise_mag = np.abs(noise_spec)
        floor = signal_mag * 0.08
        reduced_mag = np.maximum(
            signal_mag - (NOISE_REDUCTION_STRENGTH * noise_mag),
            floor,
        )

        reduced_spec = reduced_mag * np.exp(1j * np.angle(signal_spec))
        denoised = np.fft.irfft(reduced_spec, n=n_fft)[: audio.size]
        return denoised.astype(np.float32)

    def _auto_gain(self, audio: np.ndarray) -> np.ndarray:
        """Normalize voice level into a target RMS range for Whisper."""
        if audio.size == 0:
            return audio

        rms = float(np.sqrt(np.mean(audio * audio)))
        if rms <= 1e-7:
            return audio

        gain = min(TARGET_INPUT_RMS / rms, MAX_AUTO_GAIN)
        adjusted = np.clip(audio * gain, -1.0, 1.0)
        return adjusted.astype(np.float32)

    def _suppress_echo_start(self, audio: np.ndarray) -> np.ndarray:
        """Suppress early capture where speaker bleed is strongest."""
        if audio.size == 0:
            return audio

        suppress_samples = int(self.rate * ECHO_START_SUPPRESS_SECONDS)
        suppress_samples = min(max(0, suppress_samples), audio.size)
        if suppress_samples <= 0:
            return audio

        faded = audio.copy()
        fade_curve = np.linspace(
            ECHO_START_SUPPRESS_MIN_GAIN,
            1.0,
            suppress_samples,
            dtype=np.float32,
        )
        faded[:suppress_samples] *= fade_curve
        return faded.astype(np.float32)

    def _post_process_audio(
        self,
        audio: np.ndarray,
        noise_reference: np.ndarray,
        had_speaker_activity: bool,
    ) -> np.ndarray:
        """Run echo/noise/volume processing on captured audio before Whisper."""
        if audio.size == 0:
            return audio

        processed = audio.astype(np.float32)

        # Remove DC offset for cleaner downstream processing.
        processed = processed - np.mean(processed)

        if self.enable_noise_reduction:
            processed = self._noise_gate(processed, noise_reference)
            processed = self._spectral_noise_reduction(processed, noise_reference)

        if self.enable_echo_cancellation and (
            had_speaker_activity or self._is_tts_recently_active()
        ):
            processed = self._suppress_echo_start(processed)

        processed = self._pre_emphasis(processed)

        if self.enable_auto_gain:
            processed = self._auto_gain(processed)

        return np.clip(processed, -1.0, 1.0).astype(np.float32)

    def _resolve_input_device(self):
        """Resolve input device from env override or system default."""
        override = os.getenv("STT_INPUT_DEVICE", "").strip()
        if not override:
            return None

        if override.isdigit():
            return int(override)

        return override

    def _get_device_default_samplerate(self) -> int | None:
        """Return the input device default sample rate, if available."""
        try:
            target = self.input_device if self.input_device is not None else None
            info = sd.query_devices(target, "input")
            default_rate = int(float(info.get("default_samplerate", 0)))
            return default_rate if default_rate > 0 else None
        except Exception:
            return None

    def _adaptive_threshold(self, audio_buffer: list[np.ndarray]) -> float:
        """Compute an adaptive threshold from early background-noise samples."""
        if not audio_buffer:
            return ENERGY_THRESHOLD

        frames = audio_buffer[:CALIBRATION_CHUNKS]
        if not frames:
            return ENERGY_THRESHOLD

        noise = np.concatenate([frame.flatten() for frame in frames])
        if noise.size == 0:
            return ENERGY_THRESHOLD

        baseline_rms = float(np.sqrt(np.mean(noise * noise)))
        dynamic = baseline_rms * 1.8
        threshold = max(MIN_DYNAMIC_THRESHOLD, min(dynamic, MAX_DYNAMIC_THRESHOLD))
        return threshold

    def _resample_audio(
        self, audio: np.ndarray, src_rate: int, dst_rate: int
    ) -> np.ndarray:
        """Resample mono float audio with linear interpolation."""
        if src_rate == dst_rate or audio.size == 0:
            return audio

        duration = audio.size / float(src_rate)
        out_size = max(1, int(duration * dst_rate))

        src_positions = np.linspace(0.0, duration, num=audio.size, endpoint=False)
        dst_positions = np.linspace(0.0, duration, num=out_size, endpoint=False)
        return np.interp(dst_positions, src_positions, audio).astype(np.float32)

    def _load_model(self):
        """Load Whisper with GPU preference and CPU fallback."""
        try:
            model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
            print(f"[Ears] Whisper loaded on {DEVICE} ({COMPUTE_TYPE}).")
            self.active_device = DEVICE
            return model
        except Exception as error:
            error_text = str(error).lower()
            if "cublas" in error_text or "cuda" in error_text:
                print(
                    "[Ears] CUDA libraries are unavailable. Falling back to CPU transcription."
                )
                model = WhisperModel(
                    MODEL_SIZE,
                    device="cpu",
                    compute_type=CPU_COMPUTE_TYPE,
                )
                print(f"[Ears] Whisper loaded on cpu ({CPU_COMPUTE_TYPE}).")
                self.active_device = "cpu"
                return model
            raise

    def _is_speaker_echo(
        self, chunk_rms: float, speaker_rms_baseline: float = 0.15
    ) -> bool:
        """Detect if audio chunk is likely speaker output (echo) vs user speech.

        Speaker output is typically much louder than user speech when picked up
        by microphone during echo. Use a high RMS threshold to filter it out.
        """
        return chunk_rms > speaker_rms_baseline

    def listen(
        self,
        *,
        max_idle_seconds: float | None = None,
        announce: bool = True,
        silence_chunks_to_stop: int = 30,
        beam_size: int = 5,
        stop_event: threading.Event | None = None,
    ):
        """Capture microphone audio, detect speech, and return transcribed text.

        Allows barge-in while filtering out speaker echo during agent playback.
        """
        if announce:
            print("\n[Ears] Listening... (Speak now)")

        active_threshold = ENERGY_THRESHOLD
        idle_window_seconds = (
            max_idle_seconds if max_idle_seconds is not None else MAX_IDLE_SECONDS
        )

        def capture_once(samplerate: int) -> tuple[np.ndarray, np.ndarray, bool]:
            """Capture a single utterance at the requested sample rate."""
            nonlocal active_threshold
            audio_buffer: list[np.ndarray] = []
            silent_chunks = 0
            idle_chunks = 0
            speaking = False
            speech_chunks = 0
            had_speaker_activity = False
            max_idle_chunks = int((samplerate / self.chunk_size) * idle_window_seconds)

            # If TTS is playing, use a much higher threshold to filter out speaker echo
            tts_is_playing = self.tts_instance and self.tts_instance.is_audio_playing()
            echo_filter_threshold = (
                SPEAKER_ECHO_THRESHOLD  # Aggressive filtering for speaker echo
            )

            def audio_callback(indata, frames, time, status):
                """Append each audio callback frame into the rolling buffer."""
                if status:
                    print(f"Audio error: {status}")
                audio_buffer.append(indata.copy())

            with sd.InputStream(
                channels=self.channels,
                samplerate=samplerate,
                blocksize=self.chunk_size,
                callback=audio_callback,
                dtype=np.float32,
                device=self.input_device,
            ):
                while True:
                    if stop_event is not None and stop_event.is_set():
                        return (
                            np.array([], dtype=np.float32),
                            np.array([], dtype=np.float32),
                            had_speaker_activity,
                        )

                    if audio_buffer:
                        if len(audio_buffer) == CALIBRATION_CHUNKS:
                            active_threshold = self._adaptive_threshold(audio_buffer)
                            self._last_threshold = active_threshold

                        recent_chunk = audio_buffer[-1].flatten()

                        # Use RMS energy as lightweight speech detection.
                        speech_prob = float(
                            np.sqrt(np.mean(recent_chunk * recent_chunk))
                        )

                        # When TTS is playing, filter out speaker echo but allow loud user speech
                        current_tts_state = (
                            self.tts_instance and self._is_tts_recently_active()
                        )
                        if current_tts_state:
                            had_speaker_activity = True
                        if current_tts_state and speech_prob > echo_filter_threshold:
                            # This is likely echo, not user speech (speaker output is much louder)
                            # Skip this chunk to prevent false trigger on speaker audio
                            if DEBUG_AUDIO:
                                print(
                                    f"[Echo] Filtered speaker audio (RMS={speech_prob:.4f} > threshold={echo_filter_threshold:.4f})"
                                )
                            idle_chunks += 1
                            silent_chunks = 0
                            sd.sleep(50)
                            continue
                        elif current_tts_state and speech_prob > (active_threshold * 2):
                            # User speech detected as significantly above threshold during playback
                            # This is a qualified interruption - stop waiting and return it
                            if DEBUG_AUDIO:
                                print(
                                    f"[Interrupt] User detected (RMS={speech_prob:.4f} > {active_threshold * 2:.4f})"
                                )
                            speaking = True
                            speech_chunks = sum(
                                1 for _ in range(silence_chunks_to_stop)
                            )
                            break
                        elif current_tts_state and speech_prob <= active_threshold:
                            # Very quiet audio during playback - treat as idle
                            idle_chunks += 1
                            silent_chunks = 0
                            sd.sleep(50)
                            continue

                        if speech_prob > active_threshold:
                            speaking = True
                            speech_chunks += 1
                            silent_chunks = 0
                            idle_chunks = 0
                        elif speaking:
                            silent_chunks += 1
                        else:
                            idle_chunks += 1

                        # If speaking and then 1 second of silence, stop recording
                        if speaking and silent_chunks > silence_chunks_to_stop:
                            break
                        if not speaking and idle_chunks > max_idle_chunks:
                            return (
                                np.array([], dtype=np.float32),
                                np.array([], dtype=np.float32),
                                had_speaker_activity,
                            )

                    sd.sleep(50)  # Small delay to prevent busy waiting

            if not audio_buffer:
                return (
                    np.array([], dtype=np.float32),
                    np.array([], dtype=np.float32),
                    had_speaker_activity,
                )

            if speech_chunks < MIN_SPEECH_CHUNKS:
                return (
                    np.array([], dtype=np.float32),
                    np.array([], dtype=np.float32),
                    had_speaker_activity,
                )
            captured = np.concatenate([chunk.flatten() for chunk in audio_buffer])
            noise_reference = self._build_noise_reference(audio_buffer)
            return captured, noise_reference, had_speaker_activity

        capture_rate = self.rate
        noise_reference = np.array([], dtype=np.float32)
        had_speaker_activity = False
        try:
            full_audio, noise_reference, had_speaker_activity = capture_once(
                capture_rate
            )
        except Exception as first_error:
            fallback_rate = self._get_device_default_samplerate()
            if fallback_rate and fallback_rate != capture_rate:
                if announce:
                    print(
                        f"[Ears] Capture at {capture_rate} Hz failed on "
                        f"device={self.input_device!r}. Retrying at {fallback_rate} Hz."
                    )
                try:
                    capture_rate = fallback_rate
                    (
                        full_audio,
                        noise_reference,
                        had_speaker_activity,
                    ) = capture_once(capture_rate)
                except Exception as second_error:
                    if announce:
                        print(
                            f"[Ears] Audio capture failed on device={self.input_device!r}, "
                            f"samplerate={capture_rate}: {second_error}"
                        )
                    return ""
            else:
                if announce:
                    print(
                        f"[Ears] Audio capture failed on device={self.input_device!r}, "
                        f"samplerate={capture_rate}: {first_error}"
                    )
                return ""

        if full_audio.size == 0:
            return ""

        if stop_event is not None and stop_event.is_set():
            return ""

        if capture_rate != self.rate:
            full_audio = self._resample_audio(full_audio, capture_rate, self.rate)
            if noise_reference.size:
                noise_reference = self._resample_audio(
                    noise_reference, capture_rate, self.rate
                )

        full_audio = self._post_process_audio(
            full_audio,
            noise_reference=noise_reference,
            had_speaker_activity=had_speaker_activity,
        )

        if announce:
            capture_label = (
                f"{capture_rate} Hz -> {self.rate} Hz"
                if capture_rate != self.rate
                else f"{capture_rate} Hz"
            )
            print(
                f"[Ears] Captured {full_audio.size} samples at {capture_label} "
                f"(threshold={self._last_threshold:.4f})."
            )

        def run_transcribe() -> str:
            """Run Whisper transcription and eagerly consume the segment iterator."""
            # Important: consume the generator inside the try/except path,
            # because CUDA errors can happen on iteration, not on creation.
            segments, _ = self.model.transcribe(
                full_audio,
                beam_size=beam_size,
                vad_filter=True,
            )
            return " ".join(segment.text for segment in segments).strip()

        try:
            text = run_transcribe()
        except Exception as error:
            error_text = str(error).lower()
            if self.active_device == "cuda" and (
                "cublas" in error_text or "cuda" in error_text
            ):
                print(
                    "[Ears] CUDA runtime failed during transcription. Retrying on CPU."
                )
                try:
                    self.model = WhisperModel(
                        MODEL_SIZE,
                        device="cpu",
                        compute_type=CPU_COMPUTE_TYPE,
                    )
                    self.active_device = "cpu"
                    text = run_transcribe()
                except Exception as cpu_error:
                    if announce:
                        print(f"[Ears] CPU fallback transcription failed: {cpu_error}")
                    return ""
            else:
                if announce:
                    print(f"[Ears] Transcription failed: {error}")
                return ""

        return text

    def transcribe_file(self, audio_path: str | Path, *, beam_size: int = 5) -> str:
        """Transcribe an uploaded audio file with the local Whisper model."""
        source_path = Path(audio_path)
        if not source_path.exists():
            return ""

        def run_transcribe() -> str:
            segments, _ = self.model.transcribe(
                str(source_path),
                beam_size=beam_size,
                vad_filter=True,
            )
            return " ".join(segment.text for segment in segments).strip()

        try:
            return run_transcribe()
        except Exception as error:
            error_text = str(error).lower()
            if self.active_device == "cuda" and (
                "cublas" in error_text or "cuda" in error_text
            ):
                print(
                    "[Ears] CUDA runtime failed during file transcription. Retrying on CPU."
                )
                try:
                    self.model = WhisperModel(
                        MODEL_SIZE,
                        device="cpu",
                        compute_type=CPU_COMPUTE_TYPE,
                    )
                    self.active_device = "cpu"
                    return run_transcribe()
                except Exception as cpu_error:
                    print(f"[Ears] CPU fallback file transcription failed: {cpu_error}")
                    return ""

            print(f"[Ears] File transcription failed: {error}")
            return ""
