import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import os

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

        def capture_once(samplerate: int) -> np.ndarray:
            """Capture a single utterance at the requested sample rate."""
            nonlocal active_threshold
            audio_buffer: list[np.ndarray] = []
            silent_chunks = 0
            idle_chunks = 0
            speaking = False
            speech_chunks = 0
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
                            self.tts_instance and self.tts_instance.is_audio_playing()
                        )
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
                            return np.array([], dtype=np.float32)

                    sd.sleep(50)  # Small delay to prevent busy waiting

            if not audio_buffer:
                return np.array([], dtype=np.float32)

            if speech_chunks < MIN_SPEECH_CHUNKS:
                return np.array([], dtype=np.float32)

            return np.concatenate([chunk.flatten() for chunk in audio_buffer])

        capture_rate = self.rate
        try:
            full_audio = capture_once(capture_rate)
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
                    full_audio = capture_once(capture_rate)
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

        if capture_rate != self.rate:
            full_audio = self._resample_audio(full_audio, capture_rate, self.rate)

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
