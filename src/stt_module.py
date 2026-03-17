import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# Configuration
MODEL_SIZE = "base.en"  # 'base.en' is instant; 'distil-large-v3' is higher quality
DEVICE = "cuda"  # Force RTX 5070 usage
COMPUTE_TYPE = "float16"
CPU_COMPUTE_TYPE = "int8"
ENERGY_THRESHOLD = 0.01
MAX_IDLE_SECONDS = 5


class SpeechToText:
    def __init__(self):
        """Initialize Whisper model and audio capture configuration."""
        self.active_device = DEVICE
        self.model = self._load_model()

        # Audio Setup
        self.chunk_size = 512
        self.channels = 1
        self.rate = 16000

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

    def listen(
        self,
        *,
        max_idle_seconds: float | None = None,
        announce: bool = True,
        silence_chunks_to_stop: int = 30,
        beam_size: int = 5,
    ):
        """Capture microphone audio, detect speech, and return transcribed text."""
        if announce:
            print("\n[Ears] Listening... (Speak now)")

        audio_buffer = []
        silent_chunks = 0
        idle_chunks = 0
        speaking = False
        idle_window_seconds = (
            max_idle_seconds if max_idle_seconds is not None else MAX_IDLE_SECONDS
        )
        max_idle_chunks = int((self.rate / self.chunk_size) * idle_window_seconds)

        def audio_callback(indata, frames, time, status):
            """Append each audio callback frame into the rolling buffer."""
            if status:
                print(f"Audio error: {status}")
            audio_buffer.append(indata.copy())

        try:
            # Record audio using sounddevice
            with sd.InputStream(
                channels=self.channels,
                samplerate=self.rate,
                blocksize=self.chunk_size,
                callback=audio_callback,
                dtype=np.float32,
            ):
                while True:
                    if audio_buffer:
                        recent_chunk = audio_buffer[-1].flatten()

                        # Use RMS energy as lightweight speech detection.
                        speech_prob = float(
                            np.sqrt(np.mean(recent_chunk * recent_chunk))
                        )

                        if speech_prob > ENERGY_THRESHOLD:
                            speaking = True
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
                            return ""

                    sd.sleep(50)  # Small delay to prevent busy waiting
        except Exception as error:
            if announce:
                print(f"[Ears] Audio capture failed: {error}")
            return ""

        if not audio_buffer:
            return ""

        # Process the buffer
        full_audio = np.concatenate([chunk.flatten() for chunk in audio_buffer])

        def run_transcribe() -> str:
            """Run Whisper transcription and eagerly consume the segment iterator."""
            # Important: consume the generator inside the try/except path,
            # because CUDA errors can happen on iteration, not on creation.
            segments, _ = self.model.transcribe(full_audio, beam_size=beam_size)
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
