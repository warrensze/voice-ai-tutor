"""Optional WaveGlow-based TTS backend.

This module intentionally fails gracefully when torch or model checkpoints
are unavailable. The caller should fallback to another backend.
"""

from __future__ import annotations

import os
from pathlib import Path
import urllib.request
from dataclasses import dataclass
import warnings
import io
import contextlib
import threading

import numpy as np

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.nn\.utils\.weight_norm` is deprecated.*",
    category=FutureWarning,
)


@dataclass
class WaveGlowConfig:
    """Runtime configuration for WaveGlow synthesis."""

    # Prefer CUDA by default for quality/latency, but gracefully fall back.
    device: str = "cuda"
    sample_rate: int = 22050
    sigma: float = 0.8


class WaveGlowSynthesizer:
    """Synthesize speech from text using Tacotron2 + WaveGlow via torch hub."""

    def __init__(self, config: WaveGlowConfig | None = None):
        self.config = config or WaveGlowConfig()
        self._torch = None
        self._model_utils = None
        self._tacotron2 = None
        self._waveglow = None
        self._device = "cpu"
        self._ready = False
        self._load_lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def device(self) -> str:
        return self._device

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def load(self):
        """Load torch and WaveGlow model components lazily."""
        if self._ready:
            return

        with self._load_lock:
            if self._ready:
                return

            try:
                import torch
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("PyTorch is not installed.") from exc

            self._torch = torch
            self._device = self._select_device(torch)

            # Optional override for corporate/offline environments that mirror models.
            torch_hub_repo = os.getenv(
                "WAVEGLOW_TORCH_HUB_REPO", "NVIDIA/DeepLearningExamples:torchhub"
            )

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*pytorch_quantization module not found.*",
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        category=UserWarning,
                        module=r".*image_classification\\.models\\.(common|efficientnet)",
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=".*untrusted repository.*",
                        category=UserWarning,
                    )
                    self._model_utils = self._hub_load(
                        torch, torch_hub_repo, "nvidia_tts_utils"
                    )

                # NVIDIA hub entrypoints deserialize checkpoints onto CUDA by default.
                # For CPU-only machines, load model skeletons and inject state dicts with
                # explicit map_location to avoid startup failure.
                if self._device == "cpu":
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=".*pytorch_quantization module not found.*",
                            category=UserWarning,
                        )
                        warnings.filterwarnings(
                            "ignore",
                            category=UserWarning,
                            module=r".*image_classification\\.models\\.(common|efficientnet)",
                        )
                        self._tacotron2 = self._hub_load(
                            torch, torch_hub_repo, "nvidia_tacotron2", pretrained=False
                        )
                        self._waveglow = self._hub_load(
                            torch, torch_hub_repo, "nvidia_waveglow", pretrained=False
                        )

                    self._load_pretrained_state_dict(
                        torch, self._tacotron2, "tacotron2"
                    )
                    self._load_pretrained_state_dict(torch, self._waveglow, "waveglow")
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=".*pytorch_quantization module not found.*",
                            category=UserWarning,
                        )
                        warnings.filterwarnings(
                            "ignore",
                            category=UserWarning,
                            module=r".*image_classification\\.models\\.(common|efficientnet)",
                        )
                        self._tacotron2 = self._hub_load(
                            torch, torch_hub_repo, "nvidia_tacotron2"
                        )
                        self._waveglow = self._hub_load(
                            torch, torch_hub_repo, "nvidia_waveglow"
                        )
            except Exception as exc:  # pragma: no cover - network/model availability
                raise RuntimeError(
                    "Unable to load WaveGlow models from torch hub."
                ) from exc

            self._tacotron2 = self._tacotron2.to(self._device).eval()
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*torch\.nn\.utils\.weight_norm.*",
                    category=FutureWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    module=r"torch\.nn\.utils\.weight_norm",
                )
                self._waveglow = self._waveglow.remove_weightnorm(self._waveglow)
            self._waveglow = self._waveglow.to(self._device).eval()
            self._ready = True

    def _hub_load(self, torch, repo: str, model: str, **kwargs):
        """Load torch hub models quietly to keep console output readable."""
        sink_out = io.StringIO()
        sink_err = io.StringIO()
        with contextlib.redirect_stdout(sink_out), contextlib.redirect_stderr(sink_err):
            return torch.hub.load(repo, model, **kwargs)

    def synthesize(self, text: str) -> np.ndarray:
        """Return mono float32 waveform in the range [-1, 1]."""
        if not text or not text.strip():
            return np.array([], dtype=np.float32)

        if not self._ready:
            self.load()

        torch = self._torch
        assert torch is not None
        assert self._model_utils is not None
        assert self._tacotron2 is not None
        assert self._waveglow is not None

        use_cpu = self._device == "cpu"
        sequence, lengths = self._model_utils.prepare_input_sequence(
            [text], cpu_run=use_cpu
        )
        sequence = sequence.to(self._device)
        lengths = lengths.to(self._device)

        with torch.no_grad():
            mel, mel_lengths, _ = self._tacotron2.infer(sequence, lengths)
            audio = self._waveglow.infer(mel, sigma=self.config.sigma)
            audio = audio[:, : mel_lengths.max().item() * 256]

        waveform = audio[0].data.cpu().numpy().astype(np.float32)
        if waveform.size == 0:
            return waveform

        max_abs = float(np.max(np.abs(waveform)))
        if max_abs > 1.0:
            waveform = waveform / max_abs

        return waveform

    def _select_device(self, torch) -> str:
        """Select device based on config and availability."""
        requested = (self.config.device or "").strip().lower()
        if requested in {"", "auto"}:
            requested = "cuda"

        if requested.startswith("cuda"):
            return requested if torch.cuda.is_available() else "cpu"

        if requested == "cpu":
            return "cpu"

        # Unknown values fallback to CUDA-first behavior.
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_pretrained_state_dict(self, torch, model, model_name: str):
        """Download and load pretrained checkpoint state dict for CPU-safe init."""
        checkpoint_url = self._checkpoint_url(model_name)
        checkpoint_file = self._download_checkpoint(torch, checkpoint_url)

        map_location = self._device
        checkpoint = torch.load(checkpoint_file, map_location=map_location)
        state_dict = checkpoint.get("state_dict", checkpoint)
        cleaned = self._unwrap_distributed_state(state_dict)
        model.load_state_dict(cleaned)

    def _download_checkpoint(self, torch, checkpoint_url: str) -> Path:
        """Download a checkpoint file into torch cache if it is not present."""
        model_dir = Path(torch.hub._get_torch_home()) / "checkpoints"
        model_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = model_dir / Path(checkpoint_url).name
        if not checkpoint_file.exists():
            urllib.request.urlretrieve(checkpoint_url, str(checkpoint_file))

        return checkpoint_file

    def _checkpoint_url(self, model_name: str) -> str:
        """Resolve checkpoint URL by model name with environment overrides."""
        if model_name == "tacotron2":
            return os.getenv(
                "WAVEGLOW_TACOTRON2_CKPT_URL",
                "https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_fp32/versions/19.09.0/files/nvidia_tacotron2pyt_fp32_20190427",
            )

        if model_name == "waveglow":
            return os.getenv(
                "WAVEGLOW_VOCODER_CKPT_URL",
                "https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ckpt_fp32/versions/19.09.0/files/nvidia_waveglowpyt_fp32_20190427",
            )

        raise ValueError(f"Unsupported checkpoint model name: {model_name}")

    def _unwrap_distributed_state(self, state_dict: dict) -> dict:
        """Remove DistributedDataParallel prefixes when present."""
        unwrapped = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.1.", "")
            new_key = new_key.replace("module.", "")
            unwrapped[new_key] = value
        return unwrapped
