#!/usr/bin/env python3
"""Test Kokoro availability."""

try:
    from kokoro import KPipeline

    print("✓ Kokoro KPipeline is available")
    print(f"  Location: {KPipeline.__module__}")
except ImportError as e:
    print(f"✗ Kokoro not available: {e}")
    print("\nTo install Kokoro-82M, run:")
    print("  pip install kokoro-onnx")
    print("  OR")
    print("  pip install 'kokoro[onnx]'")

try:
    import sounddevice as sd

    print("✓ Sounddevice is available")
except ImportError as e:
    print(f"✗ Sounddevice not available: {e}")

try:
    import torch

    print(f"✓ PyTorch is available ({torch.__version__})")
    if torch.cuda.is_available():
        print(f"  CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("  CUDA is not available")
except ImportError as e:
    print(f"✗ PyTorch not available: {e}")
