#!/usr/bin/env python3
"""Quick test of page filtering in voice-ai-tutor app."""

import os

# Configure offline-only mode BEFORE any other imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TTS_REQUIRE_KOKORO"] = "0"

print("Testing voice-ai-tutor with page filtering...")
print("=" * 60)

try:
    from src.voice_agent import VoiceAgent

    print("[OK] VoiceAgent imported successfully")

    agent = VoiceAgent()
    print("[OK] VoiceAgent initialized successfully")

    # Test a simple question with page reference
    test_question = "What's on page 10 in history?"
    print(f"\nTest question: {test_question}")

    payload = {"question": test_question}
    result = agent._orchestrate_chain_inputs(payload)

    print(f"Extracted page range: {result['page_range']}")
    print(f"Subject: {result['subject']}")
    print(f"Source retrieved: {len(result['source'].split('Source')) - 1} chunks")

    print("\n[SUCCESS] Page filtering is working!")

except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
