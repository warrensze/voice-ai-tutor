#!/usr/bin/env python3
"""Test page filtering end-to-end."""

import os

# Configure offline-only mode BEFORE any other imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TTS_REQUIRE_KOKORO"] = "0"

from src.voice_agent import VoiceAgent

# Test question asking about specific pages
test_question = "What is on page 5 in the math subject?"
print(f"Testing with question: {test_question}")

agent = VoiceAgent()
payload = {"question": test_question, "subject": "math"}

# Call the orchestrator to see what source it retrieves
result = agent._orchestrate_chain_inputs(payload)

print(f"\nExtracted page range from question: {result['page_range']}")
print(f"Subject: {result['subject']}")
print(f"\nSource material retrieved:")
if result["source"]:
    print(result["source"][:500])
else:
    print("No source found")

print("\n" + "=" * 60)
print("TEST 2: Question requesting a page range")
print("=" * 60)

test_question2 = "Explain pages 100-105 from history"
print(f"Testing with question: {test_question2}")
payload2 = {"question": test_question2}

result2 = agent._orchestrate_chain_inputs(payload2)
print(f"\nExtracted page range from question: {result2['page_range']}")
print(f"Subject: {result2['subject']}")
print(f"Source material retrieved (first 500 chars):")
if result2["source"]:
    print(result2["source"][:500])
else:
    print("No source found")
