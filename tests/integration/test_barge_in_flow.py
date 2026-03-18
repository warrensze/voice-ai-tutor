#!/usr/bin/env python3
"""Test subject switching and barge-in simulation."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from router_agent import route_subject_sticky
from conversation_utils import barge_in_passes_threshold

# Test 1: Verify subject switching with explicit phrases
print("=" * 70)
print("TEST 1: Subject Switching with Explicit Transition Detection")
print("=" * 70)

explicit_switch_cases = [
    "I want to talk about AP World History",
    "Can we discuss Chemistry?",
    "Let's switch to Math",
    "I'd like to discuss English literature",
]

current = "english"
for question in explicit_switch_cases:
    subject, is_switch = route_subject_sticky(question, current)
    status = "✓" if (is_switch and subject != current) else "✗"
    print(f"{status} '{question}' → {subject} (switch={is_switch})")
    current = subject

# Test 2: Verify barge-in threshold detection
print("\n" + "=" * 70)
print("TEST 2: Barge-in Threshold Detection")
print("=" * 70)

VOICE_STOP_WORDS = {"quit", "stop", "exit", "bye"}
BARGE_IN_MIN_CHARS = 3

barge_in_cases = [
    ("I want to talk about Chemistry", True),
    ("Chemistry", True),
    ("q", False),  # Would match "quit" only if longer
    ("", False),
    ("hmm", True),  # 3 chars, passes
    ("Stop", True),  # In stop words
]

for text, should_pass in barge_in_cases:
    result = barge_in_passes_threshold(text, VOICE_STOP_WORDS, BARGE_IN_MIN_CHARS)
    status = "✓" if result == should_pass else "✗"
    print(f"{status} '{text}' → {result} (expected {should_pass})")

# Test 3: Simulate conversation flow with interruptions
print("\n" + "=" * 70)
print("TEST 3: Simulated Conversation Flow with Interruptions")
print("=" * 70)

conversation_flow = [
    (
        "What's 2+2?",
        "english",
        True,
        "english",
    ),  # No explicit subject signal, should remain in current subject
    (
        "I want to talk about Chemistry",
        "english",
        True,
        "chemistry",
    ),  # Should interrupt and switch
    (
        "Can you explain that more?",
        "chemistry",
        True,
        "chemistry",
    ),  # Follow-up, stays and still passes interrupt threshold
    ("Let's go back to History", "chemistry", True, "history"),  # Interrupt + switch
]

current_subject = "english"
for i, (utterance, _, should_be_barge, expected_subject) in enumerate(
    conversation_flow, 1
):
    # Check barge-in potential
    can_barge = barge_in_passes_threshold(
        utterance, VOICE_STOP_WORDS, BARGE_IN_MIN_CHARS
    )

    # Check subject routing
    subject, is_switch = route_subject_sticky(utterance, current_subject)

    barge_status = "✓" if can_barge == should_be_barge else "✗"
    subject_status = "✓" if subject == expected_subject else "✗"

    print(f"\nTurn {i}: '{utterance[:40]}...'")
    print(f"  {barge_status} Can interrupt: {can_barge}")
    print(f"  {subject_status} Subject: {subject} (expected {expected_subject})")

    current_subject = subject

print("\n" + "=" * 70)
print("✓ Tests completed")
print("=" * 70)
