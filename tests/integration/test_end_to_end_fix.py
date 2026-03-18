#!/usr/bin/env python3
"""End-to-end test simulating the reported issues and verifying fixes."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from router_agent import route_subject_sticky
from conversation_utils import barge_in_passes_threshold

print("=" * 80)
print("END-TO-END TEST: Simulating User's Reported Issues")
print("=" * 80)

VOICE_STOP_WORDS = {"quit", "stop", "exit", "bye"}
BARGE_IN_MIN_CHARS = 3

# Simulate the exact user conversation from the bug report
print(
    "\n[USER REPORT 1] 'I said - I want to talk about AP World History. It stays on English.'"
)
print("-" * 80)

current_subject = "english"
user_input = "I want to talk about AP World History"

subject, is_switch = route_subject_sticky(user_input, current_subject)
can_interrupt = barge_in_passes_threshold(
    user_input, VOICE_STOP_WORDS, BARGE_IN_MIN_CHARS
)

print(f"User input: '{user_input}'")
print(f"Current subject: {current_subject}")
print(f"Routed to subject: {subject}")
print(f"Is explicit switch: {is_switch}")
print(f"Can be used as interruption: {can_interrupt}")

if subject != "english" and is_switch:
    print("✓ FIXED: Subject properly switches to History")
else:
    print("✗ NOT FIXED: Subject should switch to History")

# Simulate second issue
print(
    "\n[USER REPORT 2] 'I then asked I want to talk about Chemistry. It stayed on English.'"
)
print("-" * 80)

current_subject = subject  # Keep the switched subject from previous
user_input = "I want to talk about Chemistry"

subject, is_switch = route_subject_sticky(user_input, current_subject)
can_interrupt = barge_in_passes_threshold(
    user_input, VOICE_STOP_WORDS, BARGE_IN_MIN_CHARS
)

print(f"User input: '{user_input}'")
print(f"Current subject: {current_subject}")
print(f"Routed to subject: {subject}")
print(f"Is explicit switch: {is_switch}")
print(f"Can be used as interruption: {can_interrupt}")

if subject == "chemistry" and is_switch:
    print("✓ FIXED: Subject properly switches to Chemistry")
else:
    print("✗ NOT FIXED: Subject should switch to Chemistry")

# Simulate interruption scenario
print("\n[USER REPORT 3] 'it is not allowing me to interrupt to change the question'")
print("-" * 80)

print("Simulating interruption scenario:")
print("  - Agent speaking on Chemistry (mode: english → history → chemistry)")
print("  - User interrupts with: 'I want to talk about Math'")

current_subject = "chemistry"
interruption = "I want to talk about Math"

can_interrupt = barge_in_passes_threshold(
    interruption, VOICE_STOP_WORDS, BARGE_IN_MIN_CHARS
)
subject, is_switch = route_subject_sticky(interruption, current_subject)

print(f"\nInterruption text: '{interruption}'")
print(f"Can interrupt: {can_interrupt}")
print(f"Would route to: {subject}")
print(f"Is explicit switch: {is_switch}")

if can_interrupt and subject == "math" and is_switch:
    print("✓ FIXED: Interruption can switch subjects properly")
else:
    print("✗ NOT FIXED: Interruption should allow subject switching")

# Additional test: verify sticky behavior still works for follow-ups
print("\n[VERIFICATION] Sticky behavior for follow-up questions")
print("-" * 80)

current_subject = "chemistry"
follow_up = "Can you explain that more?"

subject, is_switch = route_subject_sticky(follow_up, current_subject)

print(f"Follow-up question: '{follow_up}'")
print(f"Current subject: {current_subject}")
print(f"Routed to subject: {subject}")
print(f"Is switch: {is_switch}")

if subject == current_subject and not is_switch:
    print("✓ CORRECT: Follow-up maintains subject context")
else:
    print("✗ WRONG: Follow-up should maintain context")

print("\n" + "=" * 80)
print("END-TO-END TEST COMPLETE")
print("=" * 80)
