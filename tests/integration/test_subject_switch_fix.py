#!/usr/bin/env python3
"""Test the fixed subject switching with explicit transition detection."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from router_agent import route_subject_sticky

# Test cases that should trigger explicit switching
test_cases = [
    # (question, current_subject, expected_subject, should_switch)
    ("I want to talk about AP World History", "english", "history", True),
    ("I want to talk about Chemistry", "english", "chemistry", True),
    ("Can we discuss Math?", "english", "math", True),
    ("Let's switch to History", "english", "history", True),
    ("I'd like to discuss Chemistry", "english", "chemistry", True),
    # Follow-up questions that should stay
    ("Can you explain that more?", "chemistry", "chemistry", False),
    ("What does that mean?", "history", "history", False),
    # Explicit switches from one subject to another
    ("Now let's talk about English literature", "chemistry", "english", True),
    ("Can we go back to Math?", "history", "math", True),
    # Short/direct switch phrases should still switch
    ("I want history", "english", "history", True),
    ("History", "english", "history", True),
    ("I want to study AP World", "english", "history", True),
]

print("Testing Subject Switching with Explicit Transition Detection\n")
print("=" * 70)

passed = 0
failed = 0

for question, current, expected, should_switch in test_cases:
    subject, is_switch = route_subject_sticky(question, current)

    subject_ok = subject == expected
    switch_ok = is_switch == should_switch
    status = "✓" if (subject_ok and switch_ok) else "✗"

    print(f"\n{status} Q: '{question}'")
    print(f"  Current: {current} | Expected: {expected} | Got: {subject}")
    print(f"  Should switch: {should_switch} | Got: {is_switch}")

    if subject_ok and switch_ok:
        passed += 1
    else:
        failed += 1

print("\n" + "=" * 70)
print(f"Results: {passed} passed, {failed} failed")

if failed == 0:
    print("✓ All tests passed!")
    sys.exit(0)
else:
    print("✗ Some tests failed")
    sys.exit(1)
