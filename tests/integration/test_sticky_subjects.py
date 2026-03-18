#!/usr/bin/env python3
"""Test sticky subject routing."""

import os

# Configure offline-only mode BEFORE any other imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from src.router_agent import route_subject_sticky

print("=" * 70)
print("TESTING STICKY SUBJECT ROUTING")
print("=" * 70)

# Scenario: User is in chemistry, asks weakly-signaled follow-up questions
print("\nScenario 1: Chemistry context with weak follow-ups")
print("-" * 70)

current = "chemistry"
test_questions = [
    "Can you explain that again?",
    "What does that mean?",
    "Tell me more about it",
    "I don't understand",
    "Can you give an example?",
]

for q in test_questions:
    subject, is_switch = route_subject_sticky(q, current)
    status = "[SWITCH]" if is_switch else "[STICK]"
    print(f"{status} '{q}'")
    print(f"         -> {subject}")
    current = subject


# Scenario: User is in chemistry, then explicitly asks about history
print("\nScenario 2: Explicit subject switch from chemistry to history")
print("-" * 70)

current = "chemistry"
switch_questions = [
    "Tell me about World War II",
    "What is the French Revolution?",
    "Explain ancient Rome",
]

for q in switch_questions:
    subject, is_switch = route_subject_sticky(q, current)
    status = "[SWITCH]" if is_switch else "[STICK]"
    print(f"{status} '{q}'")
    print(f"         From: {current} -> To: {subject}")
    current = subject


# Scenario: User in math, various follow-ups
print("\nScenario 3: Math context with various follow-ups")
print("-" * 70)

current = "math"
mixed_questions = [
    "Can you solve that step by step?",
    "What about the next problem?",
    "Why is that the answer?",
    "What is photosynthesis?",  # Switch to english/chemistry
    "But what about the quadratic formula?",  # Should stick to english first, then might switch
]

for q in mixed_questions:
    subject, is_switch = route_subject_sticky(q, current)
    status = "[SWITCH]" if is_switch else "[STICK]"
    print(f"{status} '{q}'")
    print(f"         From: {current} -> To: {subject}")
    current = subject


# Scenario: Starting without context (None)
print("\nScenario 4: Starting without prior subject")
print("-" * 70)

questions_no_context = [
    "What's an element?",  # Should go to chemistry
    "Explain that more",  # Should stick
    "How does photosynthesis work?",  # Should stick (chemistry-ish)
]

current = None
for q in questions_no_context:
    subject, is_switch = route_subject_sticky(q, current)
    status = "[SWITCH]" if is_switch else "[START]"
    print(f"{status} '{q}'")
    print(f"         Current: {current} -> Result: {subject}")
    current = subject

print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)
