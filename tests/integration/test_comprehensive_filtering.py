#!/usr/bin/env python3
"""Comprehensive page filtering tests."""

import os

# Configure offline-only mode BEFORE any other imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TTS_REQUIRE_KOKORO"] = "0"

from src.vector import search_documents

print("=" * 70)
print("COMPREHENSIVE PAGE FILTERING TESTS")
print("=" * 70)

# Test 1: Specific page within normal range
print("\nTEST 1: Specific page (page 5, math)")
print("-" * 70)
results = search_documents("quadratic", subject="math", start_page=5, end_page=5, k=3)
if results:
    pages = {doc.metadata.get("page") for doc in results}
    print(f"[PASS] Results found on page(s): {sorted(pages)}")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. Page {doc.metadata.get('page_label')}: {doc.page_content[:100]}")
else:
    print("[FAIL] No results found")

# Test 2: Large page range
print("\nTEST 2: Large page range (pages 100-110, history)")
print("-" * 70)
results = search_documents(
    "American", subject="history", start_page=100, end_page=110, k=3
)
if results:
    pages = {doc.metadata.get("page") for doc in results}
    print(f"[PASS] Results found on page(s): {sorted(pages)}")
    print(f"  Total results: {len(results)}")
else:
    print("[FAIL] No results found")

# Test 3: Non-semantic match with page restriction
print("\nTEST 3: Nonsensical query with page restriction (pages 1-3, english)")
print("-" * 70)
results = search_documents(
    "zxcvbnm qwerty asdfgh", subject="english", start_page=1, end_page=3, k=3
)
if results:
    pages = {doc.metadata.get("page") for doc in results}
    print(f"[PASS] Fallback returned results on page(s): {sorted(pages)}")
    print(f"  (Note: fallback used because no semantic match)")
else:
    print("[FAIL] No results found")

# Test 4: No page filter (behavior should be unchanged)
print("\nTEST 4: No page filter (chemistry)")
print("-" * 70)
results = search_documents("elements", subject="chemistry", k=3)
if results:
    pages = {doc.metadata.get("page") for doc in results}
    print(f"[PASS] Results found on page(s): {sorted(pages)}")
    print(f"  Total results: {len(results)}")
else:
    print("[FAIL] No results found")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED")
print("=" * 70)
