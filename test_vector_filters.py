#!/usr/bin/env python3
"""Test script to verify vector search with page filters."""

from src.vector import search_documents

print("\n" + "=" * 60)
print("TEST 1: Search with page filter (pages 100-120)")
print("=" * 60)
results = search_documents(
    "photosynthesis", subject="english", start_page=100, end_page=120, k=5
)
print(f"\nExpected: All results should have page_label between 100-120")
for i, doc in enumerate(results, 1):
    page_label = int(doc.metadata.get("page_label", 0))
    page_num = doc.metadata.get("page", "unknown")
    print(f"  Result {i}: page_label={page_label}, page={page_num}")

print("\n" + "=" * 60)
print("TEST 2: Search with ALL pages (no page filter)")
print("=" * 60)
results = search_documents("learning", subject="english", k=5)
print(f"\nExpected: Results can be from any page")
for i, doc in enumerate(results, 1):
    page_label = doc.metadata.get("page_label", "unknown")
    print(f"  Result {i}: page_label={page_label}")
