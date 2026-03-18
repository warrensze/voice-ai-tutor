import re

PAGE_RANGE_PATTERNS = [
    re.compile(
        r"\b(?:pages?|pp\.?)(?:\s+|\s*:\s*)(\d+)\s*(?:-|to|through)\s*(\d+)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bbetween\s+pages?\s+(\d+)\s+and\s+(\d+)\b", re.IGNORECASE),
]
SINGLE_PAGE_PATTERN = re.compile(r"\b(?:page|p\.?)\s*(\d+)\b", re.IGNORECASE)


def extract_page_range(question: str) -> tuple[int | None, int | None]:
    """Extract a requested page range from the user's question."""
    for pattern in PAGE_RANGE_PATTERNS:
        match = pattern.search(question)
        if match:
            start_page = int(match.group(1))
            end_page = int(match.group(2))
            if start_page <= end_page:
                return start_page, end_page
            return end_page, start_page

    match = SINGLE_PAGE_PATTERN.search(question)
    if match:
        page = int(match.group(1))
        return page, page

    return None, None


def describe_page_range(start_page: int | None, end_page: int | None) -> str:
    """Convert a page filter into a user-friendly description."""
    if start_page is None and end_page is None:
        return "No page filter requested"
    if start_page == end_page:
        return f"Page {start_page}"
    return f"Pages {start_page} to {end_page}"


def format_source(documents) -> str:
    """Render retrieved source passages into a prompt-ready block."""
    if not documents:
        return "No matching source passages were found for this request."

    sections = []
    for index, document in enumerate(documents, start=1):
        page_label = document.metadata.get("page_label") or str(
            document.metadata.get("page", "unknown")
        )
        sections.append(
            f"Source {index} | page {page_label}\n{document.page_content.strip()}"
        )

    return "\n\n".join(sections)


def truncate_text(text: str, max_chars: int) -> str:
    """Normalize whitespace and trim long text for compact memory prompts."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def format_conversation_memory(history, max_chars_per_message: int = 500) -> str:
    """Render recent conversation turns into a compact memory section."""
    if not history:
        return "No prior conversation yet."

    lines = []
    for idx, turn in enumerate(history, start=1):
        user_text = truncate_text(turn["user"], max_chars_per_message)
        assistant_text = truncate_text(turn["assistant"], max_chars_per_message)
        lines.append(f"Turn {idx} | Student: {user_text}")
        lines.append(f"Turn {idx} | Tutor: {assistant_text}")

    return "\n".join(lines)


def barge_in_passes_threshold(text: str, stop_words: set[str], min_chars: int) -> bool:
    """Decide whether a barge-in transcript is strong enough to interrupt output."""
    cleaned = text.strip()
    if not cleaned:
        return False

    if cleaned.lower() in stop_words:
        return True

    # Filter out utterances that are mostly punctuation or symbols (likely echo/noise)
    # Keep only alphanumeric characters and count them
    words = cleaned.split()
    if not words:
        return False

    # Count actual word characters vs punctuation
    alpha_count = sum(1 for c in cleaned if c.isalnum())
    total_count = len(cleaned)

    # If more than 60% punctuation/spaces, it's likely echo or noise
    if alpha_count == 0 or (total_count - alpha_count) / total_count > 0.6:
        return False

    return len(cleaned) >= min_chars
