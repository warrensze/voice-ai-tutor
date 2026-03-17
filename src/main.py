try:
    import msvcrt
except ImportError:  # pragma: no cover - msvcrt is Windows-only
    msvcrt = None

from conversation_utils import (
    describe_page_range as _describe_page_range,
    extract_page_range as _extract_page_range,
    format_conversation_memory as _format_conversation_memory,
    truncate_text as _truncate_text,
)
from voice_agent import (
    MEMORY_MAX_CHARS_PER_MESSAGE,
    VoiceAgent,
    barge_in_passes_threshold as _barge_in_passes_threshold,
)


def describe_page_range(start_page: int | None, end_page: int | None) -> str:
    """Expose page range descriptions for app output and tests."""
    return _describe_page_range(start_page, end_page)


def extract_page_range(question: str) -> tuple[int | None, int | None]:
    """Expose page range extraction helper for app usage and tests."""
    return _extract_page_range(question)


def truncate_text(text: str, max_chars: int) -> str:
    """Expose text truncation helper for app use and tests."""
    return _truncate_text(text, max_chars)


def format_conversation_memory(history) -> str:
    """Expose memory formatting with project-level truncation settings."""
    return _format_conversation_memory(history, MEMORY_MAX_CHARS_PER_MESSAGE)


def keyboard_quit_requested() -> bool:
    """Check for a non-blocking q key press in the terminal."""
    if msvcrt is None:
        return False

    try:
        return msvcrt.kbhit() and msvcrt.getwch().lower() == "q"
    except Exception:
        return False


def barge_in_passes_threshold(text: str) -> bool:
    """Apply interruption threshold rules for duplex barge-in transcripts."""
    return _barge_in_passes_threshold(text)


def main():
    """Start the voice tutor application."""
    agent = VoiceAgent()
    agent.run()


if __name__ == "__main__":
    main()
