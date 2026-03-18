try:
    import msvcrt
except ImportError:  # pragma: no cover - msvcrt is Windows-only
    msvcrt = None

import logging

# Configure offline-only mode to prevent HuggingFace Hub requests
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# Suppress HF Hub warnings since we're offline-only
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from crash_logger import setup_crash_logging

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

LOG_DIR = setup_crash_logging()
LOGGER = logging.getLogger("voice_ai_tutor")
print(f"[Logging] Crash diagnostics enabled. Logs directory: {LOG_DIR}")


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
    LOGGER.info("Application startup requested")
    try:
        agent = VoiceAgent()
        LOGGER.info("VoiceAgent initialized")
        agent.run()
        LOGGER.info("VoiceAgent exited normally")
    except Exception:
        LOGGER.exception("Unhandled exception in main application loop")
        raise


if __name__ == "__main__":
    main()
