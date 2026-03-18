"""Unit tests for voice agent initialization and vector search integration."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class TestVoiceAgentInitialization(unittest.TestCase):
    """Test voice agent proper initialization with STT/TTS integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock heavy dependencies
        self.patches = [
            patch("langchain_ollama.ChatOllama"),
            patch("voice_agent.build_history_chain"),
            patch("voice_agent.build_chemistry_chain"),
            patch("voice_agent.build_math_chain"),
            patch("voice_agent.build_english_chain"),
            patch("voice_agent.load_subject_voice_map"),
            patch("voice_agent.search_documents"),
        ]

        for p in self.patches:
            p.start()

        if "voice_agent" in sys.modules:
            del sys.modules["voice_agent"]

    def tearDown(self):
        """Stop all patches."""
        for p in self.patches:
            p.stop()

    def test_voice_agent_initialization_order(self):
        """Test that TTS is initialized before STT (for reference)."""
        # Import after patches are set up
        import voice_agent

        # Create agent
        agent = voice_agent.VoiceAgent()

        # Verify both are initialized
        self.assertIsNotNone(agent.mouth)
        self.assertIsNotNone(agent.ears)

        # Verify STT has reference to TTS
        self.assertEqual(agent.ears.tts_instance, agent.mouth)


class TestVectorSearchIntegration(unittest.TestCase):
    """Test vector search with filters."""

    def setUp(self):
        """Set up test fixtures."""
        if "vector" in sys.modules:
            del sys.modules["vector"]


class TestRouterAgent(unittest.TestCase):
    """Test subject routing based on keywords."""

    def setUp(self):
        """Import router module for testing."""
        if "router_agent" in sys.modules:
            del sys.modules["router_agent"]
        import router_agent

        self.router = router_agent

    def test_route_history_question(self):
        """Test that history questions are routed correctly."""
        result = self.router.route_subject("Tell me about the American Civil War")
        self.assertEqual(result, "history")

    def test_route_chemistry_question(self):
        """Test that chemistry questions are routed correctly."""
        result = self.router.route_subject("Explain ionic bonding and electrons")
        self.assertEqual(result, "chemistry")

    def test_route_math_question(self):
        """Test that math questions are routed correctly."""
        result = self.router.route_subject("How do I solve a quadratic equation?")
        self.assertEqual(result, "math")

    def test_route_english_question(self):
        """Test that English questions are routed correctly."""
        result = self.router.route_subject("What is the theme of this novel?")
        self.assertEqual(result, "english")

    def test_route_empty_question(self):
        """Test that empty question defaults to English."""
        result = self.router.route_subject("")
        self.assertEqual(result, "english")

    def test_route_ambiguous_question(self):
        """Test that ambiguous question defaults to English."""
        result = self.router.route_subject("What is this?")
        self.assertEqual(result, "english")


class TestConversationUtils(unittest.TestCase):
    """Test conversation utility functions."""

    def setUp(self):
        """Import conversation utils for testing."""
        if "conversation_utils" in sys.modules:
            del sys.modules["conversation_utils"]
        import conversation_utils

        self.utils = conversation_utils

    def test_extract_page_range_from_question(self):
        """Test extracting page range from question."""
        start, end = self.utils.extract_page_range("What happened on pages 10 to 15?")
        self.assertEqual((start, end), (10, 15))

    def test_extract_single_page(self):
        """Test extracting single page reference."""
        start, end = self.utils.extract_page_range("What is on page 42?")
        self.assertEqual((start, end), (42, 42))

    def test_extract_no_page_reference(self):
        """Test question without page reference."""
        start, end = self.utils.extract_page_range("Tell me about history")
        self.assertEqual((start, end), (None, None))

    def test_describe_page_range_full(self):
        """Test page range description."""
        desc = self.utils.describe_page_range(5, 10)
        self.assertEqual(desc, "Pages 5 to 10")

    def test_describe_page_range_single(self):
        """Test single page description."""
        desc = self.utils.describe_page_range(7, 7)
        self.assertEqual(desc, "Page 7")

    def test_describe_page_range_none(self):
        """Test description with no page filter."""
        desc = self.utils.describe_page_range(None, None)
        self.assertEqual(desc, "No page filter requested")

    def test_format_source_empty(self):
        """Test formatting empty source list."""
        result = self.utils.format_source([])
        self.assertIn("No matching source", result)

    def test_truncate_text_short(self):
        """Test truncating short text."""
        result = self.utils.truncate_text("hello", 20)
        self.assertEqual(result, "hello")

    def test_truncate_text_long(self):
        """Test truncating long text."""
        long_text = "a" * 100
        result = self.utils.truncate_text(long_text, 20)
        self.assertEqual(len(result), 20)
        self.assertTrue(result.endswith("..."))

    def test_barge_in_threshold_stop_word(self):
        """Test that stop words pass threshold."""
        result = self.utils.barge_in_passes_threshold(
            "stop", {"stop", "quit", "exit"}, 3
        )
        self.assertTrue(result)

    def test_barge_in_threshold_long_text(self):
        """Test that long text passes threshold."""
        result = self.utils.barge_in_passes_threshold("hello there friend", set(), 3)
        self.assertTrue(result)

    def test_barge_in_threshold_short_text(self):
        """Test that short text fails threshold."""
        result = self.utils.barge_in_passes_threshold("ok", set(), 3)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
