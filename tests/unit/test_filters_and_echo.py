"""Unit tests for vector filtering, echo detection, and TTS playback state.

These tests verify that:
1. Vector database filters (subject and page) are properly constructed
2. Echo detection filters out speaker output during listening
3. TTS playback state is properly tracked
4. Combined filters work correctly
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class TestVectorFilters(unittest.TestCase):
    """Test vector database filter construction and application."""

    def setUp(self):
        """Import vector module for testing."""
        if "vector" in sys.modules:
            del sys.modules["vector"]
        import vector

        self.vector = vector

    def test_build_subject_filter_with_valid_subject(self):
        """Test that subject filter is properly constructed."""
        result = self.vector.build_subject_filter("history")
        self.assertEqual(result, {"subject": {"$eq": "history"}})

    def test_build_subject_filter_case_insensitive(self):
        """Test that subject filter lowercases input."""
        result = self.vector.build_subject_filter("HISTORY")
        self.assertEqual(result, {"subject": {"$eq": "history"}})

    def test_build_subject_filter_with_none(self):
        """Test that None subject returns None."""
        result = self.vector.build_subject_filter(None)
        self.assertIsNone(result)

    def test_build_subject_filter_with_empty_string(self):
        """Test that empty string subject returns None."""
        result = self.vector.build_subject_filter("")
        self.assertIsNone(result)

    def test_build_page_filter_with_range(self):
        """Test that page filter is properly constructed for a range."""
        result = self.vector.build_page_filter(5, 10)
        self.assertEqual(
            result,
            {
                "$and": [
                    {"page": {"$gte": 4}},  # 5-1 = 4 (0-based)
                    {"page": {"$lte": 9}},  # 10-1 = 9 (0-based)
                ]
            },
        )

    def test_build_page_filter_single_page(self):
        """Test that page filter works for a single page."""
        result = self.vector.build_page_filter(7, 7)
        self.assertEqual(
            result,
            {
                "$and": [
                    {"page": {"$gte": 6}},  # 7-1 = 6
                    {"page": {"$lte": 6}},  # 7-1 = 6
                ]
            },
        )

    def test_build_page_filter_with_none(self):
        """Test that None pages return None."""
        result = self.vector.build_page_filter(None, None)
        self.assertIsNone(result)

    def test_build_page_filter_validates_page_numbers(self):
        """Test that page filter validates page numbers are positive."""
        with self.assertRaises(ValueError):
            self.vector.build_page_filter(0, 5)

    def test_build_page_filter_validates_range_order(self):
        """Test that page filter validates start <= end."""
        with self.assertRaises(ValueError):
            self.vector.build_page_filter(10, 5)

    def test_combine_filters_both_present(self):
        """Test that both filters are combined correctly."""
        page_filter = self.vector.build_page_filter(1, 5)
        subject_filter = self.vector.build_subject_filter("history")
        result = self.vector.combine_filters(page_filter, subject_filter)

        self.assertIn("$and", result)
        self.assertEqual(len(result["$and"]), 3)  # 2 page conditions + 1 subject

    def test_combine_filters_only_page(self):
        """Test that only page filter is returned when subject is None."""
        page_filter = self.vector.build_page_filter(1, 5)
        result = self.vector.combine_filters(page_filter, None)
        self.assertEqual(result, page_filter)

    def test_combine_filters_only_subject(self):
        """Test that only subject filter is returned when page is None."""
        subject_filter = self.vector.build_subject_filter("chemistry")
        result = self.vector.combine_filters(None, subject_filter)
        self.assertEqual(result, subject_filter)

    def test_combine_filters_both_none(self):
        """Test that None is returned when both filters are None."""
        result = self.vector.combine_filters(None, None)
        self.assertIsNone(result)


class TestEchoDetection(unittest.TestCase):
    """Test echo detection for filtering speaker output."""

    def setUp(self):
        """Import STT module for testing."""
        if "stt_module" in sys.modules:
            del sys.modules["stt_module"]
        import stt_module

        self.stt = stt_module

    def test_echo_detection_method_exists(self):
        """Test that echo detection method is available."""
        self.assertTrue(hasattr(self.stt.SpeechToText, "_is_speaker_echo"))

    def test_echo_detection_threshold(self):
        """Test that high RMS values are detected as echo."""
        stt = self.stt.SpeechToText()
        # RMS > 0.15 should be detected as echo
        result = stt._is_speaker_echo(0.2)
        self.assertTrue(result)

    def test_normal_speech_not_detected_as_echo(self):
        """Test that normal speech RMS levels are not detected as echo."""
        stt = self.stt.SpeechToText()
        # RMS < 0.15 should not be detected as echo
        result = stt._is_speaker_echo(0.05)
        self.assertFalse(result)

    def test_echo_threshold_boundary(self):
        """Test the echo detection threshold boundary."""
        stt = self.stt.SpeechToText()
        # At exactly the threshold (0.15) should NOT be detected as echo (uses >)
        result_at_threshold = stt._is_speaker_echo(0.15)
        self.assertFalse(result_at_threshold)
        # Just above threshold should be detected
        result_above_threshold = stt._is_speaker_echo(0.151)
        self.assertTrue(result_above_threshold)


class TestTTSPlaybackState(unittest.TestCase):
    """Test TTS playback state tracking."""

    def setUp(self):
        """Import TTS module for testing."""
        if "tts_module" in sys.modules:
            del sys.modules["tts_module"]
        self.tts_module = __import__("tts_module")

    def test_is_audio_playing_method_exists(self):
        """Test that is_audio_playing method is available."""
        tts = self.tts_module.TextToSpeech()
        self.assertTrue(hasattr(tts, "is_audio_playing"))

    def test_playback_state_initially_not_playing(self):
        """Test that playback state is initially not playing."""
        tts = self.tts_module.TextToSpeech()
        self.assertFalse(tts.is_audio_playing())

    def test_playback_state_flag_exists(self):
        """Test that _is_playing flag exists in TTS."""
        tts = self.tts_module.TextToSpeech()
        self.assertTrue(hasattr(tts, "_is_playing"))


class TestSTTInitialization(unittest.TestCase):
    """Test STT module initialization with TTS reference."""

    def setUp(self):
        """Import STT module for testing."""
        if "stt_module" in sys.modules:
            del sys.modules["stt_module"]
        import stt_module

        self.stt = stt_module

    def test_stt_accepts_tts_instance(self):
        """Test that STT accepts optional TTS instance."""
        tts_mock = MagicMock()
        # Should not raise an error
        stt = self.stt.SpeechToText(tts_instance=tts_mock)
        self.assertEqual(stt.tts_instance, tts_mock)

    def test_stt_works_without_tts_instance(self):
        """Test that STT works without TTS instance."""
        stt = self.stt.SpeechToText()
        self.assertIsNone(stt.tts_instance)

    def test_stt_stores_tts_reference(self):
        """Test that STT stores TTS reference for later use."""
        tts_mock = MagicMock()
        stt = self.stt.SpeechToText(tts_instance=tts_mock)
        self.assertIsNotNone(stt.tts_instance)
        self.assertTrue(hasattr(stt.tts_instance, "is_audio_playing"))


class TestPageMetadataHandling(unittest.TestCase):
    """Test page metadata storage and conversion."""

    def setUp(self):
        """Import vector module for testing."""
        if "vector" in sys.modules:
            del sys.modules["vector"]
        import vector

        self.vector = vector

    def test_page_filter_one_based_to_zero_based_conversion(self):
        """Test that 1-based page numbers are converted to 0-based."""
        # User requests pages 1-10
        result = self.vector.build_page_filter(1, 10)
        # Should be stored as 0-9 (0-based)
        self.assertEqual(result["$and"][0]["page"]["$gte"], 0)
        self.assertEqual(result["$and"][1]["page"]["$lte"], 9)

    def test_page_filter_high_pages(self):
        """Test page filter with high page numbers."""
        # User requests pages 100-105
        result = self.vector.build_page_filter(100, 105)
        # Should be stored as 99-104 (0-based)
        self.assertEqual(result["$and"][0]["page"]["$gte"], 99)
        self.assertEqual(result["$and"][1]["page"]["$lte"], 104)


if __name__ == "__main__":
    unittest.main()
