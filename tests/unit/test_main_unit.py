import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_main_with_stubs():
    if "main" in sys.modules:
        del sys.modules["main"]

    stt_stub = types.ModuleType("stt_module")

    class SpeechToText:
        def __init__(self):
            pass

        def listen(self):
            return ""

    stt_stub.SpeechToText = SpeechToText

    tts_stub = types.ModuleType("tts_module")

    class TextToSpeech:
        def __init__(self):
            self.spoken = []

        def speak(self, text):
            self.spoken.append(text)

    tts_stub.TextToSpeech = TextToSpeech

    vector_stub = types.ModuleType("vector")
    vector_stub.search_documents = lambda *args, **kwargs: []

    ollama_stub = types.ModuleType("langchain_ollama")

    class ChatOllama:
        def __init__(self, *args, **kwargs):
            pass

        def stream(self, _prompt):
            return iter([])

    ollama_stub.ChatOllama = ChatOllama

    with patch.dict(
        sys.modules,
        {
            "stt_module": stt_stub,
            "tts_module": tts_stub,
            "vector": vector_stub,
            "langchain_ollama": ollama_stub,
        },
    ):
        return importlib.import_module("main")


class TestMainHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = load_main_with_stubs()

    def test_extract_page_range_for_range_expression(self):
        start_page, end_page = self.main.extract_page_range("Summarize pages 10-12")
        self.assertEqual((start_page, end_page), (10, 12))

    def test_extract_page_range_for_single_page(self):
        start_page, end_page = self.main.extract_page_range("What is on page 7?")
        self.assertEqual((start_page, end_page), (7, 7))

    def test_extract_page_range_for_reversed_range(self):
        start_page, end_page = self.main.extract_page_range("between pages 12 and 10")
        self.assertEqual((start_page, end_page), (10, 12))

    def test_describe_page_range(self):
        self.assertEqual(
            self.main.describe_page_range(None, None), "No page filter requested"
        )
        self.assertEqual(self.main.describe_page_range(3, 3), "Page 3")
        self.assertEqual(self.main.describe_page_range(3, 5), "Pages 3 to 5")

    def test_barge_in_threshold_allows_stop_words(self):
        self.assertTrue(self.main.barge_in_passes_threshold("stop"))

    def test_barge_in_threshold_rejects_short_noise(self):
        self.assertFalse(self.main.barge_in_passes_threshold("ok"))

    def test_barge_in_threshold_accepts_longer_phrase(self):
        self.assertTrue(self.main.barge_in_passes_threshold("hold on now"))

    def test_truncate_text_short_input(self):
        self.assertEqual(self.main.truncate_text("hello", 20), "hello")

    def test_truncate_text_long_input(self):
        self.assertEqual(self.main.truncate_text("abcdefghij", 6), "abc...")

    def test_format_conversation_memory_empty(self):
        self.assertEqual(
            self.main.format_conversation_memory([]), "No prior conversation yet."
        )

    def test_format_conversation_memory_has_turns(self):
        history = [
            {
                "user": "What is photosynthesis?",
                "assistant": "It is how plants convert light into chemical energy.",
            }
        ]
        rendered = self.main.format_conversation_memory(history)
        self.assertIn("Turn 1 | Student: What is photosynthesis?", rendered)
        self.assertIn(
            "Turn 1 | Tutor: It is how plants convert light into chemical energy.",
            rendered,
        )


class TestQuitBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = load_main_with_stubs()

    def test_keyboard_quit_requested_returns_false_without_msvcrt(self):
        with patch.object(self.main, "msvcrt", None):
            self.assertFalse(self.main.keyboard_quit_requested())

    def test_keyboard_quit_requested_returns_true_for_q(self):
        fake_msvcrt = types.SimpleNamespace(kbhit=lambda: True, getwch=lambda: "q")
        with patch.object(self.main, "msvcrt", fake_msvcrt):
            self.assertTrue(self.main.keyboard_quit_requested())

    def test_voice_agent_exits_when_user_says_stop(self):
        agent = self.main.VoiceAgent()
        agent.ears = types.SimpleNamespace(listen=lambda: "stop")

        spoken = []

        class Mouth:
            def speak(self, text):
                spoken.append(text)

        agent.mouth = Mouth()

        with patch.object(self.main, "keyboard_quit_requested", return_value=False):
            agent.run()

        self.assertTrue(spoken)
        self.assertIn("Goodbye! Keep studying!", spoken)


if __name__ == "__main__":
    unittest.main()
