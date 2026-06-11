"""Integration-style tests for VoiceAgent persistence behavior."""

import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from persistence import TutorPersistence


class _FakeChain:
    def stream(self, _payload):
        return iter([])


def _build_stub_modules():
    stubs = {}

    ollama_stub = types.ModuleType("langchain_ollama")

    class ChatOllama:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    ollama_stub.ChatOllama = ChatOllama
    stubs["langchain_ollama"] = ollama_stub

    chat_history_stub = types.ModuleType("langchain_core.chat_history")

    class InMemoryChatMessageHistory:
        def __init__(self):
            self.messages = []

        def add_message(self, message):
            self.messages.append(message)

    chat_history_stub.InMemoryChatMessageHistory = InMemoryChatMessageHistory
    stubs["langchain_core.chat_history"] = chat_history_stub

    messages_stub = types.ModuleType("langchain_core.messages")

    class HumanMessage:
        def __init__(self, content):
            self.content = content

    class AIMessage:
        def __init__(self, content):
            self.content = content

    messages_stub.HumanMessage = HumanMessage
    messages_stub.AIMessage = AIMessage
    stubs["langchain_core.messages"] = messages_stub

    runnables_stub = types.ModuleType("langchain_core.runnables")

    class RunnableLambda:
        def __init__(self, fn):
            self._fn = fn

        def invoke(self, payload):
            return self._fn(payload)

    runnables_stub.RunnableLambda = RunnableLambda
    stubs["langchain_core.runnables"] = runnables_stub

    for module_name, builder_name in (
        ("history_agent", "build_history_chain"),
        ("chemistry_agent", "build_chemistry_chain"),
        ("math_agent", "build_math_chain"),
        ("english_agent", "build_english_chain"),
    ):
        module = types.ModuleType(module_name)
        setattr(module, builder_name, lambda _llm: _FakeChain())
        stubs[module_name] = module

    router_stub = types.ModuleType("router_agent")
    router_stub.route_subject = lambda _question: "english"
    router_stub.route_subject_sticky = lambda _question, current: (current, False)
    stubs["router_agent"] = router_stub

    utils_stub = types.ModuleType("conversation_utils")
    utils_stub.barge_in_passes_threshold = lambda text, stop_words, min_chars: (
        len(text.strip()) >= min_chars or text.strip().lower() in stop_words
    )
    utils_stub.describe_page_range = lambda start, end: "No page filter requested"
    utils_stub.extract_page_range = lambda _question: (None, None)
    utils_stub.format_source = lambda _docs: ""
    utils_stub.truncate_text = lambda text, max_len: text[:max_len]
    stubs["conversation_utils"] = utils_stub

    stt_stub = types.ModuleType("stt_module")

    class SpeechToText:
        def __init__(self, tts_instance=None):
            self.tts_instance = tts_instance

        def listen(self, *args, **kwargs):
            return ""

    stt_stub.SpeechToText = SpeechToText
    stubs["stt_module"] = stt_stub

    tts_stub = types.ModuleType("tts_module")

    class TextToSpeech:
        def __init__(self, *args, **kwargs):
            self.voice = ""
            self.backend = "pyttsx3"
            self.backend_error = ""

        def set_voice(self, voice):
            self.voice = voice

        def speak_async(self, text):
            return text

        def has_pending_audio(self):
            return False

        def stop(self, wait=False, release_owner=True):
            return wait

        def speak(self, text):
            return text

        def is_available(self):
            return True

        def is_audio_playing(self):
            return False

        def recently_played(self, _seconds):
            return False

    tts_stub.TextToSpeech = TextToSpeech
    tts_stub.stop_all_tts = lambda *args, **kwargs: None
    stubs["tts_module"] = tts_stub

    vector_stub = types.ModuleType("vector")
    vector_stub.search_documents = lambda *args, **kwargs: []
    vector_stub.course_label = lambda course: {
        "algebra_ii": "Algebra II",
        "precalculus": "Precalculus",
    }.get(course, "")
    vector_stub.infer_course_from_filename = lambda _path, subject: (
        "algebra_ii" if subject == "math" else ""
    )
    vector_stub.infer_source_role_from_filename = lambda _path: "textbook"
    stubs["vector"] = vector_stub

    voice_config_stub = types.ModuleType("voice_config")
    voice_config_stub.load_subject_voice_map = lambda *args, **kwargs: {}
    stubs["voice_config"] = voice_config_stub

    local_providers_stub = types.ModuleType("local_providers")
    local_providers_stub.create_chat_model = lambda *args, **kwargs: object()
    stubs["local_providers"] = local_providers_stub

    return stubs


class TestVoiceAgentPersistenceIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.module_patch = patch.dict(sys.modules, _build_stub_modules())
        self.module_patch.start()

        if "voice_agent" in sys.modules:
            del sys.modules["voice_agent"]
        self.voice_agent = importlib.import_module("voice_agent")

    def tearDown(self):
        self.module_patch.stop()
        self.tmp_dir.cleanup()
        if "voice_agent" in sys.modules:
            del sys.modules["voice_agent"]

    def _new_persistence(self):
        return TutorPersistence(
            subjects=self.voice_agent.SUPPORTED_SUBJECTS,
            data_dir=self.tmp_dir.name,
            max_messages_per_subject=100,
        )

    def test_initialization_restores_messages_and_subject(self):
        persistence = self._new_persistence()
        persistence.append_turn(
            "history",
            "Please quiz me on World War I.",
            "Sure. What caused World War I?",
        )
        persistence.set_current_subject("history")

        with patch.object(
            self.voice_agent,
            "TutorPersistence",
            side_effect=lambda *args, **kwargs: persistence,
        ):
            agent = self.voice_agent.VoiceAgent()

        self.assertEqual(agent.current_subject, "history")
        self.assertEqual(len(agent.memories["history"].messages), 2)
        self.assertEqual(
            agent.memories["history"].messages[0].content,
            "Please quiz me on World War I.",
        )
        self.assertEqual(
            agent.memories["history"].messages[1].content,
            "Sure. What caused World War I?",
        )

    def test_ui_selected_subject_overrides_persisted_sticky_subject(self):
        persistence = self._new_persistence()
        persistence.set_current_subject("english")
        captured_subjects = []
        captured_courses = []
        captured_source_modes = []

        def fake_search_documents(*args, **kwargs):
            captured_subjects.append(kwargs.get("subject"))
            captured_courses.append(kwargs.get("course"))
            captured_source_modes.append(kwargs.get("source_mode"))
            return []

        with (
            patch.object(
                self.voice_agent,
                "TutorPersistence",
                side_effect=lambda *args, **kwargs: persistence,
            ),
            patch.object(
                self.voice_agent,
                "search_documents",
                side_effect=fake_search_documents,
            ),
        ):
            agent = self.voice_agent.VoiceAgent(load_stt=False)
            events = list(
                agent.stream_ui_turn(
                    "Can you explain this part?",
                    subject="math",
                    course="algebra_ii",
                    source_mode="textbook",
                    speak=False,
                )
            )

        subject_events = [event for event in events if event.get("type") == "subject"]
        self.assertEqual(agent.current_subject, "math")
        self.assertEqual(subject_events[0]["subject"], "math")
        self.assertEqual(captured_subjects, ["math"])
        self.assertEqual(captured_courses, ["algebra_ii"])
        self.assertEqual(captured_source_modes, ["textbook"])

    def test_remember_turn_persists_conversation_and_questions(self):
        persistence = self._new_persistence()

        with patch.object(
            self.voice_agent,
            "TutorPersistence",
            side_effect=lambda *args, **kwargs: persistence,
        ):
            agent = self.voice_agent.VoiceAgent()

        agent._remember_turn(
            "Give me two chemistry quiz questions.",
            "Here are 2 chemistry quiz questions:\n"
            "1) What is pH?\n"
            "2) What is an acid-base neutralization reaction?",
            "chemistry",
        )

        conversation, current_subject = persistence.load_conversation()
        self.assertEqual(current_subject, "chemistry")
        self.assertEqual(len(conversation["chemistry"]), 2)
        self.assertEqual(conversation["chemistry"][0]["role"], "human")
        self.assertEqual(conversation["chemistry"][1]["role"], "ai")

        question_bank = persistence.load_question_bank()
        self.assertEqual(len(question_bank["chemistry"]), 2)
        stored_questions = [q["question"] for q in question_bank["chemistry"]]
        self.assertIn("What is pH?", stored_questions)

    def test_remember_turn_ignores_non_assessment_questions(self):
        persistence = self._new_persistence()

        with patch.object(
            self.voice_agent,
            "TutorPersistence",
            side_effect=lambda *args, **kwargs: persistence,
        ):
            agent = self.voice_agent.VoiceAgent()

        agent._remember_turn(
            "Can you help me understand this concept?",
            "Do you want me to explain it with an example?",
            "chemistry",
        )

        question_bank = persistence.load_question_bank()
        self.assertEqual(question_bank["chemistry"], [])

    def test_remember_turn_extracts_from_long_assistant_output(self):
        persistence = self._new_persistence()

        with patch.object(
            self.voice_agent,
            "TutorPersistence",
            side_effect=lambda *args, **kwargs: persistence,
        ):
            agent = self.voice_agent.VoiceAgent()

        long_intro = "Background context. " * 80
        assistant_output = (
            f"{long_intro}\n"
            "1) Explain the significance of the Treaty of Versailles.\n"
            "2) Discuss one economic cause of World War II."
        )

        agent._remember_turn(
            "Please create two quiz questions on world history.",
            assistant_output,
            "history",
        )

        question_bank = persistence.load_question_bank()
        self.assertEqual(len(question_bank["history"]), 2)

    def test_remember_turn_persists_markdown_multiple_choice_questions(self):
        persistence = self._new_persistence()

        with patch.object(
            self.voice_agent,
            "TutorPersistence",
            side_effect=lambda *args, **kwargs: persistence,
        ):
            agent = self.voice_agent.VoiceAgent()

        assistant_output = (
            "Here are two multiple choice questions:\n"
            "**Question 1:** Which organelle generates ATP\n"
            "- A) Nucleus\n"
            "- B) Mitochondria\n"
            "- C) Ribosome\n"
            "- D) Golgi apparatus\n\n"
            "**Question 2:** Which process moves water across a semipermeable membrane\n"
            "- A) Diffusion\n"
            "- B) Osmosis\n"
            "- C) Active transport\n"
            "- D) Endocytosis"
        )

        agent._remember_turn(
            "Ask me two biology multiple choice questions.",
            assistant_output,
            "chemistry",
        )

        question_bank = persistence.load_question_bank()
        self.assertEqual(len(question_bank["chemistry"]), 2)
        self.assertTrue(
            all(
                item["question_type"] == "multiple_choice"
                for item in question_bank["chemistry"]
            )
        )
        self.assertIn("B) Mitochondria", question_bank["chemistry"][0]["question"])

    def test_barge_in_disabled_by_default(self):
        persistence = self._new_persistence()
        with patch.object(
            self.voice_agent,
            "TutorPersistence",
            side_effect=lambda *args, **kwargs: persistence,
        ):
            agent = self.voice_agent.VoiceAgent()

        self.assertFalse(agent.barge_in_enabled)

    def test_barge_in_can_be_enabled_via_env(self):
        with patch.dict(os.environ, {"VOICE_BARGE_IN_ENABLED": "1"}, clear=False):
            if "voice_agent" in sys.modules:
                del sys.modules["voice_agent"]
            voice_agent_enabled = importlib.import_module("voice_agent")

            persistence = TutorPersistence(
                subjects=voice_agent_enabled.SUPPORTED_SUBJECTS,
                data_dir=self.tmp_dir.name,
                max_messages_per_subject=100,
            )
            with patch.object(
                voice_agent_enabled,
                "TutorPersistence",
                side_effect=lambda *args, **kwargs: persistence,
            ):
                agent = voice_agent_enabled.VoiceAgent()

        self.assertTrue(agent.barge_in_enabled)


if __name__ == "__main__":
    unittest.main()
