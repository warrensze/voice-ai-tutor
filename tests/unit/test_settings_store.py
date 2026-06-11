import tempfile
import unittest
from pathlib import Path

from settings_store import UserSettings, load_user_settings, save_user_settings, update_user_settings


class TestSettingsStore(unittest.TestCase):
    def test_save_load_and_update_settings(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "settings.json"
            settings = UserSettings(llm_provider="ollama", tts_backend="kokoro")

            save_user_settings(settings, path)
            loaded = load_user_settings(path)

            self.assertEqual(loaded.llm_provider, "ollama")
            self.assertEqual(loaded.tts_backend, "kokoro")
            self.assertEqual(loaded.stt_provider, "faster-whisper")
            self.assertEqual(loaded.current_course, "algebra_ii")
            self.assertEqual(loaded.rag_source_mode, "auto")

            updated = update_user_settings({"llm_provider": "llamacpp"}, path)
            self.assertEqual(updated.llm_provider, "llamacpp")
            self.assertEqual(load_user_settings(path).llm_provider, "llamacpp")
            self.assertEqual(loaded.kokoro_device, "auto")
            self.assertTrue(loaded.kokoro_allow_cpu)

    def test_invalid_choices_fall_back_to_safe_defaults(self):
        settings = UserSettings.from_dict(
            {
                "llm_provider": "cloud-provider",
                "tts_backend": "remote-voice",
                "stt_provider": "cloud-stt",
                "current_subject": "astronomy",
                "current_course": "algebra 2",
                "rag_source_mode": "space-dust",
                "kokoro_device": "quantum",
                "faster_whisper_device": "neural",
            }
        )

        self.assertEqual(settings.llm_provider, "llamacpp")
        self.assertEqual(settings.tts_backend, "piper")
        self.assertEqual(settings.stt_provider, "faster-whisper")
        self.assertEqual(settings.current_subject, "english")
        self.assertEqual(settings.current_course, "algebra_ii")
        self.assertEqual(settings.rag_source_mode, "auto")
        self.assertEqual(settings.kokoro_device, "auto")
        self.assertEqual(settings.faster_whisper_device, "auto")

    def test_stt_provider_aliases_are_normalized(self):
        settings = UserSettings.from_dict({"stt_provider": "whisper.cpp"})

        self.assertEqual(settings.stt_provider, "whispercpp")


if __name__ == "__main__":
    unittest.main()
