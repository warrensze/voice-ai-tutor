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

            updated = update_user_settings({"llm_provider": "llamacpp"}, path)
            self.assertEqual(updated.llm_provider, "llamacpp")
            self.assertEqual(load_user_settings(path).llm_provider, "llamacpp")

    def test_invalid_choices_fall_back_to_safe_defaults(self):
        settings = UserSettings.from_dict(
            {
                "llm_provider": "cloud-provider",
                "tts_backend": "remote-voice",
                "current_subject": "astronomy",
            }
        )

        self.assertEqual(settings.llm_provider, "llamacpp")
        self.assertEqual(settings.tts_backend, "piper")
        self.assertEqual(settings.current_subject, "english")


if __name__ == "__main__":
    unittest.main()
