import unittest
from unittest.mock import patch

from settings_store import UserSettings
from stt_module import STTBackendUnavailable, SpeechToText, stt_backend_status


class TestSTTProviderSwitching(unittest.TestCase):
    def test_faster_whisper_status_uses_selected_model(self):
        settings = UserSettings.from_dict(
            {
                "stt_provider": "faster-whisper",
                "faster_whisper_model": "base.en",
                "faster_whisper_device": "cpu",
            }
        )

        with patch("stt_module.WhisperModel", object()):
            status = stt_backend_status(settings)

        self.assertTrue(status["ok"])
        self.assertEqual(status["provider"], "faster-whisper")
        self.assertEqual(status["model"], "base.en")
        self.assertEqual(status["device"], "cpu")

    def test_faster_whisper_constructor_uses_cpu_without_cuda_fallback(self):
        settings = UserSettings.from_dict(
            {
                "stt_provider": "faster-whisper",
                "faster_whisper_model": "base.en",
                "faster_whisper_device": "cpu",
            }
        )

        with patch("stt_module.WhisperModel") as mock_model:
            stt = SpeechToText(settings=settings)

        self.assertEqual(stt.provider, "faster-whisper")
        mock_model.assert_called_once_with(
            "base.en",
            device="cpu",
            compute_type="int8",
        )

    def test_whispercpp_reports_missing_local_requirements(self):
        settings = UserSettings.from_dict(
            {
                "stt_provider": "whispercpp",
                "whispercpp_binary_path": "/missing/whisper-cli",
                "whispercpp_model_path": "models/stt/whisper.cpp/missing.bin",
            }
        )

        status = stt_backend_status(settings)

        self.assertFalse(status["ok"])
        self.assertEqual(status["provider"], "whispercpp")
        self.assertIn("whisper.cpp is selected", status["error"])

    def test_whispercpp_does_not_fall_back_to_faster_whisper(self):
        settings = UserSettings.from_dict(
            {
                "stt_provider": "whispercpp",
                "whispercpp_binary_path": "/missing/whisper-cli",
                "whispercpp_model_path": "models/stt/whisper.cpp/missing.bin",
            }
        )

        with patch("stt_module.WhisperModel") as mock_model:
            with self.assertRaises(STTBackendUnavailable):
                SpeechToText(settings=settings)

        mock_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
