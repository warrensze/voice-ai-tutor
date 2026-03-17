import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tts_module import TextToSpeech


class FakeEngine:
    def __init__(self, fail_run=False):
        self.fail_run = fail_run
        self.rate = None
        self.volume = None
        self.spoken = []
        self.stopped = False
        self.voice_id = None
        self.available_voices = [
            type("Voice", (), {"id": "voice-zira", "name": "Microsoft Zira"})(),
            type("Voice", (), {"id": "voice-david", "name": "Microsoft David"})(),
        ]

    def setProperty(self, name, value):
        if name == "rate":
            self.rate = value
        if name == "volume":
            self.volume = value
        if name == "voice":
            self.voice_id = value

    def getProperty(self, name):
        if name == "voices":
            return self.available_voices
        return None

    def say(self, text):
        self.spoken.append(text)

    def runAndWait(self):
        if self.fail_run:
            raise RuntimeError("engine failed")

    def stop(self):
        self.stopped = True


class TestTextToSpeech(unittest.TestCase):
    def test_default_voice_is_af_heart(self):
        with patch("tts_module.KPipeline", None), patch("tts_module.sd", None):
            tts = TextToSpeech()
        self.assertEqual(tts.voice, "af_heart")

    def test_set_voice_updates_voice_name(self):
        with patch("tts_module.KPipeline", None), patch("tts_module.sd", None):
            tts = TextToSpeech()
            tts.set_voice("af_bella")
        self.assertEqual(tts.voice, "af_bella")

    def test_speak_ignores_empty_text(self):
        with patch("tts_module.KPipeline", None), patch("tts_module.sd", None):
            tts = TextToSpeech()
        with patch("tts_module.pyttsx3.init") as mock_init:
            tts.speak("")
        mock_init.assert_not_called()

    def test_speak_initializes_and_speaks(self):
        engine = FakeEngine()
        with patch("tts_module.KPipeline", None), patch("tts_module.sd", None):
            tts = TextToSpeech()

        with patch("tts_module.pyttsx3.init", return_value=engine) as mock_init:
            tts.speak("hello")

        mock_init.assert_called_once()
        self.assertEqual(engine.spoken, ["hello"])
        self.assertEqual(engine.rate, 175)
        self.assertEqual(engine.volume, 1.0)
        self.assertTrue(engine.stopped)

    def test_speak_retries_once_after_failure(self):
        failing_engine = FakeEngine(fail_run=True)
        working_engine = FakeEngine()
        with patch("tts_module.KPipeline", None), patch("tts_module.sd", None):
            tts = TextToSpeech()

        with patch(
            "tts_module.pyttsx3.init", side_effect=[failing_engine, working_engine]
        ) as mock_init:
            tts.speak("retry this")

        self.assertEqual(mock_init.call_count, 2)
        self.assertEqual(working_engine.spoken, ["retry this"])


if __name__ == "__main__":
    unittest.main()
