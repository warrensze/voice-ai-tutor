import sys
import os
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tts_module import TTSBackendUnavailable, TextToSpeech, tts_backend_status


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


class FakeProcess:
    def __init__(self):
        self.terminated = False
        self.killed = False

    def poll(self):
        return 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


class BlockingFakeProcess:
    def __init__(self):
        self.started = threading.Event()
        self.terminated = threading.Event()
        self.killed = False

    def poll(self):
        self.started.set()
        return 0 if self.terminated.is_set() else None

    def terminate(self):
        self.terminated.set()

    def kill(self):
        self.killed = True
        self.terminated.set()

    def wait(self, timeout=None):
        self.terminated.wait(timeout=timeout)
        return 0


class FailingPiperVoice:
    def synthesize(self, text, **kwargs):
        raise RuntimeError("piper failed")


class TestTextToSpeech(unittest.TestCase):
    def _create_tts(self) -> TextToSpeech:
        with (
            patch.dict(
                os.environ,
                {"TTS_REQUIRE_KOKORO": "0", "TTS_USE_MACOS_SAY": "0"},
                clear=False,
            ),
            patch.multiple(
                "tts_module",
                KPipeline=None,
                sd=None,
                WaveGlowConfig=None,
                WaveGlowSynthesizer=None,
            ),
        ):
            return TextToSpeech()

    def test_default_voice_is_af_heart(self):
        tts = self._create_tts()
        self.assertEqual(tts.voice, "af_heart")

    def test_set_voice_updates_voice_name(self):
        tts = self._create_tts()
        tts.set_voice("af_bella")
        self.assertEqual(tts.voice, "af_bella")

    def test_speak_ignores_empty_text(self):
        tts = self._create_tts()
        with patch("tts_module.pyttsx3.init") as mock_init:
            tts.speak("")
        mock_init.assert_not_called()

    def test_speak_initializes_and_speaks(self):
        engine = FakeEngine()
        tts = self._create_tts()
        tts.backend = "pyttsx3"
        tts._kokoro_pipeline = None
        tts._waveglow_synth = None

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
        tts = self._create_tts()
        tts.backend = "pyttsx3"
        tts._kokoro_pipeline = None
        tts._waveglow_synth = None

        with patch(
            "tts_module.pyttsx3.init", side_effect=[failing_engine, working_engine]
        ) as mock_init:
            tts.speak("retry this")

        self.assertEqual(mock_init.call_count, 2)
        self.assertEqual(working_engine.spoken, ["retry this"])

    def test_selected_piper_does_not_fallback_when_unavailable(self):
        with (
            patch.dict(os.environ, {"TTS_USE_MACOS_SAY": "0"}, clear=False),
            patch.multiple("tts_module", sd=object(), PiperVoice=None),
        ):
            tts = TextToSpeech(backend="piper", voice="missing-piper-voice")

        self.assertEqual(tts.backend, "piper")
        self.assertFalse(tts.is_available())
        self.assertIn("Piper", tts.backend_error)
        with patch.object(tts, "_speak_with_engine_chunks") as mock_engine:
            self.assertFalse(tts.speak_async("hello"))
            with self.assertRaises(TTSBackendUnavailable):
                tts.speak("hello")
        mock_engine.assert_not_called()

    def test_selected_piper_playback_error_does_not_fallback(self):
        tts = self._create_tts()
        tts.backend = "piper"
        tts.backend_available = True
        tts._piper_voice = FailingPiperVoice()

        with patch.object(tts, "_speak_with_engine_chunks") as mock_engine:
            with self.assertRaises(TTSBackendUnavailable):
                tts._speak_chunks("hello")

        mock_engine.assert_not_called()
        self.assertFalse(tts.is_available())
        self.assertIn("Piper playback failed", tts.backend_error)

    def test_tts_backend_status_reports_missing_piper_voice(self):
        settings = type(
            "Settings",
            (),
            {
                "tts_backend": "piper",
                "current_subject": "english",
                "piper_data_dir": "models/piper",
                "selected_voice": lambda self, subject=None: "missing-piper-voice",
            },
        )()

        with patch.multiple("tts_module", sd=object(), PiperVoice=object()):
            status = tts_backend_status(settings)

        self.assertFalse(status["ok"])
        self.assertEqual(status["backend"], "piper")
        self.assertIn("was not found", status["error"])

    def test_new_tts_instance_stops_previous_instance_before_speaking(self):
        first = self._create_tts()
        second = self._create_tts()
        first.stop = MagicMock()
        second._speak_chunks = MagicMock()

        second.speak("hello from the selected voice")

        first.stop.assert_called()
        second._speak_chunks.assert_called_once_with("hello from the selected voice")

    def test_split_for_async_queue_preserves_all_text(self):
        tts = self._create_tts()

        tts._max_async_chars = 24
        parts = tts._split_for_async_queue(
            "This answer is long enough to require splitting into multiple spoken segments."
        )

        self.assertTrue(parts)
        self.assertTrue(all(len(part) <= 24 for part in parts))
        reconstructed = " ".join(parts)
        self.assertIn("long enough", reconstructed)
        self.assertIn("spoken segments", reconstructed)

    def test_async_queue_backpressure_does_not_drop_segments(self):
        tts = self._create_tts()

        tts._max_async_queue = 1
        tts._max_async_chars = 48
        spoken: list[str] = []

        def slow_speak(text: str):
            spoken.append(text)
            time.sleep(0.01)

        tts._speak_chunks = slow_speak

        chunks = [
            "Segment one.",
            "Segment two.",
            "Segment three.",
            "Segment four.",
        ]

        for chunk in chunks:
            tts.speak_async(chunk)

        tts.wait_until_done()
        self.assertEqual(spoken, chunks)

    def test_async_worker_resets_after_speech_error(self):
        tts = self._create_tts()
        attempts = []

        def fail_once(text: str):
            attempts.append(text)
            raise RuntimeError("temporary speech failure")

        tts._speak_chunks = fail_once
        tts.speak_async("First response.")
        tts.wait_until_done()
        self.assertIsNone(tts._speak_thread)

        spoken: list[str] = []
        tts._speak_chunks = lambda text: spoken.append(text)
        tts.speak_async("Second response.")
        tts.wait_until_done()

        self.assertEqual(attempts, ["First response."])
        self.assertEqual(spoken, ["Second response."])

    def test_macos_say_path_speaks_chunks_sequentially(self):
        tts = self._create_tts()
        tts._use_macos_say = True
        processes = [FakeProcess(), FakeProcess()]

        with (
            patch("tts_module.platform.system", return_value="Darwin"),
            patch("tts_module.shutil.which", return_value="/usr/bin/say"),
            patch("tts_module.subprocess.Popen", side_effect=processes) as mock_popen,
            patch("tts_module.pyttsx3.init") as mock_init,
        ):
            tts._speak_with_engine_chunks("First sentence. Second sentence.")

        self.assertEqual(mock_popen.call_count, 2)
        self.assertEqual(
            [call.args[0][-1] for call in mock_popen.call_args_list],
            ["First sentence.", "Second sentence."],
        )
        mock_init.assert_not_called()

    def test_stop_terminates_active_macos_say_process(self):
        tts = self._create_tts()
        tts._use_macos_say = True
        process = BlockingFakeProcess()

        with (
            patch("tts_module.platform.system", return_value="Darwin"),
            patch("tts_module.shutil.which", return_value="/usr/bin/say"),
            patch("tts_module.subprocess.Popen", return_value=process),
        ):
            thread = threading.Thread(
                target=tts._speak_with_engine_chunks,
                args=("Keep speaking until stopped.",),
            )
            thread.start()
            self.assertTrue(process.started.wait(timeout=1.0))
            tts.stop(wait=False)
            thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(process.terminated.is_set())


if __name__ == "__main__":
    unittest.main()
