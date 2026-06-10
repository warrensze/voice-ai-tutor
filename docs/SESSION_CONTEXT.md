# Voice AI Tutor Session Context

Last updated: 2026-06-09

## User Constraints

- Fully local and free: no cloud voice, hosted inference, paid APIs, or cloud OCR.
- Do not upgrade dependencies unless they are needed or offer clear value.
- Prefer llama.cpp as the new efficient local LLM path, while keeping Ollama selectable.
- Make Piper a first-class selectable TTS option, while keeping Kokoro and pyttsx3.
- Do not silently switch providers. If a selected TTS backend is unavailable or
  fails, show the failure instead of substituting another backend.

## Current Architecture

- Existing terminal app remains available through `python src/main.py` or `voice-ai-tutor`.
- New local web UI is served by FastAPI through `voice-ai-tutor-web`.
- React/Vite frontend lives in `frontend/`.
- Saved UI/runtime settings live in `data/user_settings.json`.
- User-added RAG files live in `data/library/`.
- Provider-specific vector stores live under `data/vector_stores/`; the legacy Ollama default can still use `chrome_langchain_db`.

## Current Local Runtime State

- `llama.cpp` is installed through Homebrew.
- Qwen chat model is downloaded/running through `llama-server`:
  - `Qwen/Qwen3-8B-GGUF:Q4_K_M`
  - `http://127.0.0.1:8080/v1`
- Nomic embedding model is downloaded/running through `llama-server`:
  - `nomic-ai/nomic-embed-text-v1.5-GGUF:Q4_K_M`
  - `http://127.0.0.1:8081/v1`
- Built-in `assets/` RAG sources are indexed for the llama.cpp/Nomic embedding store.
- Current built-in index size: 19,273 chunks.
- Local Piper voice models are installed under `models/piper/`:
  - US English: `en_US-lessac-medium`, `en_US-amy-medium`, `en_US-ryan-medium`
  - UK English: `en_GB-alan-medium`, `en_GB-cori-medium`,
    `en_GB-northern_english_male-medium`,
    `en_GB-southern_english_female-low`, `en_GB-vctk-medium`
  - Chinese: `zh_CN-chaowen-medium`, `zh_CN-huayan-medium`,
    `zh_CN-huayan-x_low`, `zh_CN-xiao_ya-medium`
  - Current Piper folder size is about `707M`.
- Working Piper Mandarin voices: `zh_CN-huayan-medium` is the best current
  choice; `zh_CN-huayan-x_low` also works but is lower quality. Both use the
  local espeak `cmn` voice.
- Piper voices `zh_CN-chaowen-medium` and `zh_CN-xiao_ya-medium` require the
  local Python package `g2pw`; until that is installed, they are shown as
  unavailable rather than silently failing or switching voices.

## Implemented Plan

- Local browser voice-chat UI with mic control, transcript, subject chips, source panel, settings drawer, and study library drawer.
- Provider switching:
  - LLM: `llamacpp` or `ollama`
  - embeddings: provider-matched by default
  - TTS: `piper`, `kokoro`, or `pyttsx3`
- Only one TTS instance should be active at a time. Backend speech now uses a
  process-wide owner guard plus a process-wide playback lock across Piper,
  Kokoro, WaveGlow, and pyttsx3. UI TTS switches, voice tests, stop actions, and
  new spoken turns stop current audio first.
- Browser spoken turns now stay in the speaking state until the backend TTS
  queue drains, and the UI blocks a second active chat turn while one is running.
- Stop/interrupt path: `/api/voice/stop` now cancels active turn events, stops
  TTS, resets the runtime signature, and returns independently from the chat
  WebSocket. WebSocket generation runs in a worker thread so the FastAPI event
  loop can still handle Stop/status requests while the model is thinking.
- Time limits:
  - UI turn timeout: `VOICE_TUTOR_TURN_TIMEOUT_SECONDS`, default `60` seconds.
  - llama.cpp chat socket timeout: `VOICE_TUTOR_LLM_TIMEOUT_SECONDS`, default
    `30` seconds.
- Spoken web turns stream sentence chunks through the TTS queue for the selected
  backend only. The queue has a process-wide playback lock, so this should not
  overlap voices.
- TTS selection is strict: Piper, Kokoro, and pyttsx3 do not automatically
  substitute for each other. `/api/status` exposes `tts_health`, and the UI voice
  panel shows the selected backend as ready or unavailable.
- On macOS, the selected pyttsx3 backend uses the blocking local `say` command by
  default (`TTS_USE_MACOS_SAY=1`) so sentence chunks cannot be queued into
  overlapping OS speech; Stop terminates the active `say` process.
- The async TTS worker always marks itself finished in `finally`, even after a
  pyttsx3 error or interruption, so later responses can speak again.
- `/api/voices` lists selectable voices for Kokoro, Piper, and pyttsx3. The UI
  Settings drawer has a per-current-subject voice picker and an apply-to-all
  button.
- Kokoro voices are exposed with readable labels by language, gender, and voice
  name, including American English, British English, Spanish, French, Hindi,
  Italian, Japanese, Brazilian Portuguese, and Mandarin Chinese voices.
- The main voice panel now has an always-visible voice selector and Test button.
  Quick voice selection applies to all subjects for the active TTS backend.
- UI layout should keep the browser viewport fixed: the chat history and right
  voice/source panel scroll internally, while the chat input remains visible at
  the bottom of the chat panel.
- Current Piper voice is configured as `en_US-lessac-medium`; `tts_health` is
  healthy after installing the local Piper voice files.
- Piper voice labels are formatted for UI readability, such as
  `UK English · Northern English Male · Medium`.
- llama.cpp is self-managed by default:
  - installs `llama.cpp` with Homebrew if `llama-server` is missing
  - starts Qwen chat on `127.0.0.1:8080`
  - starts Nomic embeddings on `127.0.0.1:8081`
- Piper uses local `.onnx` voices only; no automatic downloads.
- Browser mic recordings are uploaded to the local backend and transcribed with local faster-whisper.
- Library uploads support PDF, EPUB, and OCR/text files, then index into Chroma with asset metadata.

## Useful Commands

- Backend: `python -m web_server` or `voice-ai-tutor-web`
- Frontend dev server: `cd frontend && npm install && npm run dev`
- Frontend production build: `cd frontend && npm run build`
- Legacy terminal app: `python src/main.py`

## Next Checklist

- Download or place Piper voices under `models/piper/`.
- Fix `builtin_sources` in `/api/status`: chunk counts show the built-in
  `assets/` sources are indexed, but the source-list payload is currently empty.
- Manual audio check: ask one short question and confirm only one sentence/voice
  is audible at a time.
