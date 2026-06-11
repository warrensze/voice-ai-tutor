# Voice AI Tutor Session Context

Last updated: 2026-06-11

## User Constraints

- Fully local and free: no cloud voice, hosted inference, paid APIs, or cloud OCR.
- Do not upgrade dependencies unless they are needed or offer clear value.
- Prefer llama.cpp as the new efficient local LLM path, while keeping Ollama selectable.
- Make Piper a first-class selectable TTS option, while keeping Kokoro and pyttsx3.
- Do not silently switch providers. If a selected TTS backend is unavailable or
  fails, show the failure instead of substituting another backend.
- STT should also be local and explicit: support only `faster-whisper` and
  `whisper.cpp`, with no hidden provider fallback.
- Math RAG should scale beyond one book by using visible study sets rather than
  searching every math source by default.

## Current Architecture

- Existing terminal app remains available through `python src/main.py` or `voice-ai-tutor`.
- New local web UI is served by FastAPI through `voice-ai-tutor-web`.
- React/Vite frontend lives in `frontend/`.
- Saved UI/runtime settings live in `data/user_settings.json`.
- User-added RAG files live in `data/library/`.
- Provider-specific vector stores live under `data/vector_stores/`; the legacy Ollama default can still use `chrome_langchain_db`.
- Math study-set settings live in `data/user_settings.json`:
  `current_course` (`algebra_ii` or `precalculus`) and `rag_source_mode`
  (`auto`, `textbook`, `workbook`, or `all`).

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
- Built-in Algebra 2 is treated as `course=algebra_ii` and
  `source_role=textbook` by filename alias, so the new study-set filters work
  without forcing a full vector DB rebuild.
- Browser mic STT defaults to local `faster-whisper` with `base.en`.
- `whisper.cpp` is now selectable, but it requires local `whisper-cli`, a local
  GGML model file, and local `ffmpeg` for browser-recorded audio conversion.
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
  - STT: `faster-whisper` or `whisper.cpp`
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
- Kokoro device handling is saved in settings. Default `kokoro_device` is `auto`
  with `kokoro_allow_cpu=true`, so CUDA machines use CUDA and this Mac uses CPU
  without requiring `KOKORO_ALLOW_CPU=1`.
- The main voice panel now has an always-visible voice selector and Test button.
  Quick voice selection applies to all subjects for the active TTS backend.
- UI layout should keep the browser viewport fixed: the chat history and right
  voice/source panel scroll internally, while the chat input remains visible at
  the bottom of the chat panel.
- Browser chat turns now send the selected UI subject over the WebSocket. The
  backend treats that subject as authoritative for specialist routing and RAG
  filtering; sticky text routing is only used when no UI subject is provided.
- Math turns also send the selected course and source mode. Retrieval filters
  by subject plus active course, and optionally source role for Textbook or
  Workbook mode. This keeps future Algebra II workbook and Precalculus sources
  from mixing unless the user changes the active study set.
- Library uploads now store `course`, `source_role`, and `topic_tags` metadata.
  The Study Library upload UI asks for course and asset type when the subject is
  Math.
- Built-in RAG discovery uses the project-root `assets/` path, not the process
  working directory, so web-server startup from `src/` still finds the seeded
  PDFs.
- Current Piper voice is configured as `en_US-lessac-medium`; `tts_health` is
  healthy after installing the local Piper voice files.
- Piper voice labels are formatted for UI readability, such as
  `UK English · Northern English Male · Medium`.
- llama.cpp is self-managed by default:
  - installs `llama.cpp` with Homebrew if `llama-server` is missing
  - starts Qwen chat on `127.0.0.1:8080`
  - starts Nomic embeddings on `127.0.0.1:8081`
- Piper uses local `.onnx` voices only; no automatic downloads.
- Browser mic recordings are uploaded to the local backend and transcribed with
  the selected local STT provider. `/api/status` exposes `stt_health`, and the
  UI settings drawer exposes provider-specific STT settings.
- Library uploads support PDF, EPUB, and OCR/text files, then index into Chroma with asset metadata.
- Heat note from the 2026-06-09 process sample: the tutor `web_server` and both
  `llama-server` processes were nearly idle; the largest CPU users were iOS
  Simulator/Xcode support processes, `WindowServer`, and VS Code GPU/renderer.
  `llama-server` can still heat the laptop during active generation/embedding.

## Useful Commands

- Backend: `python -m web_server` or `voice-ai-tutor-web`
- Frontend dev server: `cd frontend && npm install && npm run dev`
- Frontend production build: `cd frontend && npm run build`
- Legacy terminal app: `python src/main.py`

## Next Checklist

- Download or place Piper voices under `models/piper/`.
- If using `whisper.cpp`, install/place `whisper-cli`, a local GGML Whisper
  model under `models/stt/whisper.cpp/`, and ensure `ffmpeg` is installed.
- Manual audio check: ask one short question and confirm only one sentence/voice
  is audible at a time.
- When adding new math assets, tag Algebra II workbook as
  `course=algebra_ii`, `source_role=workbook`; tag Precalculus textbook as
  `course=precalculus`, `source_role=textbook`.
