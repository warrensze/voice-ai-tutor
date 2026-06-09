# Voice AI Tutor Session Context

Last updated: 2026-06-08

## User Constraints

- Fully local and free: no cloud voice, hosted inference, paid APIs, or cloud OCR.
- Do not upgrade dependencies unless they are needed or offer clear value.
- Prefer llama.cpp as the new efficient local LLM path, while keeping Ollama selectable.
- Make Piper a first-class selectable TTS option, while keeping Kokoro and pyttsx3.

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
- If the selected backend falls back to pyttsx3, spoken web turns now speak the
  completed response as one pyttsx3 engine run instead of feeding pyttsx3
  sentence-by-sentence.
- `/api/voices` lists selectable voices for Kokoro, Piper, and pyttsx3. The UI
  Settings drawer has a per-current-subject voice picker and an apply-to-all
  button.
- Current Piper voice is configured as `en_US-lessac-medium`, but no local Piper
  `.onnx` model exists under `models/piper/`, so Piper currently falls back.
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
