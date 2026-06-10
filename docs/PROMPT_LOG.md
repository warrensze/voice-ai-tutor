# Prompt Log

2026-06-08

- Review `voice-ai-tutor`, review docs, check newer local technology and source libraries, recommend updates, and create concise context docs.
- Constraint added: no online/cloud voice path and no spending money.
- Constraint added: no dependency upgrades unless something is broken or the newer version has significant benefit.
- Investigate a more efficient local source than Ollama, possibly llama.cpp.
- Investigate Piper as a possible alternative or secondary TTS option.
- Make Piper an enabled option that users can switch to easily.
- Start with llama.cpp but keep it selectable through the UI.
- Add a nice local UI similar to voice-chat apps.
- Add the ability for users to immediately add more RAG assets such as PDFs and EPUBs.
- Implementation requested for the local voice tutor UI, provider switching, Piper, llama.cpp support, RAG library, and session docs.
- Asked whether the llama.cpp option had a model behind it; chose Qwen3-8B GGUF for chat and Nomic Embed GGUF for embeddings.
- Requested that Qwen3 setup require no user configuration or setup; app should auto-bootstrap local llama.cpp.
- Requested pre-login setup now: install `llama.cpp`, download/start Qwen3, and prepare local models before the UI is used. Completed, including Nomic embeddings and built-in RAG indexing.
- Reported response sounded like it played from two sources; requested exactly one active TTS at a time with UI switching still allowed.
- Clarified first spoken response immediately sounds like two voices/sentences at
  once; implemented stricter TTS ownership, playback serialization, chat turn
  guarding, and speaking-state wait.
- Still heard overlapping speech; requested service restart and UI voice
  selection. Restarted the local web service, added voice discovery/selection,
  changed pyttsx3 fallback to full-response speech, and cleaned websocket
  disconnect speech shutdown.
- Reported the tutor stuck thinking and Stop not working; added interrupt
  cancellation, moved chat generation off the FastAPI event loop, added
  `/api/voice/stop` cancellation of active turn events, and added response
  timeouts.

2026-06-09

- Reported speech only played one sentence and then stopped; restored
  sentence-chunk speech streaming for pyttsx3 fallback and fixed TTS worker
  cleanup so future speech starts after errors/stops.
- Reported speech overlap again, apparently first and second halves of a response
  speaking at the same time; added macOS `say` subprocess playback for pyttsx3
  fallback, with one blocking process per chunk and Stop termination support.
- Clarified that no provider should silently fall back to another provider;
  changed TTS to strict selected-backend behavior and surfaced `tts_health` in
  the API/UI.
- Requested a local Piper sampling of English accents and Chinese voices;
  downloaded US English, UK English, and Chinese Piper `.onnx` plus `.onnx.json`
  voice files into `models/piper/`.
- Requested an easy way to choose voices; added an always-visible voice selector
  and Test button to the main voice panel, with friendlier Piper labels and
  quick selection applied across all subjects.
- Reported two Chinese Piper voices were not working and asked for a Mandarin
  rather than regional-dialect option; identified `zh_CN-huayan-medium` as the
  best working Mandarin voice and marked pinyin voices unavailable when `g2pw`
  is missing.
- Asked for Kokoro voices to be listed and selectable when Kokoro is used; added
  readable Kokoro voice labels and included Kokoro in the same UI voice picker.
