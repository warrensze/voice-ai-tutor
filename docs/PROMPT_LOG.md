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
