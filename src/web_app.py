"""FastAPI backend for the local Voice AI Tutor browser UI."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from llamacpp_manager import (
    ensure_llamacpp_bootstrap,
    llamacpp_bootstrap_status,
    refresh_llamacpp_status,
)
from local_providers import provider_status
from rag_library import LibraryManager
from settings_store import (
    PROJECT_ROOT,
    load_user_settings,
    save_user_settings,
    settings_status_payload,
    update_user_settings,
)
from stt_module import STTBackendUnavailable, SpeechToText, stt_backend_status
from tts_module import list_tts_voices, stop_all_tts, tts_backend_status
from vector import get_ingestion_summary
from voice_agent import VoiceAgent

UI_TURN_TIMEOUT_SECONDS = float(
    os.getenv("VOICE_TUTOR_TURN_TIMEOUT_SECONDS", "60")
)

app = FastAPI(title="Voice AI Tutor", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

library = LibraryManager()
_runtime_lock = threading.Lock()
_runtime: VoiceAgent | None = None
_runtime_signature = ""
_stt_lock = threading.Lock()
_stt: SpeechToText | None = None
_turn_events_lock = threading.Lock()
_active_turn_events: set[threading.Event] = set()


def _runtime_for_settings() -> VoiceAgent:
    global _runtime, _runtime_signature
    settings = load_user_settings()
    signature = settings.provider_signature()
    with _runtime_lock:
        if _runtime is None or signature != _runtime_signature:
            if _runtime is not None:
                try:
                    _runtime.mouth.stop(wait=False)
                except Exception:
                    pass
                stop_all_tts(wait=False)
            _runtime = VoiceAgent(settings=settings, load_stt=False)
            _runtime_signature = signature
        return _runtime


def _stt_instance() -> SpeechToText:
    global _stt
    with _stt_lock:
        if _stt is None:
            settings = load_user_settings()
            runtime = _runtime_for_settings()
            _stt = SpeechToText(tts_instance=runtime.mouth, settings=settings)
        return _stt


def _register_turn_event(event: threading.Event):
    with _turn_events_lock:
        _active_turn_events.add(event)


def _unregister_turn_event(event: threading.Event):
    with _turn_events_lock:
        _active_turn_events.discard(event)


def _request_turn_stop():
    global _runtime_signature
    with _turn_events_lock:
        events = list(_active_turn_events)
    for event in events:
        event.set()
    with _runtime_lock:
        runtime = _runtime
    if runtime is not None:
        try:
            runtime.cancel_current_turn()
        except Exception:
            pass
        try:
            runtime.mouth.stop(wait=False)
        except Exception:
            pass
    stop_all_tts(wait=False)
    _runtime_signature = ""


async def _send_turn_from_worker(
    websocket: WebSocket,
    runtime: VoiceAgent,
    question: str,
    *,
    subject: str | None,
    speak: bool,
    stop_event: threading.Event,
):
    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    started_at = time.monotonic()

    def push_event(event: dict[str, Any] | None):
        try:
            loop.call_soon_threadsafe(event_queue.put_nowait, event)
        except RuntimeError:
            pass

    def worker():
        try:
            for event in runtime.stream_ui_turn(
                question,
                subject=subject,
                speak=speak,
                stop_event=stop_event,
                timeout_seconds=UI_TURN_TIMEOUT_SECONDS,
            ):
                push_event(event)
                if event.get("type") in {"done", "error", "stopped"}:
                    break
                if stop_event.is_set():
                    break
        except Exception as error:
            push_event({"type": "error", "message": str(error)})
        finally:
            push_event(None)

    thread = threading.Thread(
        target=worker,
        daemon=True,
        name="TutorTurnWorker",
    )
    thread.start()

    terminal_sent = False
    while True:
        if stop_event.is_set():
            await websocket.send_json(
                {
                    "type": "stopped",
                    "reason": "cancelled",
                    "message": "Stopped.",
                }
            )
            terminal_sent = True
            break

        if (
            UI_TURN_TIMEOUT_SECONDS > 0
            and (time.monotonic() - started_at) >= UI_TURN_TIMEOUT_SECONDS
        ):
            stop_event.set()
            _request_turn_stop()
            await websocket.send_json(
                {
                    "type": "stopped",
                    "reason": "timeout",
                    "message": (
                        "I stopped because this response took longer than "
                        f"{int(UI_TURN_TIMEOUT_SECONDS)} seconds."
                    ),
                }
            )
            terminal_sent = True
            break

        try:
            event = await asyncio.wait_for(event_queue.get(), timeout=0.25)
        except asyncio.TimeoutError:
            continue

        if event is None:
            break

        if stop_event.is_set():
            continue

        await websocket.send_json(event)
        if event.get("type") in {"done", "error", "stopped"}:
            terminal_sent = True
            break

    if terminal_sent:
        stop_event.set()


def _check_url(url: str, *, timeout: float = 1.2) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=timeout) as response:
            return {"ok": True, "status": response.status}
    except URLError as error:
        return {"ok": False, "error": str(error)}
    except Exception as error:
        return {"ok": False, "error": str(error)}


def _provider_health() -> dict[str, Any]:
    settings = load_user_settings()
    if settings.llm_provider == "llamacpp" or settings.embedding_provider == "llamacpp":
        bootstrap_status = ensure_llamacpp_bootstrap(settings)
    else:
        bootstrap_status = refresh_llamacpp_status()

    status = provider_status(settings)
    if settings.llm_provider == "ollama":
        chat_url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    else:
        chat_url = f"{settings.llamacpp_chat_base_url.rstrip('/')}/models"

    if settings.embedding_provider == "ollama":
        embedding_url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    else:
        embedding_url = f"{settings.llamacpp_embedding_base_url.rstrip('/')}/models"

    return {
        **status,
        "chat_health": _check_url(chat_url),
        "embedding_health": _check_url(embedding_url),
        "tts_health": tts_backend_status(settings),
        "stt_health": stt_backend_status(settings),
        "llamacpp_bootstrap": bootstrap_status,
    }


@app.get("/api/settings")
def get_settings():
    return settings_status_payload(load_user_settings())


@app.get("/api/voices")
def get_voices():
    settings = load_user_settings()
    return {
        "voices": list_tts_voices(settings),
        "selected": {
            "backend": settings.tts_backend,
            "subject": settings.current_subject,
            "voice": settings.selected_voice(settings.current_subject),
        },
    }


@app.put("/api/settings")
async def put_settings(payload: dict[str, Any]):
    _request_turn_stop()
    old_settings = load_user_settings()
    settings = update_user_settings(payload)
    save_user_settings(settings)
    global _runtime_signature, _stt
    if (
        old_settings.tts_backend != settings.tts_backend
        or old_settings.provider_signature() != settings.provider_signature()
    ):
        stop_all_tts(wait=False)
        _stt = None
    _runtime_signature = ""
    if settings.llm_provider == "llamacpp" or settings.embedding_provider == "llamacpp":
        ensure_llamacpp_bootstrap(settings)
    return settings_status_payload(settings)


@app.get("/api/status")
def get_status():
    settings = load_user_settings()
    return {
        "providers": _provider_health(),
        "vector": get_ingestion_summary(settings=settings),
        "settings": settings.to_dict(),
    }


@app.post("/api/providers/llamacpp/bootstrap")
def bootstrap_llamacpp():
    settings = load_user_settings()
    return ensure_llamacpp_bootstrap(settings)


@app.get("/api/providers/llamacpp/bootstrap")
def get_llamacpp_bootstrap():
    return llamacpp_bootstrap_status()


@app.get("/api/library")
def list_library():
    return {"assets": library.list_assets()}


@app.post("/api/library/assets")
async def upload_asset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    subject: str = Form("english"),
    title: str = Form(""),
    notes: str = Form(""),
):
    suffix = Path(file.filename or "asset").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        tmp_path = Path(handle.name)
        handle.write(await file.read())

    try:
        asset = library.add_asset(
            tmp_path,
            original_filename=file.filename,
            subject=subject,
            title=title,
            notes=notes,
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    if not asset.get("duplicate"):
        background_tasks.add_task(
            library.index_asset,
            asset["id"],
            settings=load_user_settings(),
        )
    return asset


@app.patch("/api/library/assets/{asset_id}")
async def update_asset(asset_id: str, payload: dict[str, Any]):
    asset = library.update_asset(asset_id, payload)
    return asset


@app.post("/api/library/assets/{asset_id}/reindex")
def reindex_asset(asset_id: str, background_tasks: BackgroundTasks):
    asset = library.update_asset(asset_id, {"status": "queued"})
    background_tasks.add_task(
        library.index_asset,
        asset_id,
        settings=load_user_settings(),
    )
    return asset


@app.delete("/api/library/assets/{asset_id}")
def delete_asset(asset_id: str):
    removed = library.remove_asset(asset_id, settings=load_user_settings())
    return {"removed": removed is not None, "asset": removed}


@app.get("/api/library/assets/{asset_id}/preview")
def preview_asset(asset_id: str):
    return {"asset_id": asset_id, "text": library.preview_asset(asset_id)}


@app.post("/api/voice/test")
async def test_voice(payload: dict[str, Any]):
    _request_turn_stop()
    runtime = _runtime_for_settings()
    if not runtime.mouth.is_available():
        return {
            "ok": False,
            "backend": runtime.mouth.backend,
            "voice": runtime.mouth.voice,
            "error": runtime.mouth.backend_error,
        }
    text = str(payload.get("text") or "This is the local tutor voice.")
    runtime.mouth.stop(wait=False, release_owner=False)
    stop_all_tts(except_instance=runtime.mouth, wait=False)
    spoken = runtime.mouth.speak_async(text)
    return {
        "ok": bool(spoken),
        "backend": runtime.mouth.backend,
        "voice": runtime.mouth.voice,
        "error": "" if spoken else runtime.mouth.backend_error,
    }


@app.post("/api/voice/stop")
def stop_voice():
    _request_turn_stop()
    return {"ok": True}


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    settings = load_user_settings()
    suffix = Path(file.filename or "recording.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        tmp_path = Path(handle.name)
        handle.write(await file.read())

    try:
        text = _stt_instance().transcribe_file(tmp_path)
    except STTBackendUnavailable as error:
        return {
            "ok": False,
            "text": "",
            "provider": settings.stt_provider,
            "error": str(error),
        }
    except Exception as error:
        return {
            "ok": False,
            "text": "",
            "provider": settings.stt_provider,
            "error": f"Transcription failed: {error}",
        }
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return {
        "ok": True,
        "text": text,
        "provider": settings.stt_provider,
        "error": "",
    }


@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON."})
                continue

            if payload.get("type") == "stop":
                _request_turn_stop()
                await websocket.send_json({"type": "stopped"})
                continue

            question = str(payload.get("question") or "").strip()
            if not question:
                await websocket.send_json(
                    {"type": "error", "message": "Question is empty."}
                )
                continue

            runtime = _runtime_for_settings()
            speak = bool(payload.get("speak", load_user_settings().speak_responses))
            subject = str(payload.get("subject") or "").strip().lower() or None
            stop_event = threading.Event()
            _register_turn_event(stop_event)
            await websocket.send_json({"type": "status", "status": "thinking"})
            try:
                await _send_turn_from_worker(
                    websocket,
                    runtime,
                    question,
                    subject=subject,
                    speak=speak,
                    stop_event=stop_event,
                )
            finally:
                _unregister_turn_event(stop_event)
    except WebSocketDisconnect:
        _request_turn_stop()


dist_dir = PROJECT_ROOT / "frontend" / "dist"
if dist_dir.exists():
    assets_dir = dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{path:path}")
    def serve_frontend(path: str):
        requested = dist_dir / path
        if requested.exists() and requested.is_file():
            return FileResponse(requested)
        return FileResponse(dist_dir / "index.html")
