"""Local web server entrypoint for the Voice AI Tutor UI."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("VOICE_TUTOR_HOST", "127.0.0.1")
    port = int(os.getenv("VOICE_TUTOR_PORT", "8000"))
    uvicorn.run("web_app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
