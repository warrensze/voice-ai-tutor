"""Crash and runtime logging helpers for post-crash diagnosis."""

from __future__ import annotations

import atexit
import faulthandler
import logging
import os
import signal
import sys
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "voice_ai_tutor"
_LOGGING_INITIALIZED = False
_LOG_DIR: Path | None = None
_FATAL_LOG_HANDLE = None


def _flush_handlers() -> None:
    """Flush all logging handlers best-effort."""
    seen = set()
    for logger_name in ("", LOGGER_NAME):
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            if id(handler) in seen:
                continue
            seen.add(id(handler))
            try:
                handler.flush()
            except Exception:
                pass


def _configure_python_hooks(logger: logging.Logger) -> None:
    """Install global crash hooks for unhandled exceptions."""

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical(
            "Unhandled top-level exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        _flush_handlers()
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = handle_exception

    if hasattr(threading, "excepthook"):

        def handle_thread_exception(args):
            logger.critical(
                "Unhandled exception in thread '%s'",
                getattr(args.thread, "name", "unknown"),
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
            _flush_handlers()

        threading.excepthook = handle_thread_exception


def _configure_faulthandler(log_dir: Path, logger: logging.Logger) -> None:
    """Enable fault logging for native crashes and hard faults."""
    global _FATAL_LOG_HANDLE

    fatal_log_path = log_dir / "fatal.log"
    _FATAL_LOG_HANDLE = open(fatal_log_path, "a", encoding="utf-8", buffering=1)
    _FATAL_LOG_HANDLE.write(
        f"\n===== Session {datetime.utcnow().isoformat()}Z pid={os.getpid()} =====\n"
    )

    faulthandler.enable(file=_FATAL_LOG_HANDLE, all_threads=True)

    register_fault_handler = getattr(faulthandler, "register", None)
    if register_fault_handler is None:
        logger.info(
            "faulthandler.register is unavailable in this runtime; "
            "continuing with faulthandler.enable only"
        )
        return

    # Register handlers where supported so we get low-level traces on hard crashes.
    for signal_name in ("SIGABRT", "SIGFPE", "SIGILL", "SIGSEGV", "SIGTERM"):
        sig = getattr(signal, signal_name, None)
        if sig is None:
            continue
        try:
            register_fault_handler(sig, file=_FATAL_LOG_HANDLE, all_threads=True)
        except (RuntimeError, OSError, ValueError):
            # Some signals cannot be registered on some platforms/runtimes.
            continue

    logger.info("Fault handler enabled: %s", fatal_log_path)


def setup_crash_logging(log_dir: str | Path | None = None) -> Path:
    """Initialize process-wide crash logging and return the log directory."""
    global _LOGGING_INITIALIZED, _LOG_DIR

    if _LOGGING_INITIALIZED and _LOG_DIR is not None:
        return _LOG_DIR

    project_root = Path(__file__).resolve().parents[1]
    resolved_log_dir = Path(log_dir) if log_dir else (project_root / "logs")
    resolved_log_dir.mkdir(parents=True, exist_ok=True)

    app_log_path = resolved_log_dir / "app.log"
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not any(
        isinstance(handler, RotatingFileHandler)
        and Path(getattr(handler, "baseFilename", "")) == app_log_path
        for handler in logger.handlers
    ):
        file_handler = RotatingFileHandler(
            app_log_path,
            maxBytes=5_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _configure_python_hooks(logger)
    _configure_faulthandler(resolved_log_dir, logger)

    def on_exit() -> None:
        logger.info("Process exiting (pid=%s)", os.getpid())
        _flush_handlers()

    atexit.register(on_exit)

    logger.info("Crash logging initialized in %s", resolved_log_dir)
    logger.info("Python executable: %s", sys.executable)
    logger.info("Python version: %s", sys.version.replace("\n", " "))

    _LOG_DIR = resolved_log_dir
    _LOGGING_INITIALIZED = True
    return resolved_log_dir
