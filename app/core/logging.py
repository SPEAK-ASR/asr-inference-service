"""Structured JSON logging configuration.

Logs are emitted as one JSON object per line with consistent keys so they can
be ingested by log pipelines without further parsing. Use `bind(...)` style
extras via the standard `logging.LoggerAdapter` if you need per-session keys.

Usage:
    from app.core.logging import configure_logging, get_logger

    configure_logging()
    log = get_logger(__name__)
    log.info("session_started", extra={"session_id": "s123"})
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

from app.core.config import get_settings

_LOG_RECORD_BUILTINS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render log records as JSON lines, preserving `extra={...}` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in _LOG_RECORD_BUILTINS or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info

        return json.dumps(payload, ensure_ascii=False)


_CONFIGURED = False


def configure_logging() -> None:
    """Install the JSON formatter on the root logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)

    for noisy in ("uvicorn.access",):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger; ensures formatting is configured first."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
