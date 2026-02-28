"""Logging helpers that prepend DriftWatch run identifiers."""

from __future__ import annotations

import logging


class RunLoggerAdapter(logging.LoggerAdapter):
    """Prefix log messages with the active run identifier."""

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        """Insert the run identifier ahead of the log message."""

        run_id = self.extra.get("run_id")
        if not run_id:
            return msg, kwargs
        return f"[{run_id}] {msg}", kwargs


def get_run_logger(logger: logging.Logger, run_id: str | None) -> logging.LoggerAdapter:
    """Return a logger adapter that prefixes messages with a run identifier."""

    return RunLoggerAdapter(logger, {"run_id": run_id})
