# forecast_pipeline/logging_utils.py
from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from logging.handlers import QueueHandler, QueueListener
from typing import Any, Dict, Iterable, Optional
import contextvars


# ---------------------------
# Context variables (thread/mp safe)
# ---------------------------
_LOG_CONTEXT: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)

def _merge_context(extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = dict(_LOG_CONTEXT.get())
    if extra:
        base.update({k: v for k, v in extra.items() if v is not None})
    return base

class ContextAdapter(logging.LoggerAdapter):
    """LoggerAdapter that merges adapter.extra + kwargs.extra + contextvars into record extras."""
    def process(self, msg, kwargs):
        # start with adapter's fixed context
        merged = dict(getattr(self, "extra", {}) or {})
        # then any per-call extras
        call_extra = kwargs.get("extra") or {}
        merged.update(call_extra)
        # finally merge contextvars (thread/mp safe)
        merged = _merge_context(merged)
        kwargs["extra"] = merged
        return msg, kwargs


# ---------------------------
# Formatters
# ---------------------------
class PlainFormatter(logging.Formatter):
    DEFAULT = "%(asctime)s | %(levelname)s | %(processName)s[%(process)d] | %(threadName)s | %(name)s | %(message)s"
    def __init__(self, fmt: str = DEFAULT, datefmt: str = "%Y-%m-%d %H:%M:%S"):
        super().__init__(fmt=fmt, datefmt=datefmt)

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "process": {"pid": record.process, "name": record.processName},
            "thread": {"name": record.threadName},
            "msg": record.getMessage(),
        }
        # Merge structured extras
        for k, v in getattr(record, "extra", {}).__dict__.items() if hasattr(getattr(record, "extra", {}), "__dict__") else (getattr(record, "extra", {}) or {}).items():
            payload[k] = v
        # Standard extras (if present)
        for k in ("run_id", "well", "experiment_id", "job_id", "phase"):
            if getattr(record, k, None) is not None:
                payload[k] = getattr(record, k)
        # Attach pathname/line if error
        if record.levelno >= logging.ERROR:
            payload["where"] = f"{record.pathname}:{record.lineno}"
        return json.dumps(payload, ensure_ascii=False)

# ---------------------------
# Install / Configure
# ---------------------------
_listener: Optional[QueueListener] = None

def install_basic_config(
    level: int | str = logging.INFO,
    *,
    json_logs: bool = False,
    log_to_file: Optional[str] = None,
    use_queue: bool = False,
    propagate: bool = False,
) -> None:
    """
    Configure root logging once. Safe to call multiple times (idempotent).
    - json_logs: switch to JSON line formatter (good for ingestion).
    - log_to_file: path to write logs (in addition to stdout).
    - use_queue: enable QueueHandler/QueueListener for multi-process safety.
    """
    global _listener
    root = logging.getLogger()
    if isinstance(level, str):
        level = logging._nameToLevel.get(level.upper(), logging.INFO)
    root.setLevel(level)

    # Remove previous handlers to avoid duplicates in notebooks / reloads
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = JSONFormatter() if json_logs else PlainFormatter()
    handlers: Iterable[logging.Handler]

    if use_queue:
        # Centralize output in the main process
        q: queue.Queue = queue.Queue(-1)
        qh = QueueHandler(q)
        qh.setLevel(level)
        root.addHandler(qh)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        handlers = [stream_handler]

        if log_to_file:
            fh = logging.FileHandler(log_to_file, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(formatter)
            handlers = [stream_handler, fh]

        _listener = QueueListener(q, *handlers, respect_handler_level=True)
        _listener.start()
        atexit.register(_stop_listener)
    else:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

        if log_to_file:
            fh = logging.FileHandler(log_to_file, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(formatter)
            root.addHandler(fh)

    root.propagate = propagate

def _stop_listener():
    global _listener
    if _listener is not None:
        try:
            _listener.stop()
        finally:
            _listener = None

def get_logger(name: Optional[str] = None, *, context: Optional[Dict[str, Any]] = None) -> ContextAdapter:
    """Return a ContextAdapter that automatically injects contextvars, with optional fixed context."""
    base = logging.getLogger(name)
    return ContextAdapter(base, context or {})


# ---------------------------
# Context helpers
# ---------------------------
@contextmanager
def log_context(**kwargs: Any):
    """
    Temporarily attach structured fields to all log records within the block.
    Example:
        with log_context(run_id="R1", well="15/9-F-1"):
            logger.info("hello")  # => includes run_id & well
    """
    ctx = dict(_LOG_CONTEXT.get())
    ctx.update({k: v for k, v in kwargs.items() if v is not None})
    token = _LOG_CONTEXT.set(ctx)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)

# ---------------------------
# Phases & Timers
# ---------------------------
@dataclass
class PhaseResult:
    name: str
    status: str
    seconds: float

_PHASE_WIDTH = 100 

def box_log(logger: logging.Logger, title: str, lines: List[str], width: int = 102) -> None:
    """Logs a list of strings inside a formatted box."""
    top_bottom_border = "┌" + "─" * (width - 2) + "┐"
    middle_border = "├" + "─" * (width - 2) + "┤"
    
    def pad(s: str) -> str:
        return "│ " + s.ljust(width - 4) + " │"

    logger.info(top_bottom_border)
    logger.info(pad(title))
    logger.info(middle_border)
    for line in lines:
        logger.info(pad(line))
    logger.info("└" + "─" * (width - 2) + "┘")

def _format_phase_line(
    name: str,
    icon: str,
    label: str,
    duration: Optional[float] = None,
    status: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Helper to build a perfectly aligned log line for a phase, using spaces."""
    
    # 1. Monta a parte esquerda (fixa)
    left_part = f"{icon} {label}: {name}"
    
    # 2. Monta a parte direita (opcional)
    right_parts = []
    if duration is not None:
        right_parts.append(f"{duration:.3f}s")
    if status is not None:
        right_parts.append(f"[{status}]")
    if extra:
        extra_str = ", ".join(f"{k}={v}" for k, v in extra.items())
        right_parts.append(f"({extra_str})")
        
    right_part = " ".join(right_parts)
    
    # 3. Calcula o preenchimento dinâmico com ESPAÇOS
    current_len = len(left_part) + len(right_part)
    # Garante pelo menos um espaço de separação
    padding_needed = max(1, _PHASE_WIDTH - current_len)
    
    # A MUDANÇA ESTÁ AQUI: use ' ' em vez de '─'
    padding = " " * padding_needed 
    
    return f"{left_part}{padding}{right_part}"

@contextmanager
def phase(logger: logging.Logger, name: str, **extra):
    """
    Phase context with aligned timing and status reporting.
    Logs start and end messages in a clean, box-like format.
    """
    t0 = time.perf_counter()
    # Usa o formatador para a linha de início
    logger.info(_format_phase_line(name, "▶️", "START", extra=extra), extra=_merge_context({"phase": name, **extra}))
    
    status = "ok"
    icon = "✅"
    
    try:
        yield
    except KeyboardInterrupt:
        status = "interrupted"
        icon = "⏹️"
        logger.warning(
            _format_phase_line(name, icon, "FAIL", status=status, extra=extra),
            extra=_merge_context({"phase": name, "status": status, **extra})
        )
        raise
    except Exception:
        status = "error"
        icon = "💥"
        logger.exception(
            _format_phase_line(name, icon, "ERROR", status=status, extra=extra),
            extra=_merge_context({"phase": name, "status": status, **extra})
        )
        raise
    finally:
        dt = time.perf_counter() - t0
        # Usa o formatador para a linha de fim
        logger.info(
            _format_phase_line(name, icon, "END", duration=dt, status=status, extra=extra),
            extra=_merge_context({"phase": name, "status": status, **extra})
        )

# NOTA: Você precisará da função _merge_context que você já tem no seu código original.
# Se não tiver, aqui está uma implementação simples:
def _merge_context(d: dict) -> dict:
    # Apenas retorna o dict, pois o ContextAdapter fará a mesclagem.
    # No seu código, você pode ter uma lógica mais complexa, mantenha a sua.
    return d


# ---------------------------
# DA usage helper (purely informative)
# ---------------------------
def log_da_usage(logger: logging.Logger, used: bool, *, reason: str = "") -> None:
    """
    Log whether Data Augmentation (DA) was used.
    Example:
        log_da_usage(logger, used=(data_sample < 1.0), reason="downsample 50%")
    """
    msg = "Data Augmentation: USED" if used else "Data Augmentation: NOT USED"
    logger.info(msg, extra=_merge_context({"da_used": used, "da_reason": reason or None}))

# ---------------------------
# Multiprocessing initializer
# ---------------------------
def get_process_pool_initializer(level: int | str) -> tuple[Any, tuple]:
    """
    Returns (initializer, initargs) to pass into ProcessPoolExecutor so that
    each worker sets the desired log level and attaches minimal handlers.
    Usage:
        init_fn, init_args = get_process_pool_initializer(LOG_LEVEL)
        with ProcessPoolExecutor(..., initializer=init_fn, initargs=init_args) as pool:
            ...
    """
    if isinstance(level, str):
        level = logging._nameToLevel.get(level.upper(), logging.INFO)

    def _worker_init():
        # keep workers quiet unless requested
        root = logging.getLogger()
        root.setLevel(level)
        if not root.handlers:
            h = logging.StreamHandler(sys.stdout)
            h.setLevel(level)
            h.setFormatter(PlainFormatter())
            root.addHandler(h)
        root.propagate = False

    return _worker_init, tuple()

# ---------------------------
# Convenience: env-driven bootstrap
# ---------------------------
def bootstrap_from_env(prefix: str = "FP") -> None:
    """
    Initialize logging from environment variables:
      - FP_LOG_LEVEL=DEBUG|INFO|...
      - FP_LOG_JSON=1/0
      - FP_LOG_FILE=/path/to/file.log
      - FP_LOG_QUEUE=1/0
    """
    lvl = os.getenv(f"{prefix}_LOG_LEVEL", "INFO")
    json_on = os.getenv(f"{prefix}_LOG_JSON", "0") in ("1", "true", "TRUE")
    logfile = os.getenv(f"{prefix}_LOG_FILE") or None
    use_queue = os.getenv(f"{prefix}_LOG_QUEUE", "1") in ("1", "true", "TRUE")
    install_basic_config(level=lvl, json_logs=json_on, log_to_file=logfile, use_queue=use_queue)
