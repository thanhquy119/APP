"""
Task-context monitoring and classification helpers.

This module provides a privacy-first layer that samples foreground application
metadata (window title + process), classifies work context, and computes a
compact risk summary for runtime decisions.
"""

from __future__ import annotations

import ctypes
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from ctypes import wintypes

logger = logging.getLogger(__name__)


try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None


PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

IS_WINDOWS_CONTEXT_AVAILABLE = False
_user32 = None
_kernel32 = None

try:
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _user32.GetForegroundWindow.restype = wintypes.HWND
    _user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    _user32.GetWindowTextLengthW.restype = ctypes.c_int
    _user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _user32.GetWindowTextW.restype = ctypes.c_int
    _user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    _user32.GetWindowThreadProcessId.restype = wintypes.DWORD

    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    _kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL

    IS_WINDOWS_CONTEXT_AVAILABLE = True
except Exception as exc:  # pragma: no cover - non-Windows fallback
    logger.warning("Task context Windows API is not available: %s", exc)


DEFAULT_TASK_KEYWORDS = (
    "code",
    "visual studio",
    "pycharm",
    "cursor",
    "terminal",
    "notion",
    "word",
    "excel",
    "powerpoint",
    "docs",
    "sheets",
    "slides",
    "research",
    "study",
    "learning",
    "jira",
    "trello",
    "slack",
    "teams",
    "outlook",
    "mail",
    "github",
    "gitlab",
)

DEFAULT_DISTRACTING_KEYWORDS = (
    "youtube",
    "facebook",
    "instagram",
    "tiktok",
    "netflix",
    "game",
    "steam",
    "discord",
    "reddit",
    "news",
    "shopping",
)

DEFAULT_NEUTRAL_KEYWORDS = (
    "explorer",
    "setting",
    "settings",
    "control panel",
    "task manager",
    "file",
)

DEFAULT_EXCLUDED_KEYWORDS = (
    "focusguardian",
    "notification",
    "toast",
)


@dataclass
class TaskContextSample:
    """One foreground-window sample with classification metadata."""

    timestamp: float
    window_title: str
    process_name: str
    process_id: int
    app_id: str
    category: str = "unknown"
    confidence: float = 0.0
    reason: str = ""

    def to_privacy_safe_dict(self) -> Dict[str, Any]:
        """Serialize metadata without raw window title for persistence/sync."""
        return {
            "timestamp": float(self.timestamp),
            "process_name": str(self.process_name or ""),
            "process_id": int(self.process_id or 0),
            "app_id": str(self.app_id or ""),
            "category": str(self.category or "unknown"),
            "confidence": float(max(0.0, min(1.0, self.confidence))),
            "reason": str(self.reason or ""),
        }


@dataclass
class TaskContextConfig:
    """Runtime configuration for context sampling and classification."""

    enabled: bool = True
    sample_interval_seconds: float = 5.0
    lookback_seconds: float = 300.0
    max_samples: int = 2400

    task_keywords: Tuple[str, ...] = tuple(DEFAULT_TASK_KEYWORDS)
    distracting_keywords: Tuple[str, ...] = tuple(DEFAULT_DISTRACTING_KEYWORDS)
    neutral_keywords: Tuple[str, ...] = tuple(DEFAULT_NEUTRAL_KEYWORDS)

    task_apps: Tuple[str, ...] = tuple()
    distracting_apps: Tuple[str, ...] = tuple()

    excluded_keywords: Tuple[str, ...] = tuple(DEFAULT_EXCLUDED_KEYWORDS)
    excluded_apps: Tuple[str, ...] = tuple()


@dataclass
class TaskContextStats:
    """Aggregated context quality for a recent window."""

    total_samples: int = 0
    task_related_samples: int = 0
    distracting_samples: int = 0
    neutral_samples: int = 0
    unknown_samples: int = 0

    task_alignment_ratio: float = 0.0
    distracting_ratio: float = 0.0
    neutral_ratio: float = 0.0
    unknown_ratio: float = 0.0

    risk_score: float = 0.0
    context_switch_count: int = 0
    current_category: str = "unknown"
    current_app_id: str = ""
    updated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": int(self.total_samples),
            "task_related_samples": int(self.task_related_samples),
            "distracting_samples": int(self.distracting_samples),
            "neutral_samples": int(self.neutral_samples),
            "unknown_samples": int(self.unknown_samples),
            "task_alignment_ratio": float(self.task_alignment_ratio),
            "distracting_ratio": float(self.distracting_ratio),
            "neutral_ratio": float(self.neutral_ratio),
            "unknown_ratio": float(self.unknown_ratio),
            "risk_score": float(self.risk_score),
            "context_switch_count": int(self.context_switch_count),
            "current_category": str(self.current_category or "unknown"),
            "current_app_id": str(self.current_app_id or ""),
            "updated_at": float(self.updated_at or 0.0),
        }


class TaskContextMonitor:
    """Read active foreground window metadata on Windows."""

    def get_active_context(self, timestamp: Optional[float] = None) -> TaskContextSample:
        now_ts = float(timestamp if timestamp is not None else time.time())

        if not IS_WINDOWS_CONTEXT_AVAILABLE or _user32 is None:
            return TaskContextSample(
                timestamp=now_ts,
                window_title="",
                process_name="",
                process_id=0,
                app_id="unknown",
                category="unknown",
                confidence=0.0,
                reason="Windows foreground-window API unavailable",
            )

        title = ""
        process_id = 0
        process_name = ""

        try:
            hwnd = _user32.GetForegroundWindow()
            if hwnd:
                title = self._read_window_title(hwnd)
                process_id = self._read_process_id(hwnd)
                process_name = self._resolve_process_name(process_id)
        except Exception as exc:
            logger.debug("Failed to query active window context: %s", exc)

        app_id = self._build_app_id(process_name=process_name, process_id=process_id)

        return TaskContextSample(
            timestamp=now_ts,
            window_title=title,
            process_name=process_name,
            process_id=process_id,
            app_id=app_id,
        )

    @staticmethod
    def _build_app_id(process_name: str, process_id: int) -> str:
        name = str(process_name or "").strip().lower()
        if name:
            return name
        if process_id > 0:
            return f"pid_{int(process_id)}"
        return "unknown"

    @staticmethod
    def _read_window_title(hwnd: int) -> str:
        if _user32 is None:
            return ""
        length = int(_user32.GetWindowTextLengthW(hwnd))
        if length <= 0:
            return ""

        buff = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buff, len(buff))
        return str(buff.value or "").strip()

    @staticmethod
    def _read_process_id(hwnd: int) -> int:
        if _user32 is None:
            return 0
        process_id = wintypes.DWORD(0)
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        return int(process_id.value or 0)

    @staticmethod
    def _resolve_process_name(process_id: int) -> str:
        if process_id <= 0:
            return ""

        if psutil is not None:
            try:
                proc = psutil.Process(process_id)
                return str(proc.name() or "").strip().lower()
            except Exception:
                pass

        if not IS_WINDOWS_CONTEXT_AVAILABLE or _kernel32 is None:
            return ""

        handle = _kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, process_id)
        if not handle:
            return ""

        try:
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            ok = _kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size))
            if not ok:
                return ""

            path = Path(str(buffer.value or "").strip())
            return path.name.lower()
        except Exception:
            return ""
        finally:
            _kernel32.CloseHandle(handle)


class TaskContextClassifier:
    """Classify active-window samples into task-related context states."""

    def __init__(self, config: Optional[TaskContextConfig] = None):
        self.config = config or TaskContextConfig()
        self._samples: Deque[TaskContextSample] = deque(maxlen=self.config.max_samples)
        self._last_logged_signature: str = ""
        self._last_log_at: float = 0.0

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _normalize_tokens(raw: Any) -> Tuple[str, ...]:
        if raw is None:
            return tuple()

        values: List[str] = []
        if isinstance(raw, str):
            parts = raw.replace("\n", ",").replace(";", ",").split(",")
            values = [p.strip().lower() for p in parts]
        elif isinstance(raw, (list, tuple, set)):
            for item in raw:
                token = str(item or "").strip().lower()
                if token:
                    values.append(token)
        else:
            token = str(raw or "").strip().lower()
            if token:
                values.append(token)

        deduped: List[str] = []
        seen: set[str] = set()
        for token in values:
            if not token or token in seen:
                continue
            seen.add(token)
            deduped.append(token)
        return tuple(deduped)

    @staticmethod
    def _contains_any(text: str, candidates: Sequence[str]) -> bool:
        body = str(text or "").strip().lower()
        if not body:
            return False
        for item in candidates:
            token = str(item or "").strip().lower()
            if token and token in body:
                return True
        return False

    @staticmethod
    def _matches_app(sample: TaskContextSample, app_rules: Sequence[str]) -> bool:
        process_name = str(sample.process_name or "").strip().lower()
        app_id = str(sample.app_id or "").strip().lower()

        for rule in app_rules:
            token = str(rule or "").strip().lower()
            if not token:
                continue
            if token == process_name or token == app_id:
                return True
            if token in process_name or token in app_id:
                return True
        return False

    def update_from_app_config(self, app_config: Dict[str, Any]) -> None:
        """Update classifier config from persisted app settings."""
        data = dict(app_config or {})

        try:
            sample_interval = float(data.get("task_context_sample_interval_seconds", 5.0) or 5.0)
        except (TypeError, ValueError):
            sample_interval = 5.0

        try:
            lookback_minutes = float(data.get("task_context_lookback_minutes", 5.0) or 5.0)
        except (TypeError, ValueError):
            lookback_minutes = 5.0

        try:
            max_samples = int(float(data.get("task_context_max_samples", self.config.max_samples) or self.config.max_samples))
        except (TypeError, ValueError):
            max_samples = self.config.max_samples

        next_config = TaskContextConfig(
            enabled=bool(data.get("enable_task_context_monitoring", True)),
            sample_interval_seconds=max(2.0, min(30.0, sample_interval)),
            lookback_seconds=max(60.0, min(3600.0, lookback_minutes * 60.0)),
            max_samples=max(120, min(12000, max_samples)),
            task_keywords=self._normalize_tokens(data.get("task_context_task_keywords", self.config.task_keywords))
            or tuple(DEFAULT_TASK_KEYWORDS),
            distracting_keywords=self._normalize_tokens(
                data.get("task_context_distracting_keywords", self.config.distracting_keywords)
            )
            or tuple(DEFAULT_DISTRACTING_KEYWORDS),
            neutral_keywords=self._normalize_tokens(data.get("task_context_neutral_keywords", self.config.neutral_keywords))
            or tuple(DEFAULT_NEUTRAL_KEYWORDS),
            task_apps=self._normalize_tokens(data.get("task_context_task_apps", self.config.task_apps)),
            distracting_apps=self._normalize_tokens(
                data.get("task_context_distracting_apps", self.config.distracting_apps)
            ),
            excluded_keywords=self._normalize_tokens(
                data.get("task_context_excluded_keywords", self.config.excluded_keywords)
            )
            or tuple(DEFAULT_EXCLUDED_KEYWORDS),
            excluded_apps=self._normalize_tokens(data.get("task_context_excluded_apps", self.config.excluded_apps)),
        )

        self.config = next_config

        if self._samples.maxlen != self.config.max_samples:
            self._samples = deque(self._samples, maxlen=self.config.max_samples)

    def classify(self, sample: TaskContextSample) -> Tuple[str, float, str]:
        """Return category, confidence, and a short rule trace."""
        title = str(sample.window_title or "").strip().lower()

        if self._matches_app(sample, self.config.excluded_apps) or self._contains_any(title, self.config.excluded_keywords):
            return "excluded", 0.0, "excluded rule"

        if self._matches_app(sample, self.config.distracting_apps):
            return "distracting", 0.95, "distracting app"

        if self._contains_any(title, self.config.distracting_keywords):
            return "distracting", 0.88, "distracting keyword"

        if self._matches_app(sample, self.config.task_apps):
            return "task_related", 0.92, "task app"

        if self._contains_any(title, self.config.task_keywords):
            return "task_related", 0.82, "task keyword"

        if self._contains_any(title, self.config.neutral_keywords):
            return "neutral", 0.68, "neutral keyword"

        if not title and not str(sample.process_name or "").strip():
            return "unknown", 0.15, "empty context"

        if str(sample.process_name or "").strip():
            return "unknown", 0.45, "unmapped app"

        return "unknown", 0.28, "insufficient data"

    def annotate(self, sample: TaskContextSample) -> TaskContextSample:
        category, confidence, reason = self.classify(sample)
        sample.category = category
        sample.confidence = self._clamp(float(confidence), 0.0, 1.0)
        sample.reason = str(reason or "")

        self._samples.append(sample)
        self._log_active_context(sample)
        return sample

    def _log_active_context(self, sample: TaskContextSample) -> None:
        """Log active app/window title so developers can verify context detection.

        Browser APIs are not used here, so this shows the active window title
        (often the active browser tab title), not every open browser tab.
        """
        now = float(sample.timestamp or time.time())
        process = str(sample.process_name or "unknown").strip() or "unknown"
        title = str(sample.window_title or "").strip()
        category = str(sample.category or "unknown").strip() or "unknown"
        reason = str(sample.reason or "").strip()
        confidence = self._clamp(float(sample.confidence or 0.0), 0.0, 1.0)

        signature = f"{process}|{title}|{category}|{reason}"
        # Log on context changes, and also periodically so long sessions are visible in focusguardian.log.
        if signature == self._last_logged_signature and (now - self._last_log_at) < 15.0:
            return

        self._last_logged_signature = signature
        self._last_log_at = now
        logger.debug(
            "TaskContext active_window process=%s category=%s confidence=%.2f reason=%s title=%r",
            process,
            category,
            confidence,
            reason,
            title[:160],
        )

    def clear_samples(self) -> None:
        self._samples.clear()

    def recent_samples(self, now: Optional[float] = None) -> List[TaskContextSample]:
        if not self._samples:
            return []

        now_ts = float(now if now is not None else time.time())
        lookback = max(30.0, float(self.config.lookback_seconds))
        oldest_ts = now_ts - lookback

        return [s for s in self._samples if float(s.timestamp) >= oldest_ts]

    def compute_stats(self, now: Optional[float] = None) -> TaskContextStats:
        recent = self.recent_samples(now=now)
        if not recent:
            return TaskContextStats(updated_at=float(now if now is not None else time.time()))

        counts = {
            "task_related": 0,
            "distracting": 0,
            "neutral": 0,
            "unknown": 0,
        }
        valid_samples = 0
        switch_count = 0
        last_app = ""

        for item in recent:
            category = str(item.category or "unknown").strip().lower()
            if category == "excluded":
                continue

            valid_samples += 1
            if category not in counts:
                category = "unknown"
            counts[category] += 1

            current_app = str(item.app_id or "").strip().lower()
            if current_app:
                if last_app and current_app != last_app:
                    switch_count += 1
                last_app = current_app

        denominator = max(1, valid_samples)
        task_ratio = counts["task_related"] / denominator
        distracting_ratio = counts["distracting"] / denominator
        neutral_ratio = counts["neutral"] / denominator
        unknown_ratio = counts["unknown"] / denominator

        risk = (
            (distracting_ratio * 0.62)
            + ((1.0 - task_ratio) * 0.28)
            + (unknown_ratio * 0.10)
        )
        current_category = str(recent[-1].category or "unknown").strip().lower()
        if current_category == "distracting":
            risk = max(risk, 0.72)
        elif current_category == "task_related":
            risk = min(risk, 0.45)

        risk = self._clamp(risk, 0.0, 1.0)

        return TaskContextStats(
            total_samples=int(valid_samples),
            task_related_samples=int(counts["task_related"]),
            distracting_samples=int(counts["distracting"]),
            neutral_samples=int(counts["neutral"]),
            unknown_samples=int(counts["unknown"]),
            task_alignment_ratio=float(task_ratio),
            distracting_ratio=float(distracting_ratio),
            neutral_ratio=float(neutral_ratio),
            unknown_ratio=float(unknown_ratio),
            risk_score=float(risk),
            context_switch_count=int(switch_count),
            current_category=str(recent[-1].category or "unknown"),
            current_app_id=str(recent[-1].app_id or ""),
            updated_at=float(recent[-1].timestamp),
        )

    @staticmethod
    def summarize_for_report(stats: TaskContextStats) -> Dict[str, Any]:
        """Return compact, privacy-safe summary for analytics and messaging."""
        return {
            "samples": int(stats.total_samples),
            "task_alignment_ratio": float(stats.task_alignment_ratio),
            "distracting_ratio": float(stats.distracting_ratio),
            "risk_score": float(stats.risk_score),
            "context_switch_count": int(stats.context_switch_count),
            "current_category": str(stats.current_category or "unknown"),
            "current_app_id": str(stats.current_app_id or ""),
            "updated_at": float(stats.updated_at or 0.0),
        }
