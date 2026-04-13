"""
Session analytics and personalization helpers.

Stores per-profile session history to JSON and computes recommended
work/break durations based on recent focus behavior.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionAnalyticsStore:
    """Persist session data and build personalized timing recommendations."""

    def __init__(self, base_dir: Optional[Path] = None, max_sessions: int = 300):
        self.base_dir = base_dir or Path("analytics") / "profiles"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_sessions = max_sessions

    @staticmethod
    def sanitize_profile_name(profile_name: str) -> str:
        """Convert profile name to a safe filename-friendly key."""
        raw = (profile_name or "").strip()
        if not raw:
            raw = "default"

        safe_chars: List[str] = []
        for ch in raw:
            if ch.isalnum() or ch in ("-", "_"):
                safe_chars.append(ch)
            elif ch.isspace():
                safe_chars.append("_")

        safe = "".join(safe_chars).strip("_")
        return safe or "default"

    def _profile_path(self, profile_name: str) -> Path:
        safe = self.sanitize_profile_name(profile_name)
        return self.base_dir / f"{safe}.json"

    @staticmethod
    def _default_profile(profile_name: str) -> Dict[str, Any]:
        return {
            "profile_name": profile_name,
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "sessions": [],
            "recommendation": {
                "work_minutes": 25,
                "break_minutes": 5,
                "confidence": 0.0,
                "reason": "No data yet",
                "based_on_sessions": 0,
            },
        }

    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """Load profile data from disk, creating defaults when missing."""
        path = self._profile_path(profile_name)
        if not path.exists():
            return self._default_profile(profile_name)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return self._default_profile(profile_name)
            data.setdefault("profile_name", profile_name)
            data.setdefault("sessions", [])
            data.setdefault("recommendation", self._default_profile(profile_name)["recommendation"])
            return data
        except Exception as exc:
            logger.warning("Failed to load profile analytics for '%s': %s", profile_name, exc)
            return self._default_profile(profile_name)

    def save_profile(self, profile_name: str, data: Dict[str, Any]) -> None:
        """Save profile data to disk."""
        data["updated_at"] = int(time.time())
        path = self._profile_path(profile_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_recommendation(
        self,
        profile_name: str,
        default_work: int = 25,
        default_break: int = 5,
    ) -> Dict[str, Any]:
        """Get the current recommendation for a profile."""
        profile = self.load_profile(profile_name)
        sessions = profile.get("sessions", [])
        recommendation = self._build_recommendation(
            sessions,
            default_work=default_work,
            default_break=default_break,
        )
        profile["recommendation"] = recommendation
        self.save_profile(profile_name, profile)
        return recommendation

    def record_session(
        self,
        profile_name: str,
        session_record: Dict[str, Any],
        default_work: int = 25,
        default_break: int = 5,
    ) -> Dict[str, Any]:
        """Append one session and recompute recommendation."""
        profile = self.load_profile(profile_name)
        sessions = profile.get("sessions", [])
        sessions.append(session_record)

        if len(sessions) > self.max_sessions:
            sessions = sessions[-self.max_sessions:]

        recommendation = self._build_recommendation(
            sessions,
            default_work=default_work,
            default_break=default_break,
        )

        profile["sessions"] = sessions
        profile["recommendation"] = recommendation
        self.save_profile(profile_name, profile)

        return recommendation

    def _build_recommendation(
        self,
        sessions: List[Dict[str, Any]],
        default_work: int,
        default_break: int,
    ) -> Dict[str, Any]:
        """
        Build recommendation from recent sessions.

        Uses focus ratio, average score, and distraction density
        to choose personalized work/break timing.
        """
        default_work = int(max(15, min(60, default_work)))
        default_break = int(max(3, min(20, default_break)))

        valid_sessions: List[Dict[str, Any]] = []
        for sess in sessions[-30:]:
            duration = float(sess.get("session_seconds", 0.0))
            if duration >= 5 * 60:
                valid_sessions.append(sess)

        if not valid_sessions:
            return {
                "work_minutes": default_work,
                "break_minutes": default_break,
                "confidence": 0.0,
                "reason": "Not enough history yet",
                "based_on_sessions": 0,
            }

        focus_ratios: List[float] = []
        avg_scores: List[float] = []
        distractions_per_hour: List[float] = []

        for sess in valid_sessions:
            duration = max(1.0, float(sess.get("session_seconds", 1.0)))
            focus_seconds = max(0.0, float(sess.get("focus_seconds", 0.0)))
            focus_ratios.append(min(1.0, focus_seconds / duration))
            avg_scores.append(float(sess.get("avg_score", 0.0)))

            distractions = max(0.0, float(sess.get("distraction_count", 0.0)))
            distractions_per_hour.append(distractions / (duration / 3600.0))

        avg_focus_ratio = statistics.fmean(focus_ratios)
        avg_score = statistics.fmean(avg_scores)
        avg_distraction_per_hour = statistics.fmean(distractions_per_hour)

        # Base recommendation from sustained focus quality.
        if avg_focus_ratio >= 0.82 and avg_distraction_per_hour <= 1.5 and avg_score >= 85:
            work_minutes = 45
        elif avg_focus_ratio >= 0.74 and avg_distraction_per_hour <= 2.5 and avg_score >= 78:
            work_minutes = 35
        elif avg_focus_ratio >= 0.64 and avg_distraction_per_hour <= 4.0:
            work_minutes = 30
        elif avg_focus_ratio >= 0.54:
            work_minutes = 25
        else:
            work_minutes = 20

        # Break duration based on fatigue/distraction pressure.
        if avg_distraction_per_hour > 6.0 or avg_score < 55:
            break_minutes = 10
        elif avg_distraction_per_hour > 4.0 or avg_score < 65:
            break_minutes = 8
        elif avg_distraction_per_hour > 2.5 or avg_score < 75:
            break_minutes = 6
        else:
            break_minutes = 5

        work_minutes = int(max(15, min(60, work_minutes)))
        break_minutes = int(max(3, min(20, break_minutes)))

        confidence = min(1.0, len(valid_sessions) / 12.0)
        reason = (
            f"focus_ratio={avg_focus_ratio:.0%}, avg_score={avg_score:.1f}, "
            f"distractions/hour={avg_distraction_per_hour:.1f}"
        )

        return {
            "work_minutes": work_minutes,
            "break_minutes": break_minutes,
            "confidence": confidence,
            "reason": reason,
            "based_on_sessions": len(valid_sessions),
        }
