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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .supabase_sync import SupabaseSessionSync
from .personalization import (
    PersonalizationManager,
    UserBaseline,
    UserBaselineStore,
    is_session_eligible_for_personalization,
    personalization_stage,
    science_informed_break_minutes,
    session_personalization_weight,
)

logger = logging.getLogger(__name__)

FOCUSED_STATE_NAMES = {"ON_SCREEN_READING", "OFFSCREEN_WRITING"}


class SessionAnalyticsStore:
    """Persist session data and build personalized timing recommendations."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        max_sessions: int = 300,
        cloud_config: Optional[Dict[str, Any]] = None,
    ):
        self.base_dir = base_dir or Path("analytics") / "profiles"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_sessions = max_sessions

        self.baseline_store = UserBaselineStore()
        self.personalization_manager = PersonalizationManager(self.baseline_store)

        self.supabase_sync = SupabaseSessionSync()
        if cloud_config:
            self.configure_supabase(cloud_config)

    def configure_supabase(self, app_config: Dict[str, Any]) -> None:
        """Apply Supabase sync settings from app config."""
        self.supabase_sync.configure_from_app_config(app_config)

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

    @staticmethod
    def _sessions_after_baseline_reset(
        profile: Dict[str, Any],
        sessions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Keep old history for reports but exclude it from a reset baseline."""
        try:
            reset_at = int(float(profile.get("baseline_reset_at", 0) or 0))
        except (TypeError, ValueError):
            reset_at = 0
        if reset_at <= 0:
            return list(sessions or [])

        filtered: List[Dict[str, Any]] = []
        for session in sessions or []:
            if not isinstance(session, dict):
                continue
            try:
                ts = int(float(session.get("timestamp", 0) or 0))
            except (TypeError, ValueError):
                ts = 0
            if ts >= reset_at:
                filtered.append(session)
        return filtered

    def get_recommendation(
        self,
        profile_name: str,
        default_work: int = 25,
        default_break: int = 5,
        minutes_since_last_break: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get the current recommendation for a profile."""
        bundle = self.get_personalization_bundle(
            profile_name,
            default_work=default_work,
            default_break=default_break,
            minutes_since_last_break=minutes_since_last_break,
        )
        return bundle.get("recommendation", {})

    def get_personalization_bundle(
        self,
        profile_name: str,
        default_work: int = 25,
        default_break: int = 5,
        minutes_since_last_break: Optional[float] = None,
        focus_engine_defaults: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build recommendation + baseline + threshold bundle for one profile.

        This is the main integration point for UI/engine personalization flow.
        """
        profile = self.load_profile(profile_name)
        all_sessions = profile.get("sessions", [])
        sessions = self._sessions_after_baseline_reset(profile, all_sessions)

        baseline = self._load_or_refresh_baseline(
            profile_name=profile_name,
            sessions=sessions,
            default_work=default_work,
            default_break=default_break,
            reset_at=profile.get("baseline_reset_at"),
        )

        recommendation = self._build_recommendation(
            sessions,
            default_work=default_work,
            default_break=default_break,
            baseline=baseline,
            minutes_since_last_break=minutes_since_last_break,
        )

        thresholds = self.personalization_manager.build_thresholds(
            profile_name=profile_name,
            baseline=baseline,
            focus_defaults=focus_engine_defaults,
        )

        profile["recommendation"] = recommendation
        profile["baseline"] = baseline.to_dict()
        self.save_profile(profile_name, profile)

        return {
            "recommendation": recommendation,
            "baseline": baseline.to_dict(),
            "thresholds": thresholds.to_dict(),
        }

    def get_user_baseline(
        self,
        profile_name: str,
        default_work: int = 25,
        default_break: int = 5,
    ) -> Dict[str, Any]:
        profile = self.load_profile(profile_name)
        sessions = self._sessions_after_baseline_reset(profile, profile.get("sessions", []))
        baseline = self._load_or_refresh_baseline(
            profile_name=profile_name,
            sessions=sessions,
            default_work=default_work,
            default_break=default_break,
            reset_at=profile.get("baseline_reset_at"),
        )
        return baseline.to_dict()

    def get_personalized_thresholds(
        self,
        profile_name: str,
        default_work: int = 25,
        default_break: int = 5,
        focus_engine_defaults: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        profile = self.load_profile(profile_name)
        sessions = self._sessions_after_baseline_reset(profile, profile.get("sessions", []))
        baseline = self._load_or_refresh_baseline(
            profile_name=profile_name,
            sessions=sessions,
            default_work=default_work,
            default_break=default_break,
            reset_at=profile.get("baseline_reset_at"),
        )
        thresholds = self.personalization_manager.build_thresholds(
            profile_name=profile_name,
            baseline=baseline,
            focus_defaults=focus_engine_defaults,
        )
        return thresholds.to_dict()

    def get_personalization_status(self, profile_name: str) -> Dict[str, Any]:
        """Return lightweight baseline status for UI."""
        profile = self.load_profile(profile_name)
        sessions = self._sessions_after_baseline_reset(profile, profile.get("sessions", []))
        eligible_count = 0
        for session in sessions:
            duration = self._safe_float(session.get("session_seconds_cleaned"))
            if duration is None:
                duration = self._safe_float(session.get("session_seconds"))
            if duration is not None and is_session_eligible_for_personalization(session, duration):
                eligible_count += 1

        recommendation = dict(profile.get("recommendation", {}) or {})
        baseline_payload = dict(profile.get("baseline", {}) or {})
        stage = str(
            recommendation.get("adaptation_stage")
            or baseline_payload.get("adaptation_stage")
            or personalization_stage(eligible_count)
        )
        if eligible_count < 3:
            label = "Chưa đủ dữ liệu"
            stage = "cold_start"
        elif eligible_count <= 7:
            label = "Đang học"
            stage = "hybrid"
        else:
            label = "Đã có baseline tạm"
            stage = "personalized"

        try:
            confidence = float(recommendation.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        return {
            "profile_name": profile_name,
            "label": label,
            "stage": stage,
            "eligible_sessions": int(eligible_count),
            "based_on_sessions": int(recommendation.get("based_on_sessions", eligible_count) or eligible_count),
            "confidence": max(0.0, min(1.0, confidence)),
            "baseline_reset_at": int(profile.get("baseline_reset_at", 0) or 0),
        }

    def reset_profile_baseline(
        self,
        profile_name: str,
        *,
        default_work: int = 25,
        default_break: int = 5,
    ) -> Dict[str, Any]:
        """Reset only the personalization baseline/recommendation for one profile."""
        profile_name = (profile_name or "default").strip() or "default"
        now = int(time.time())
        profile = self.load_profile(profile_name)
        safe_work = int(max(15, min(60, int(default_work or 25))))
        safe_break = int(max(3, min(20, int(default_break or 5))))
        baseline = UserBaseline(
            profile_name=profile_name,
            session_count=0,
            recommended_work_minutes=safe_work,
            recommended_break_minutes=safe_break,
            personalization_weight=0.0,
            last_quality_score=0.0,
            updated_at=now,
        )
        recommendation = {
            "work_minutes": safe_work,
            "break_minutes": safe_break,
            "confidence": 0.0,
            "reason": "Baseline reset for current profile",
            "based_on_sessions": 0,
            "adaptation_stage": "cold_start",
        }

        profile["baseline_reset_at"] = now
        profile["baseline"] = baseline.to_dict()
        profile["recommendation"] = recommendation
        self.save_profile(profile_name, profile)
        self.baseline_store.save_baseline(baseline)

        baseline_payload = baseline.to_dict()
        baseline_payload["adaptation_stage"] = "cold_start"
        self.supabase_sync.upsert_user_baseline(baseline_payload)
        return self.get_personalization_status(profile_name)

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _merge_adjacent_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        for seg in segments:
            state = str(seg.get("state", "")).strip()
            seconds = float(seg.get("seconds", 0.0) or 0.0)
            reason_type = str(seg.get("uncertain_reason_type", "") or "").strip().lower()
            if not state or seconds <= 0.0:
                continue

            if (
                merged
                and merged[-1]["state"] == state
                and str(merged[-1].get("uncertain_reason_type", "") or "").strip().lower() == reason_type
            ):
                merged[-1]["seconds"] += seconds
            else:
                merged.append(
                    {
                        "state": state,
                        "seconds": seconds,
                        "uncertain_reason_type": reason_type,
                    }
                )
        return merged

    def _normalize_state_segments(self, state_segments: Any) -> List[Dict[str, Any]]:
        """
        Normalize and denoise short UNCERTAIN segments.

        Rule priority:
        - Merge adjacent segments with same state
        - Merge short UNCERTAIN (<~1.8s) between two same focused states
        - Re-attach short measurement-noise UNCERTAIN to nearby focused state
        """
        parsed: List[Dict[str, Any]] = []
        if isinstance(state_segments, list):
            for raw in state_segments:
                if not isinstance(raw, dict):
                    continue
                state = str(raw.get("state", "")).strip()
                seconds = self._safe_float(raw.get("seconds"))
                reason_type = str(raw.get("uncertain_reason_type", "") or "").strip().lower()
                if not state or seconds is None or seconds <= 0.0:
                    continue
                parsed.append(
                    {
                        "state": state,
                        "seconds": float(seconds),
                        "uncertain_reason_type": reason_type,
                    }
                )

        if not parsed:
            return []

        segments = self._merge_adjacent_segments(parsed)
        idx = 1
        short_uncertain_limit = 1.8

        while idx < len(segments) - 1:
            current = segments[idx]
            if current["state"] == "UNCERTAIN" and current["seconds"] <= short_uncertain_limit:
                prev_seg = segments[idx - 1]
                next_seg = segments[idx + 1]

                if prev_seg["state"] == next_seg["state"] and prev_seg["state"] in FOCUSED_STATE_NAMES:
                    prev_seg["seconds"] += current["seconds"] + next_seg["seconds"]
                    segments.pop(idx + 1)
                    segments.pop(idx)
                    segments = self._merge_adjacent_segments(segments)
                    idx = max(1, idx - 1)
                    continue

                if current.get("uncertain_reason_type") == "measurement_noise":
                    if prev_seg["state"] in FOCUSED_STATE_NAMES and next_seg["state"] in FOCUSED_STATE_NAMES:
                        if prev_seg["seconds"] >= next_seg["seconds"]:
                            prev_seg["seconds"] += current["seconds"]
                        else:
                            next_seg["seconds"] += current["seconds"]
                        segments.pop(idx)
                        segments = self._merge_adjacent_segments(segments)
                        idx = max(1, idx - 1)
                        continue

                    if prev_seg["state"] in FOCUSED_STATE_NAMES:
                        prev_seg["seconds"] += current["seconds"]
                        segments.pop(idx)
                        segments = self._merge_adjacent_segments(segments)
                        idx = max(1, idx - 1)
                        continue

                    if next_seg["state"] in FOCUSED_STATE_NAMES:
                        next_seg["seconds"] += current["seconds"]
                        segments.pop(idx)
                        segments = self._merge_adjacent_segments(segments)
                        idx = max(1, idx - 1)
                        continue

            idx += 1

        return self._merge_adjacent_segments(segments)

    @staticmethod
    def _aggregate_state_seconds(segments: List[Dict[str, Any]]) -> Dict[str, float]:
        aggregated: Dict[str, float] = {}
        for seg in segments:
            state = str(seg.get("state", "")).strip()
            seconds = float(seg.get("seconds", 0.0) or 0.0)
            if not state or seconds <= 0.0:
                continue
            aggregated[state] = aggregated.get(state, 0.0) + seconds
        return aggregated

    def _compute_analytics_quality_score(
        self,
        session_seconds: float,
        face_presence_ratio: float,
        uncertain_seconds_raw: float,
        uncertain_seconds_cleaned: float,
    ) -> float:
        duration_score = self._clamp(session_seconds / 1200.0, 0.0, 1.0)
        face_score = self._clamp(face_presence_ratio, 0.0, 1.0)
        uncertain_ratio_raw = uncertain_seconds_raw / max(1e-6, session_seconds)
        uncertainty_score = 1.0 - self._clamp(uncertain_ratio_raw, 0.0, 1.0)

        recovered_ratio = 0.0
        if uncertain_seconds_raw > 1e-6:
            recovered_ratio = self._clamp(
                (uncertain_seconds_raw - uncertain_seconds_cleaned) / uncertain_seconds_raw,
                0.0,
                1.0,
            )

        quality = (
            (duration_score * 0.25)
            + (face_score * 0.4)
            + (uncertainty_score * 0.25)
            + (recovered_ratio * 0.1)
        )
        return self._clamp(quality, 0.0, 1.0)

    def _clean_session_record_for_analytics(self, session_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean noisy UNCERTAIN spans so analytics/baseline learn from stable behavior.

        Raw fields are preserved for debugging and cloud observability.
        """
        raw = dict(session_record)

        session_seconds = float(raw.get("session_seconds", 0.0) or 0.0)
        if session_seconds <= 0.0:
            return raw

        raw_state_seconds_payload = raw.get("state_seconds", {}) or {}
        raw_state_seconds: Dict[str, float] = {}
        for state, seconds in raw_state_seconds_payload.items():
            try:
                raw_state_seconds[str(state)] = max(0.0, float(seconds or 0.0))
            except (TypeError, ValueError):
                continue

        raw_focus_from_states = (
            raw_state_seconds.get("ON_SCREEN_READING", 0.0)
            + raw_state_seconds.get("OFFSCREEN_WRITING", 0.0)
        )
        focus_seconds_raw = float(raw.get("focus_seconds", raw_focus_from_states) or raw_focus_from_states)
        uncertain_seconds_raw = float(raw_state_seconds.get("UNCERTAIN", 0.0) or 0.0)

        face_presence_ratio = self._safe_float(raw.get("face_presence_ratio"))
        if face_presence_ratio is None:
            face_presence_ratio = 1.0
        face_presence_ratio = self._clamp(face_presence_ratio, 0.0, 1.0)

        uncertain_noise_seconds = max(0.0, float(raw.get("uncertain_measurement_noise_seconds", 0.0) or 0.0))
        uncertain_behavioral_seconds = max(0.0, float(raw.get("uncertain_behavioral_seconds", 0.0) or 0.0))

        segments = self._normalize_state_segments(raw.get("state_segments"))
        cleaned_state_seconds = dict(raw_state_seconds)
        if segments:
            cleaned_state_seconds = self._aggregate_state_seconds(segments)
            for state in raw_state_seconds:
                cleaned_state_seconds.setdefault(state, 0.0)

            raw_coverage = sum(float(v or 0.0) for v in raw_state_seconds.values())
            cleaned_coverage = sum(float(v or 0.0) for v in cleaned_state_seconds.values())
            if cleaned_coverage < max(1.0, raw_coverage * 0.75):
                # Segment stream may be incomplete (e.g. startup window excluded) -> keep raw coverage.
                cleaned_state_seconds = dict(raw_state_seconds)

        uncertain_seconds_cleaned = max(0.0, float(cleaned_state_seconds.get("UNCERTAIN", 0.0) or 0.0))

        # If uncertainty is tiny and face is stable, treat most of it as measurement noise.
        uncertain_ratio_raw = uncertain_seconds_raw / max(1e-6, session_seconds)
        if uncertain_seconds_cleaned > 0.0 and uncertain_ratio_raw <= 0.08 and face_presence_ratio >= 0.72:
            dominant_focused_state = (
                "ON_SCREEN_READING"
                if cleaned_state_seconds.get("ON_SCREEN_READING", 0.0)
                >= cleaned_state_seconds.get("OFFSCREEN_WRITING", 0.0)
                else "OFFSCREEN_WRITING"
            )

            noise_ratio = 0.0
            if uncertain_seconds_raw > 1e-6:
                noise_ratio = self._clamp(uncertain_noise_seconds / uncertain_seconds_raw, 0.0, 1.0)

            transfer_ratio = max(0.65, noise_ratio if noise_ratio > 0.0 else 0.7)
            transfer_seconds = uncertain_seconds_cleaned * transfer_ratio
            cleaned_state_seconds["UNCERTAIN"] = max(0.0, uncertain_seconds_cleaned - transfer_seconds)
            cleaned_state_seconds[dominant_focused_state] = (
                cleaned_state_seconds.get(dominant_focused_state, 0.0) + transfer_seconds
            )
            uncertain_seconds_cleaned = cleaned_state_seconds["UNCERTAIN"]

        focus_seconds_cleaned = (
            cleaned_state_seconds.get("ON_SCREEN_READING", 0.0)
            + cleaned_state_seconds.get("OFFSCREEN_WRITING", 0.0)
        )
        focus_seconds_cleaned = self._clamp(focus_seconds_cleaned, 0.0, session_seconds)

        avg_score_raw = float(raw.get("avg_score", 0.0) or 0.0)
        recovered_uncertain_seconds = max(0.0, uncertain_seconds_raw - uncertain_seconds_cleaned)
        recovered_ratio = recovered_uncertain_seconds / max(1e-6, session_seconds)
        avg_score_cleaned = self._clamp(avg_score_raw + (recovered_ratio * 16.0), 0.0, 100.0)

        distraction_count_raw = max(0.0, float(raw.get("distraction_count", 0.0) or 0.0))
        distraction_count_cleaned = distraction_count_raw
        if recovered_ratio >= 0.12 and distraction_count_cleaned > 0:
            distraction_count_cleaned = max(0.0, distraction_count_cleaned - 1.0)

        score_drop_raw = float(raw.get("score_drop_per_hour", 0.0) or 0.0)
        score_drop_factor = max(0.5, 1.0 - recovered_ratio * 0.7)
        score_drop_cleaned = score_drop_raw * score_drop_factor

        analytics_quality_score = self._compute_analytics_quality_score(
            session_seconds=session_seconds,
            face_presence_ratio=face_presence_ratio,
            uncertain_seconds_raw=uncertain_seconds_raw,
            uncertain_seconds_cleaned=uncertain_seconds_cleaned,
        )
        session_quality_weight = self._clamp(0.15 + analytics_quality_score * 0.85, 0.12, 1.0)
        personalization_weight = session_personalization_weight(raw, session_seconds)
        session_quality_weight = session_quality_weight * personalization_weight

        cleaned = dict(raw)
        cleaned.update(
            {
                "state_seconds_raw": raw_state_seconds,
                "state_seconds": cleaned_state_seconds,
                "state_segments_cleaned": segments,
                "focus_seconds_raw": float(max(0.0, focus_seconds_raw)),
                "focus_seconds_cleaned": float(max(0.0, focus_seconds_cleaned)),
                "session_seconds_cleaned": float(session_seconds),
                "avg_score_raw": float(avg_score_raw),
                "avg_score_cleaned": float(avg_score_cleaned),
                "distraction_count_cleaned": float(distraction_count_cleaned),
                "blink_rate_per_min_cleaned": float(raw.get("blink_rate_per_min", 0.0) or 0.0),
                "eye_closure_ratio_cleaned": float(raw.get("eye_closure_ratio", 0.0) or 0.0),
                "perclos_cleaned": float(raw.get("perclos", 0.0) or 0.0),
                "score_drop_per_hour_raw": float(score_drop_raw),
                "score_drop_per_hour_cleaned": float(score_drop_cleaned),
                "uncertain_seconds_raw": float(uncertain_seconds_raw),
                "uncertain_seconds_cleaned": float(max(0.0, uncertain_seconds_cleaned)),
                "uncertain_measurement_noise_seconds": float(uncertain_noise_seconds),
                "uncertain_behavioral_seconds": float(uncertain_behavioral_seconds),
                "analytics_quality_score": float(analytics_quality_score),
                "session_quality_weight": float(session_quality_weight),
                "personalization_eligible": bool(personalization_weight > 0.0),
                "personalization_sample_weight": float(personalization_weight),
                "face_presence_ratio": float(face_presence_ratio),
            }
        )
        return cleaned

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
        cleaned_record = self._clean_session_record_for_analytics(session_record)
        sessions.append(cleaned_record)

        if len(sessions) > self.max_sessions:
            sessions = sessions[-self.max_sessions:]

        learning_sessions = self._sessions_after_baseline_reset(profile, sessions)
        baseline = self.baseline_store.update_from_sessions(
            profile_name=profile_name,
            sessions=learning_sessions,
            default_work=default_work,
            default_break=default_break,
        )

        recommendation = self._build_recommendation(
            learning_sessions,
            default_work=default_work,
            default_break=default_break,
            baseline=baseline,
            minutes_since_last_break=cleaned_record.get("minutes_since_last_break"),
        )

        profile["sessions"] = sessions
        profile["recommendation"] = recommendation
        profile["baseline"] = baseline.to_dict()
        self.save_profile(profile_name, profile)

        # Best-effort remote sync. Never block or fail local analytics.
        self.supabase_sync.append_session(cleaned_record)
        baseline_payload = baseline.to_dict()
        baseline_payload["adaptation_stage"] = personalization_stage(baseline.session_count)
        recent_quality_scores: List[float] = []
        for sess in sessions[-5:]:
            try:
                recent_quality_scores.append(float(sess.get("analytics_quality_score", 0.0)))
            except (TypeError, ValueError):
                continue
        baseline_payload["last_quality_score"] = (
            float(statistics.fmean(recent_quality_scores))
            if recent_quality_scores
            else float(cleaned_record.get("analytics_quality_score", 0.0) or 0.0)
        )
        self.supabase_sync.upsert_user_baseline(baseline_payload)

        return recommendation

    def _load_or_refresh_baseline(
        self,
        profile_name: str,
        sessions: List[Dict[str, Any]],
        default_work: int,
        default_break: int,
        reset_at: Any = 0,
    ) -> UserBaseline:
        baseline = self.baseline_store.update_from_sessions(
            profile_name=profile_name,
            sessions=sessions,
            default_work=default_work,
            default_break=default_break,
        )

        # Fallback: when local history is sparse, try to hydrate from Supabase.
        if baseline.session_count < 3:
            remote = self.supabase_sync.load_user_baseline(profile_name)
            remote_updated_at = 0
            try:
                remote_updated_at = int(float((remote or {}).get("updated_at", 0) or 0))
            except (TypeError, ValueError):
                remote_updated_at = 0
            try:
                reset_ts = int(float(reset_at or 0))
            except (TypeError, ValueError):
                reset_ts = 0
            if remote and remote_updated_at >= reset_ts:
                baseline = self.baseline_store.merge_remote_baseline(profile_name, remote)

        return baseline

    def build_session_habit_report(
        self,
        session_record: Dict[str, Any],
        profile_name: str = "default",
    ) -> Dict[str, Any]:
        """
        Build a human-readable habit report from a completed session record.

        Returns a dict with:
        - session_seconds, effective_work_ratio, avg_work_readiness
        - fatigue_trend, distraction_trend
        - decline_start_minutes
        - break_effectiveness (list)
        - next_session_suggestion (str)
        - next_work_minutes, next_break_minutes
        """
        r = session_record
        session_seconds = float(r.get("session_seconds", 0) or 0)
        focus_seconds = float(r.get("focus_seconds_cleaned", r.get("focus_seconds", 0)) or 0)
        avg_score = float(r.get("avg_score_cleaned", r.get("avg_score", 0)) or 0)
        distraction_count = int(r.get("distraction_count_cleaned", r.get("distraction_count", 0)) or 0)
        score_drop = float(r.get("score_drop_per_hour_cleaned", r.get("score_drop_per_hour", 0)) or 0)
        fatigue_onset = r.get("fatigue_onset_minutes")
        perclos = float(r.get("perclos_cleaned", r.get("perclos", 0)) or 0)
        blink_rate = float(r.get("blink_rate_per_min_cleaned", r.get("blink_rate_per_min", 0)) or 0)
        eye_closure = float(r.get("eye_closure_ratio_cleaned", r.get("eye_closure_ratio", 0)) or 0)

        effective_work_ratio = self._clamp(focus_seconds / max(1.0, session_seconds), 0.0, 1.0)

        # Fatigue trend
        fatigue_signals = 0
        if perclos > 0.18:
            fatigue_signals += 1
        if eye_closure > 0.25:
            fatigue_signals += 1
        if blink_rate > 22:
            fatigue_signals += 1
        if score_drop > 10:
            fatigue_signals += 1

        if fatigue_signals >= 3:
            fatigue_trend = "Cao — dấu hiệu mệt mỏi rõ"
        elif fatigue_signals == 2:
            fatigue_trend = "Trung bình — có dấu hiệu mệt nhẹ"
        elif fatigue_signals == 1:
            fatigue_trend = "Thấp — ổn định"
        else:
            fatigue_trend = "Không đáng kể"

        # Distraction trend
        session_hours = max(session_seconds / 3600.0, 1e-6)
        dist_per_hour = distraction_count / session_hours
        if dist_per_hour > 6:
            distraction_trend = "Cao — nhiều lần lệch nhịp"
        elif dist_per_hour > 3:
            distraction_trend = "Trung bình"
        elif dist_per_hour > 1:
            distraction_trend = "Thấp"
        else:
            distraction_trend = "Rất thấp — tập trung tốt"

        # Decline start
        decline_start_minutes = None
        if fatigue_onset is not None:
            try:
                decline_start_minutes = float(fatigue_onset)
            except (TypeError, ValueError):
                pass
        if decline_start_minutes is None and score_drop > 8 and session_seconds > 600:
            decline_start_minutes = round(session_seconds / 60.0 * 0.6, 1)

        # Break effectiveness from checkins
        checkins = r.get("checkins", []) or []
        break_effectiveness: List[Dict[str, Any]] = []
        for ci in checkins:
            if not isinstance(ci, dict):
                continue
            answer = str(ci.get("answer", "") or "")
            if answer in ("on_task", "slight_drift"):
                transfer_score = 0.8 if answer == "on_task" else 0.5
            elif answer == "off_task":
                transfer_score = 0.2
            elif answer == "need_break":
                transfer_score = 0.1
            else:
                continue
            break_effectiveness.append({
                "break_type": "nghỉ ngắn",
                "transfer_score": transfer_score,
                "answer": answer,
            })

        # Next session suggestion
        rec = self.get_recommendation(
            profile_name,
            default_work=int(r.get("work_interval_minutes_used", 25) or 25),
            default_break=int(r.get("break_duration_minutes_used", 5) or 5),
        )
        next_work = int(rec.get("work_minutes", 25))
        next_break = int(rec.get("break_minutes", 5))

        suggestion_parts: List[str] = []
        if fatigue_signals >= 2:
            suggestion_parts.append("Nghỉ đủ trước phiên sau.")
        if dist_per_hour > 4:
            suggestion_parts.append("Thử sprint ngắn 15-20 phút để giảm phân tâm.")
        task_type = str(r.get("session_context", {}).get("task_type", "") or "")
        if task_type in ("reading", "review") and effective_work_ratio < 0.6:
            suggestion_parts.append("Chia nhỏ tài liệu thành các đoạn ngắn hơn.")
        if not suggestion_parts:
            suggestion_parts.append(f"Phiên sau: làm việc {next_work}p, nghỉ {next_break}p.")

        return {
            "session_seconds": int(session_seconds),
            "effective_work_ratio": effective_work_ratio,
            "avg_work_readiness": avg_score,
            "fatigue_trend": fatigue_trend,
            "distraction_trend": distraction_trend,
            "decline_start_minutes": decline_start_minutes,
            "break_effectiveness": break_effectiveness,
            "next_session_suggestion": " ".join(suggestion_parts),
            "next_work_minutes": next_work,
            "next_break_minutes": next_break,
            "distraction_count": distraction_count,
            "focus_seconds": int(focus_seconds),
        }

    def get_weekly_pattern(self, profile_name: str) -> Dict[str, Any]:
        """
        Analyse recent sessions to find the user's best working patterns.

        Returns a dict with:
        - best_hour_of_day (int or None)
        - best_work_duration_minutes (int)
        - best_break_type (str)
        - task_types_with_low_focus (list[str])
        - based_on_sessions (int)
        """
        profile = self.load_profile(profile_name)
        sessions = profile.get("sessions", [])

        valid = [
            s for s in sessions[-30:]
            if is_session_eligible_for_personalization(
                s,
                float(s.get("session_seconds_cleaned", s.get("session_seconds", 0)) or 0),
            )
        ]

        if len(valid) < 5:
            return {
                "best_hour_of_day": None,
                "best_work_duration_minutes": 25,
                "best_break_type": "nghỉ ngắn",
                "task_types_with_low_focus": [],
                "based_on_sessions": len(valid),
                "note": "Cần thêm dữ liệu (ít nhất 5 phiên) để phân tích xu hướng tuần.",
            }

        # Best hour of day by avg_score
        hour_scores: Dict[int, List[float]] = {}
        for s in valid:
            ts = s.get("timestamp")
            score = float(s.get("avg_score_cleaned", s.get("avg_score", 0)) or 0)
            if ts:
                try:
                    hour = datetime.fromtimestamp(int(ts)).hour
                    hour_scores.setdefault(hour, []).append(score)
                except (TypeError, ValueError, OSError):
                    pass

        best_hour = None
        if hour_scores:
            best_hour = max(hour_scores, key=lambda h: statistics.fmean(hour_scores[h]))

        # Best work duration
        durations = []
        for s in valid:
            dur = float(s.get("session_seconds_cleaned", s.get("session_seconds", 0)) or 0)
            score = float(s.get("avg_score_cleaned", s.get("avg_score", 0)) or 0)
            if dur > 0 and score > 0:
                durations.append((dur / 60.0, score))

        best_duration = 25
        if durations:
            # Weight duration by score
            total_weight = sum(sc for _, sc in durations)
            if total_weight > 0:
                weighted_dur = sum(d * sc for d, sc in durations) / total_weight
                best_duration = int(round(self._clamp(weighted_dur, 15, 60)))

        # Task types with low focus
        task_focus: Dict[str, List[float]] = {}
        for s in valid:
            ctx = s.get("session_context", {}) or {}
            task_type = str(ctx.get("task_type", "") or "")
            score = float(s.get("avg_score_cleaned", s.get("avg_score", 0)) or 0)
            if task_type:
                task_focus.setdefault(task_type, []).append(score)

        low_focus_tasks = [
            t for t, scores in task_focus.items()
            if len(scores) >= 2 and statistics.fmean(scores) < 65
        ]

        return {
            "best_hour_of_day": best_hour,
            "best_work_duration_minutes": best_duration,
            "best_break_type": "nghỉ ngắn",
            "task_types_with_low_focus": low_focus_tasks,
            "based_on_sessions": len(valid),
            "note": "",
        }

    def build_work_rhythm_summary(
        self,
        profile_name: str,
        *,
        live_session: Optional[Dict[str, Any]] = None,
        now_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Build day/week/month work-rhythm aggregates for the results dashboard.

        The report prioritizes actionable indicators:
        - effective work time and ratio
        - work-readiness score
        - distraction density
        - state composition and best time window
        """
        now_value = int(now_ts or time.time())
        now_dt = datetime.fromtimestamp(now_value)
        profile = self.load_profile(profile_name)
        sessions = list(profile.get("sessions", []) or [])
        if isinstance(live_session, dict) and live_session:
            live_payload = dict(live_session)
            live_payload.setdefault("timestamp", now_value)
            live_payload["_is_live_session"] = True
            sessions.append(live_payload)

        day_start_dt = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start_dt = (now_dt - timedelta(days=now_dt.weekday())).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        month_start_dt = now_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        periods = {
            "day": self._build_work_rhythm_period(
                sessions,
                start_dt=day_start_dt,
                end_dt=now_dt,
                bucket_mode="hour",
                label="Hôm nay",
            ),
            "week": self._build_work_rhythm_period(
                sessions,
                start_dt=week_start_dt,
                end_dt=now_dt,
                bucket_mode="day",
                label="Tuần này",
            ),
            "month": self._build_work_rhythm_period(
                sessions,
                start_dt=month_start_dt,
                end_dt=now_dt,
                bucket_mode="day",
                label="Tháng này",
            ),
        }

        return {
            "profile_name": profile_name,
            "generated_at": now_value,
            "recommendation": dict(profile.get("recommendation", {}) or {}),
            "periods": periods,
        }

    @classmethod
    def _build_work_rhythm_period(
        cls,
        sessions: List[Dict[str, Any]],
        *,
        start_dt: datetime,
        end_dt: datetime,
        bucket_mode: str,
        label: str,
    ) -> Dict[str, Any]:
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
        buckets = cls._make_work_rhythm_buckets(start_dt, end_dt, bucket_mode)
        bucket_by_key = {item["key"]: item for item in buckets}

        totals = cls._empty_work_rhythm_accumulator()
        used_sessions: List[Dict[str, Any]] = []

        for session in sessions:
            if not isinstance(session, dict):
                continue
            ts = cls._session_timestamp(session)
            if ts is None or ts < start_ts or ts > end_ts:
                continue
            duration = cls._session_duration_seconds(session)
            if duration <= 0.0:
                continue

            used_sessions.append(session)
            cls._add_session_to_work_rhythm_accumulator(totals, session)

            bucket_key = cls._work_rhythm_bucket_key(datetime.fromtimestamp(ts), bucket_mode)
            bucket = bucket_by_key.get(bucket_key)
            if bucket is not None:
                cls._add_session_to_work_rhythm_accumulator(bucket, session)

        points = [cls._finalize_work_rhythm_bucket(item) for item in buckets]
        result = cls._finalize_work_rhythm_period(
            totals,
            label=label,
            points=points,
            sessions=used_sessions,
        )
        result["start_timestamp"] = start_ts
        result["end_timestamp"] = end_ts
        return result

    @staticmethod
    def _empty_work_rhythm_accumulator() -> Dict[str, Any]:
        return {
            "session_count": 0,
            "total_seconds": 0.0,
            "focus_seconds": 0.0,
            "score_weighted_sum": 0.0,
            "score_weight": 0.0,
            "distraction_count": 0.0,
            "break_count": 0.0,
            "state_seconds": {
                "focused": 0.0,
                "distraction": 0.0,
                "fatigue": 0.0,
                "away": 0.0,
                "uncertain": 0.0,
            },
            "fatigue_onset_values": [],
            "live_session_included": False,
        }

    @classmethod
    def _make_work_rhythm_buckets(
        cls,
        start_dt: datetime,
        end_dt: datetime,
        bucket_mode: str,
    ) -> List[Dict[str, Any]]:
        buckets: List[Dict[str, Any]] = []
        if bucket_mode == "hour":
            cursor = start_dt
            while cursor <= end_dt:
                bucket = cls._empty_work_rhythm_accumulator()
                bucket.update(
                    {
                        "key": cls._work_rhythm_bucket_key(cursor, bucket_mode),
                        "label": f"{cursor.hour:02d}h",
                    }
                )
                buckets.append(bucket)
                cursor += timedelta(hours=1)
            return buckets

        cursor = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        last = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        while cursor <= last:
            bucket = cls._empty_work_rhythm_accumulator()
            bucket.update(
                {
                    "key": cls._work_rhythm_bucket_key(cursor, bucket_mode),
                    "label": cursor.strftime("%d/%m"),
                }
            )
            buckets.append(bucket)
            cursor += timedelta(days=1)
        return buckets

    @staticmethod
    def _work_rhythm_bucket_key(dt: datetime, bucket_mode: str) -> str:
        if bucket_mode == "hour":
            return dt.strftime("%Y-%m-%d %H")
        return dt.strftime("%Y-%m-%d")

    @staticmethod
    def _session_timestamp(session: Dict[str, Any]) -> Optional[int]:
        try:
            value = session.get("timestamp")
            if value is None:
                return None
            return int(float(value))
        except (TypeError, ValueError, OSError):
            return None

    @staticmethod
    def _session_metric(session: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
        for key in keys:
            try:
                value = session.get(key)
                if value is not None and value != "":
                    return float(value)
            except (TypeError, ValueError):
                continue
        return float(default)

    @classmethod
    def _session_duration_seconds(cls, session: Dict[str, Any]) -> float:
        return max(
            0.0,
            cls._session_metric(
                session,
                "session_seconds_cleaned",
                "session_seconds",
                default=0.0,
            ),
        )

    @classmethod
    def _session_focus_seconds(cls, session: Dict[str, Any]) -> float:
        return max(
            0.0,
            cls._session_metric(
                session,
                "focus_seconds_cleaned",
                "focus_seconds",
                "focus_seconds_display",
                default=0.0,
            ),
        )

    @classmethod
    def _session_avg_score(cls, session: Dict[str, Any]) -> float:
        return cls._clamp(
            cls._session_metric(
                session,
                "avg_score_cleaned",
                "avg_score",
                "avg_score_display",
                default=0.0,
            ),
            0.0,
            100.0,
        )

    @classmethod
    def _session_distractions(cls, session: Dict[str, Any]) -> float:
        return max(
            0.0,
            cls._session_metric(
                session,
                "distraction_count_cleaned",
                "distraction_count",
                default=0.0,
            ),
        )

    @classmethod
    def _session_state_categories(cls, session: Dict[str, Any]) -> Dict[str, float]:
        raw = session.get("state_seconds")
        if not isinstance(raw, dict):
            raw = session.get("state_seconds_raw")
        if not isinstance(raw, dict):
            raw = {}

        def value(name: str) -> float:
            try:
                return max(0.0, float(raw.get(name, 0.0) or 0.0))
            except (TypeError, ValueError):
                return 0.0

        focused = value("ON_SCREEN_READING") + value("OFFSCREEN_WRITING")
        categories = {
            "focused": focused,
            "distraction": value("PHONE_DISTRACTION"),
            "fatigue": value("DROWSY_FATIGUE"),
            "away": value("AWAY"),
            "uncertain": value("UNCERTAIN"),
        }

        state_total = sum(categories.values())
        duration = cls._session_duration_seconds(session)
        if state_total <= 0.0 and duration > 0.0:
            focus_seconds = min(duration, cls._session_focus_seconds(session))
            categories["focused"] = focus_seconds
            categories["uncertain"] = max(0.0, duration - focus_seconds)
        return categories

    @classmethod
    def _add_session_to_work_rhythm_accumulator(
        cls,
        acc: Dict[str, Any],
        session: Dict[str, Any],
    ) -> None:
        duration = cls._session_duration_seconds(session)
        focus_seconds = min(duration, cls._session_focus_seconds(session)) if duration > 0 else 0.0
        score = cls._session_avg_score(session)
        score_weight = max(1.0, duration)

        acc["session_count"] = int(acc.get("session_count", 0) or 0) + 1
        acc["total_seconds"] = float(acc.get("total_seconds", 0.0) or 0.0) + duration
        acc["focus_seconds"] = float(acc.get("focus_seconds", 0.0) or 0.0) + focus_seconds
        acc["score_weighted_sum"] = float(acc.get("score_weighted_sum", 0.0) or 0.0) + score * score_weight
        acc["score_weight"] = float(acc.get("score_weight", 0.0) or 0.0) + score_weight
        acc["distraction_count"] = float(acc.get("distraction_count", 0.0) or 0.0) + cls._session_distractions(session)
        acc["break_count"] = float(acc.get("break_count", 0.0) or 0.0) + cls._session_metric(session, "break_count", default=0.0)
        acc["live_session_included"] = bool(acc.get("live_session_included", False)) or bool(session.get("_is_live_session"))

        state_totals = acc.setdefault("state_seconds", {})
        for key, seconds in cls._session_state_categories(session).items():
            state_totals[key] = float(state_totals.get(key, 0.0) or 0.0) + float(seconds or 0.0)

        fatigue_onset = session.get("fatigue_onset_minutes")
        try:
            if fatigue_onset is not None and fatigue_onset != "":
                fatigue_value = float(fatigue_onset)
                if fatigue_value > 0:
                    acc.setdefault("fatigue_onset_values", []).append(fatigue_value)
        except (TypeError, ValueError):
            pass

    @staticmethod
    def _finalize_work_rhythm_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
        total_seconds = float(bucket.get("total_seconds", 0.0) or 0.0)
        focus_seconds = float(bucket.get("focus_seconds", 0.0) or 0.0)
        score_weight = float(bucket.get("score_weight", 0.0) or 0.0)
        avg_score = (
            float(bucket.get("score_weighted_sum", 0.0) or 0.0) / score_weight
            if score_weight > 0.0
            else None
        )
        hours = max(total_seconds / 3600.0, 1e-6)
        return {
            "key": str(bucket.get("key", "") or ""),
            "label": str(bucket.get("label", "") or ""),
            "session_count": int(bucket.get("session_count", 0) or 0),
            "total_minutes": round(total_seconds / 60.0, 2),
            "focus_minutes": round(focus_seconds / 60.0, 2),
            "focus_ratio": (focus_seconds / total_seconds) if total_seconds > 0.0 else 0.0,
            "avg_score": avg_score,
            "distractions_per_hour": (
                float(bucket.get("distraction_count", 0.0) or 0.0) / hours
                if total_seconds > 0.0
                else 0.0
            ),
        }

    @classmethod
    def _finalize_work_rhythm_period(
        cls,
        totals: Dict[str, Any],
        *,
        label: str,
        points: List[Dict[str, Any]],
        sessions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        total_seconds = float(totals.get("total_seconds", 0.0) or 0.0)
        focus_seconds = float(totals.get("focus_seconds", 0.0) or 0.0)
        session_count = int(totals.get("session_count", 0) or 0)
        score_weight = float(totals.get("score_weight", 0.0) or 0.0)
        avg_score = (
            float(totals.get("score_weighted_sum", 0.0) or 0.0) / score_weight
            if score_weight > 0.0
            else 0.0
        )
        hours = max(total_seconds / 3600.0, 1e-6)
        distraction_count = float(totals.get("distraction_count", 0.0) or 0.0)
        break_count = float(totals.get("break_count", 0.0) or 0.0)
        focus_ratio = focus_seconds / total_seconds if total_seconds > 0.0 else 0.0
        state_seconds = dict(totals.get("state_seconds", {}) or {})
        state_total = sum(float(v or 0.0) for v in state_seconds.values())
        if state_total <= 0.0 and total_seconds > 0.0:
            state_seconds = {
                "focused": focus_seconds,
                "distraction": 0.0,
                "fatigue": 0.0,
                "away": 0.0,
                "uncertain": max(0.0, total_seconds - focus_seconds),
            }
            state_total = total_seconds

        state_distribution = {}
        for key in ("focused", "distraction", "fatigue", "away", "uncertain"):
            seconds = float(state_seconds.get(key, 0.0) or 0.0)
            state_distribution[key] = {
                "seconds": round(seconds, 2),
                "ratio": seconds / state_total if state_total > 0.0 else 0.0,
            }

        active_points = [point for point in points if point.get("session_count", 0) > 0]
        best_point = None
        if active_points:
            best_point = max(
                active_points,
                key=lambda point: (
                    float(point.get("avg_score") or 0.0),
                    float(point.get("focus_minutes", 0.0) or 0.0),
                ),
            )

        score_delta = cls._score_delta_for_sessions(sessions)
        fatigue_values = list(totals.get("fatigue_onset_values", []) or [])
        avg_fatigue_onset = statistics.fmean(fatigue_values) if fatigue_values else None
        distraction_rate = distraction_count / hours if total_seconds > 0.0 else 0.0

        insights = cls._build_work_rhythm_insights(
            label=label,
            session_count=session_count,
            focus_ratio=focus_ratio,
            avg_score=avg_score,
            distraction_rate=distraction_rate,
            state_distribution=state_distribution,
            best_point=best_point,
            score_delta=score_delta,
            avg_fatigue_onset=avg_fatigue_onset,
            live_session_included=bool(totals.get("live_session_included", False)),
        )

        return {
            "label": label,
            "session_count": session_count,
            "total_seconds": round(total_seconds, 2),
            "focus_seconds": round(focus_seconds, 2),
            "focus_ratio": focus_ratio,
            "avg_score": avg_score,
            "distraction_count": int(round(distraction_count)),
            "break_count": int(round(break_count)),
            "distractions_per_hour": distraction_rate,
            "avg_session_minutes": (total_seconds / 60.0 / session_count) if session_count else 0.0,
            "avg_fatigue_onset_minutes": avg_fatigue_onset,
            "score_delta": score_delta,
            "best_bucket_label": str(best_point.get("label", "") if best_point else ""),
            "live_session_included": bool(totals.get("live_session_included", False)),
            "state_distribution": state_distribution,
            "points": points,
            "insights": insights,
        }

    @classmethod
    def _score_delta_for_sessions(cls, sessions: List[Dict[str, Any]]) -> float:
        scored: List[float] = []
        for session in sorted(
            sessions,
            key=lambda item: cls._session_timestamp(item) or 0,
        ):
            score = cls._session_avg_score(session)
            if score > 0.0:
                scored.append(score)
        if len(scored) < 2:
            return 0.0
        pivot = max(1, len(scored) // 2)
        first = scored[:pivot]
        second = scored[pivot:]
        if not first or not second:
            return 0.0
        return float(statistics.fmean(second) - statistics.fmean(first))

    @staticmethod
    def _build_work_rhythm_insights(
        *,
        label: str,
        session_count: int,
        focus_ratio: float,
        avg_score: float,
        distraction_rate: float,
        state_distribution: Dict[str, Dict[str, float]],
        best_point: Optional[Dict[str, Any]],
        score_delta: float,
        avg_fatigue_onset: Optional[float],
        live_session_included: bool,
    ) -> List[str]:
        if session_count <= 0:
            return [f"{label} chưa có phiên đủ dữ liệu để tổng hợp."]

        insights: List[str] = []
        if focus_ratio >= 0.78 and avg_score >= 78:
            insights.append("Nhịp làm việc đang ổn định: phần lớn thời gian là trạng thái hiệu quả.")
        elif focus_ratio >= 0.62:
            insights.append("Nhịp làm việc ở mức dùng được, nhưng vẫn còn khoảng dao động cần giảm.")
        else:
            insights.append("Thời gian hiệu quả còn thấp; nên chia phiên ngắn hơn và đặt mục tiêu nhỏ hơn.")

        if distraction_rate >= 6.0:
            insights.append("Mật độ lệch nhịp cao; nên giảm thông báo và chuẩn bị tài liệu trước khi bắt đầu.")
        elif distraction_rate >= 3.0:
            insights.append("Có một số lần lệch nhịp; nên nghỉ ngắn trước khi điểm sẵn sàng tụt sâu.")
        else:
            insights.append("Mật độ lệch nhịp thấp, đây là dấu hiệu tốt cho khả năng duy trì tác vụ.")

        fatigue_ratio = float(state_distribution.get("fatigue", {}).get("ratio", 0.0) or 0.0)
        if fatigue_ratio >= 0.14:
            if avg_fatigue_onset:
                insights.append(f"Dấu hiệu mệt xuất hiện đáng kể, thường bắt đầu quanh phút {avg_fatigue_onset:.0f}.")
            else:
                insights.append("Dấu hiệu mệt xuất hiện đáng kể; phiên sau nên nghỉ sớm hơn.")

        if best_point and str(best_point.get("label", "") or ""):
            insights.append(f"Khung tốt nhất trong kỳ này: {best_point['label']}.")

        if score_delta >= 5.0:
            insights.append("Điểm sẵn sàng đang tăng ở các phiên gần đây.")
        elif score_delta <= -5.0:
            insights.append("Điểm sẵn sàng đang giảm ở các phiên gần đây; nên giảm độ dài phiên kế tiếp.")

        if live_session_included:
            insights.append("Báo cáo đã cộng cả phiên đang chạy nên số liệu sẽ tiếp tục thay đổi.")

        return insights[:5]

    @staticmethod
    def _compute_score_trend(valid_sessions: List[Dict[str, Any]]) -> float:
        avg_scores: List[float] = []
        for sess in valid_sessions[-10:]:
            try:
                avg_scores.append(float(sess.get("avg_score_cleaned", sess.get("avg_score", 0.0))))
            except (TypeError, ValueError):
                continue

        if len(avg_scores) < 4:
            return 0.0

        pivot = len(avg_scores) // 2
        old_part = avg_scores[:pivot]
        new_part = avg_scores[pivot:]
        if not old_part or not new_part:
            return 0.0

        return statistics.fmean(new_part) - statistics.fmean(old_part)

    def _build_recommendation(
        self,
        sessions: List[Dict[str, Any]],
        default_work: int,
        default_break: int,
        baseline: Optional[UserBaseline] = None,
        minutes_since_last_break: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Build recommendation from recent sessions.

        Uses focus ratio, average score, and distraction density
        plus user baseline and trends to choose personalized work/break timing.
        """
        default_work = int(max(15, min(60, default_work)))
        default_break = min(
            int(max(3, min(20, default_break))),
            science_informed_break_minutes(default_work, 0.0),
        )

        valid_sessions: List[Dict[str, Any]] = []
        for sess in sessions[-30:]:
            duration = float(sess.get("session_seconds_cleaned", sess.get("session_seconds", 0.0)) or 0.0)
            if is_session_eligible_for_personalization(sess, duration):
                valid_sessions.append(sess)

        if not valid_sessions:
            return {
                "work_minutes": default_work,
                "break_minutes": default_break,
                "confidence": 0.0,
                "reason": "Not enough history yet",
                "based_on_sessions": 0,
                "adaptation_stage": "cold_start",
            }

        focus_ratios: List[float] = []
        avg_scores: List[float] = []
        distractions_per_hour: List[float] = []
        task_alignment_scores: List[float] = []
        digital_risks: List[float] = []
        context_switches_per_hour: List[float] = []
        fatigue_onsets: List[float] = []
        blink_rates: List[float] = []
        closure_ratios: List[float] = []
        perclos_values: List[float] = []
        quality_scores: List[float] = []

        for sess in valid_sessions:
            duration = max(
                1.0,
                float(sess.get("session_seconds_cleaned", sess.get("session_seconds", 1.0)) or 1.0),
            )
            focus_seconds = max(
                0.0,
                float(sess.get("focus_seconds_cleaned", sess.get("focus_seconds", 0.0)) or 0.0),
            )
            focus_ratios.append(min(1.0, focus_seconds / duration))
            avg_scores.append(float(sess.get("avg_score_cleaned", sess.get("avg_score", 0.0)) or 0.0))

            distractions = max(
                0.0,
                float(sess.get("distraction_count_cleaned", sess.get("distraction_count", 0.0)) or 0.0),
            )
            distractions_per_hour.append(distractions / (duration / 3600.0))

            task_alignment = sess.get("task_alignment_avg")
            digital_risk = sess.get("digital_distraction_risk_avg")
            context_switch_count = sess.get("context_switch_count")

            try:
                if task_alignment is not None:
                    task_alignment_scores.append(self._clamp(float(task_alignment), 0.0, 1.0))
            except (TypeError, ValueError):
                pass

            try:
                if digital_risk is not None:
                    digital_risks.append(self._clamp(float(digital_risk), 0.0, 1.0))
            except (TypeError, ValueError):
                pass

            try:
                if context_switch_count is not None:
                    context_switches_per_hour.append(
                        max(0.0, float(context_switch_count)) / (duration / 3600.0)
                    )
            except (TypeError, ValueError):
                pass

            quality = sess.get("analytics_quality_score")
            try:
                if quality is not None:
                    quality_scores.append(float(quality))
            except (TypeError, ValueError):
                pass

            fatigue_onset = sess.get("fatigue_onset_minutes")
            try:
                if fatigue_onset is not None:
                    fatigue_value = float(fatigue_onset)
                    if fatigue_value > 0:
                        fatigue_onsets.append(fatigue_value)
            except (TypeError, ValueError):
                pass

            for key, target in (
                ("blink_rate_per_min_cleaned", blink_rates),
                ("eye_closure_ratio_cleaned", closure_ratios),
                ("perclos_cleaned", perclos_values),
            ):
                value = sess.get(key)
                if value is None:
                    fallback_key = key.replace("_cleaned", "")
                    value = sess.get(fallback_key)
                try:
                    if value is not None:
                        target.append(float(value))
                except (TypeError, ValueError):
                    continue

        avg_focus_ratio = statistics.fmean(focus_ratios)
        avg_score = statistics.fmean(avg_scores)
        avg_distraction_per_hour = statistics.fmean(distractions_per_hour)
        avg_fatigue_onset = statistics.fmean(fatigue_onsets) if fatigue_onsets else 0.0
        avg_blink_rate = statistics.fmean(blink_rates) if blink_rates else 0.0
        avg_closure_ratio = statistics.fmean(closure_ratios) if closure_ratios else 0.0
        avg_perclos = statistics.fmean(perclos_values) if perclos_values else 0.0
        avg_quality_score = statistics.fmean(quality_scores) if quality_scores else 0.72
        avg_task_alignment = statistics.fmean(task_alignment_scores) if task_alignment_scores else 0.75
        avg_digital_risk = statistics.fmean(digital_risks) if digital_risks else 0.0
        avg_context_switches_per_hour = (
            statistics.fmean(context_switches_per_hour)
            if context_switches_per_hour
            else 0.0
        )
        score_trend = self._compute_score_trend(valid_sessions)

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

        # Break duration follows observed strain, not a blind copy of previous breaks.
        strain_level = 0.0
        if avg_distraction_per_hour > 6.0 or avg_score < 55:
            strain_level = max(strain_level, 0.75)
        elif avg_distraction_per_hour > 4.0 or avg_score < 65:
            strain_level = max(strain_level, 0.55)
        elif avg_distraction_per_hour > 2.5 or avg_score < 75:
            strain_level = max(strain_level, 0.32)

        if score_trend <= -4.0:
            strain_level = max(strain_level, 0.58)

        break_minutes = science_informed_break_minutes(work_minutes, strain_level)

        work_minutes = int(max(15, min(60, work_minutes)))
        break_minutes = int(max(3, min(20, break_minutes)))

        adaptation_stage = "cold_start"
        personalization_weight = 0.0

        if baseline is not None:
            adaptation_stage = personalization_stage(baseline.session_count)
            personalization_weight = float(max(0.0, min(1.0, baseline.personalization_weight)))

            baseline_work = int(max(15, min(60, baseline.recommended_work_minutes)))
            baseline_break_strain = 0.0
            if baseline.average_distraction_density > 4.5 or baseline.average_focus_score_baseline < 70:
                baseline_break_strain = max(baseline_break_strain, 0.45)
            if baseline.focus_score_decay_per_hour > 8.0:
                baseline_break_strain = max(baseline_break_strain, 0.55)
            baseline_break = min(
                int(max(3, min(20, baseline.recommended_break_minutes))),
                science_informed_break_minutes(baseline_work, baseline_break_strain),
            )
            anchor_weight = max(0.35, min(0.95, 0.18 + personalization_weight * 0.72))

            work_minutes = int(round((work_minutes * (1.0 - anchor_weight)) + (baseline_work * anchor_weight)))
            break_minutes = int(round((break_minutes * (1.0 - anchor_weight)) + (baseline_break * anchor_weight)))

            fatigue_based_work = max(18, min(55, int(round(baseline.average_fatigue_onset_minutes * 0.82))))
            if fatigue_based_work > 0:
                blend_weight = max(0.3, personalization_weight)
                work_minutes = int(round((work_minutes * (1.0 - blend_weight)) + (fatigue_based_work * blend_weight)))

            if baseline.average_distraction_density > 4.5:
                strain_level = max(strain_level, 0.50)
                break_minutes += 1
            if baseline.average_focus_score_baseline < 70:
                strain_level = max(strain_level, 0.48)
                work_minutes -= 3
                break_minutes += 1

            if baseline.focus_score_decay_per_hour > 8.0:
                strain_level = max(strain_level, 0.55)
                work_minutes -= 2
                break_minutes += 1

            if score_trend <= -4.0:
                strain_level = max(strain_level, 0.62)
                work_minutes -= 4
                break_minutes += 2
            elif score_trend >= 4.0 and avg_distraction_per_hour < 2.0:
                work_minutes += 2
                break_minutes -= 1

            if avg_fatigue_onset > 0 and avg_fatigue_onset < 30:
                strain_level = max(strain_level, 0.60)
                work_minutes = min(work_minutes, int(round(avg_fatigue_onset * 0.85)))
                break_minutes += 1

            # Escalate break recommendation only when multiple eye-fatigue signals worsen.
            eye_fatigue_signals = 0
            if avg_blink_rate > baseline.blink_rate_baseline * 1.25:
                eye_fatigue_signals += 1
            if avg_closure_ratio > baseline.eye_closure_ratio_baseline + 0.06:
                eye_fatigue_signals += 1
            if avg_perclos > baseline.perclos_baseline + 0.05:
                eye_fatigue_signals += 1

            if eye_fatigue_signals >= 2:
                strain_level = max(strain_level, 0.68)
                work_minutes -= 3
                break_minutes += 2

            if avg_quality_score < 0.5:
                work_minutes = min(work_minutes, baseline_work)
                break_minutes = max(break_minutes, min(baseline_break, science_informed_break_minutes(work_minutes, 0.45)))

        if minutes_since_last_break is not None:
            try:
                break_minutes_since = max(0.0, float(minutes_since_last_break))
                if break_minutes_since >= work_minutes * 0.75:
                    strain_level = max(strain_level, 0.35)
                    break_minutes = max(break_minutes, 6)
                if break_minutes_since >= work_minutes * 1.1:
                    strain_level = max(strain_level, 0.55)
                    break_minutes = max(break_minutes, 8)
                    work_minutes = min(work_minutes, int(round(default_work * 0.92)))
            except (TypeError, ValueError):
                pass

        # Digital-context pressure: shorten work interval when task alignment is low
        # or when digital distraction / context switching is high.
        if avg_digital_risk >= 0.65 or avg_task_alignment < 0.45:
            strain_level = max(strain_level, 0.60)
            work_minutes -= 5
            break_minutes += 2
        elif avg_digital_risk >= 0.45 or avg_context_switches_per_hour >= 18:
            strain_level = max(strain_level, 0.45)
            work_minutes -= 3
            break_minutes += 1

        if len(valid_sessions) < 5:
            history_weight = self._clamp(len(valid_sessions) / 5.0, 0.15, 0.85)
            work_minutes = int(round((default_work * (1.0 - history_weight)) + (work_minutes * history_weight)))
            break_minutes = int(round((default_break * (1.0 - history_weight)) + (break_minutes * history_weight)))
            strain_level *= max(0.35, history_weight)

        work_minutes = int(max(15, min(60, work_minutes)))
        break_cap = science_informed_break_minutes(work_minutes, strain_level)
        break_minutes = int(max(3, min(break_cap, break_minutes)))

        confidence = min(
            1.0,
            (len(valid_sessions) / 12.0)
            * (0.62 + personalization_weight * 0.25 + max(0.0, min(1.0, avg_quality_score)) * 0.13),
        )
        reason = (
            f"{adaptation_stage}: làm việc {work_minutes}p, nghỉ {break_minutes}p | "
            f"focus={avg_focus_ratio:.0%}, score={avg_score:.1f}, "
            f"xao_nhang/giờ={avg_distraction_per_hour:.1f}, trend={score_trend:+.1f}, "
            f"quality={avg_quality_score:.2f}, "
            f"task_alignment={avg_task_alignment:.0%}, digital_risk={avg_digital_risk:.0%}, "
            f"context_switches/giờ={avg_context_switches_per_hour:.1f}"
        )

        return {
            "work_minutes": work_minutes,
            "break_minutes": break_minutes,
            "confidence": confidence,
            "reason": reason,
            "based_on_sessions": len(valid_sessions),
            "adaptation_stage": adaptation_stage,
        }
