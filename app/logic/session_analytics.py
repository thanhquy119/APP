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

from .google_sheets_sync import GoogleSheetsSessionSync
from .personalization import (
    PersonalizationManager,
    UserBaseline,
    UserBaselineStore,
    personalization_stage,
)

logger = logging.getLogger(__name__)

FOCUSED_STATE_NAMES = {"ON_SCREEN_READING", "OFFSCREEN_WRITING"}


class SessionAnalyticsStore:
    """Persist session data and build personalized timing recommendations."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        max_sessions: int = 300,
        google_config: Optional[Dict[str, Any]] = None,
    ):
        self.base_dir = base_dir or Path("analytics") / "profiles"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_sessions = max_sessions

        self.baseline_store = UserBaselineStore()
        self.personalization_manager = PersonalizationManager(self.baseline_store)

        self.google_sync = GoogleSheetsSessionSync()
        if google_config:
            self.configure_google_sheets(google_config)

    def configure_google_sheets(self, app_config: Dict[str, Any]) -> None:
        """Apply Google Sheets sync settings from app config."""
        self.google_sync.configure_from_app_config(app_config)

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
        sessions = profile.get("sessions", [])

        baseline = self._load_or_refresh_baseline(
            profile_name=profile_name,
            sessions=sessions,
            default_work=default_work,
            default_break=default_break,
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
        sessions = profile.get("sessions", [])
        baseline = self._load_or_refresh_baseline(
            profile_name=profile_name,
            sessions=sessions,
            default_work=default_work,
            default_break=default_break,
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
        sessions = profile.get("sessions", [])
        baseline = self._load_or_refresh_baseline(
            profile_name=profile_name,
            sessions=sessions,
            default_work=default_work,
            default_break=default_break,
        )
        thresholds = self.personalization_manager.build_thresholds(
            profile_name=profile_name,
            baseline=baseline,
            focus_defaults=focus_engine_defaults,
        )
        return thresholds.to_dict()

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

        baseline = self.baseline_store.update_from_sessions(
            profile_name=profile_name,
            sessions=sessions,
            default_work=default_work,
            default_break=default_break,
        )

        recommendation = self._build_recommendation(
            sessions,
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
        self.google_sync.append_session(cleaned_record)
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
        self.google_sync.upsert_user_baseline(baseline_payload)

        return recommendation

    def _load_or_refresh_baseline(
        self,
        profile_name: str,
        sessions: List[Dict[str, Any]],
        default_work: int,
        default_break: int,
    ) -> UserBaseline:
        baseline = self.baseline_store.update_from_sessions(
            profile_name=profile_name,
            sessions=sessions,
            default_work=default_work,
            default_break=default_break,
        )

        # Fallback: when local history is sparse, try to hydrate from Google Sheets.
        if baseline.session_count < 3:
            remote = self.google_sync.load_user_baseline(profile_name)
            if remote:
                baseline = self.baseline_store.merge_remote_baseline(profile_name, remote)

        return baseline

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
        default_break = int(max(3, min(20, default_break)))

        valid_sessions: List[Dict[str, Any]] = []
        for sess in sessions[-30:]:
            duration = float(sess.get("session_seconds_cleaned", sess.get("session_seconds", 0.0)) or 0.0)
            if duration >= 5 * 60:
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

        adaptation_stage = "cold_start"
        personalization_weight = 0.0

        if baseline is not None:
            adaptation_stage = personalization_stage(baseline.session_count)
            personalization_weight = float(max(0.0, min(1.0, baseline.personalization_weight)))

            baseline_work = int(max(15, min(60, baseline.recommended_work_minutes)))
            baseline_break = int(max(3, min(20, baseline.recommended_break_minutes)))
            anchor_weight = max(0.35, min(0.95, 0.18 + personalization_weight * 0.72))

            work_minutes = int(round((work_minutes * (1.0 - anchor_weight)) + (baseline_work * anchor_weight)))
            break_minutes = int(round((break_minutes * (1.0 - anchor_weight)) + (baseline_break * anchor_weight)))

            fatigue_based_work = max(18, min(55, int(round(baseline.average_fatigue_onset_minutes * 0.82))))
            if fatigue_based_work > 0:
                blend_weight = max(0.3, personalization_weight)
                work_minutes = int(round((work_minutes * (1.0 - blend_weight)) + (fatigue_based_work * blend_weight)))

            if baseline.average_distraction_density > 4.5:
                break_minutes += 1
            if baseline.average_focus_score_baseline < 70:
                work_minutes -= 3
                break_minutes += 1

            if baseline.focus_score_decay_per_hour > 8.0:
                work_minutes -= 2
                break_minutes += 1

            if score_trend <= -4.0:
                work_minutes -= 4
                break_minutes += 2
            elif score_trend >= 4.0 and avg_distraction_per_hour < 2.0:
                work_minutes += 2
                break_minutes -= 1

            if avg_fatigue_onset > 0 and avg_fatigue_onset < 30:
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
                work_minutes -= 3
                break_minutes += 2

            if avg_quality_score < 0.5:
                work_minutes = min(work_minutes, baseline_work)
                break_minutes = max(break_minutes, baseline_break)

        if minutes_since_last_break is not None:
            try:
                break_minutes_since = max(0.0, float(minutes_since_last_break))
                if break_minutes_since >= work_minutes * 0.75:
                    break_minutes = max(break_minutes, 6)
                if break_minutes_since >= work_minutes * 1.1:
                    break_minutes = max(break_minutes, 8)
                    work_minutes = min(work_minutes, int(round(default_work * 0.92)))
            except (TypeError, ValueError):
                pass

        work_minutes = int(max(15, min(60, work_minutes)))
        break_minutes = int(max(3, min(20, break_minutes)))

        confidence = min(
            1.0,
            (len(valid_sessions) / 12.0)
            * (0.62 + personalization_weight * 0.25 + max(0.0, min(1.0, avg_quality_score)) * 0.13),
        )
        reason = (
            f"{adaptation_stage}: làm việc {work_minutes}p, nghỉ {break_minutes}p | "
            f"focus={avg_focus_ratio:.0%}, score={avg_score:.1f}, "
            f"xao_nhang/giờ={avg_distraction_per_hour:.1f}, trend={score_trend:+.1f}, "
            f"quality={avg_quality_score:.2f}"
        )

        return {
            "work_minutes": work_minutes,
            "break_minutes": break_minutes,
            "confidence": confidence,
            "reason": reason,
            "based_on_sessions": len(valid_sessions),
            "adaptation_stage": adaptation_stage,
        }
