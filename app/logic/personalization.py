"""
Per-user baseline modeling and adaptive threshold generation.

This module keeps personalization logic outside of FocusEngine's state machine,
then maps user-specific baselines into runtime thresholds.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..vision.blink import BlinkConfig, BlinkDetector

logger = logging.getLogger(__name__)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _blend(default_value: float, personalized_value: float, weight: float) -> float:
    w = _clamp(float(weight), 0.0, 1.0)
    return (default_value * (1.0 - w)) + (personalized_value * w)


def _trim_iqr_outliers(values: Sequence[float]) -> List[float]:
    cleaned = [float(v) for v in values if v is not None]
    if len(cleaned) < 5:
        return cleaned

    sorted_vals = sorted(cleaned)
    q1 = sorted_vals[len(sorted_vals) // 4]
    q3 = sorted_vals[(len(sorted_vals) * 3) // 4]
    iqr = q3 - q1
    if iqr <= 1e-9:
        return cleaned

    low = q1 - (1.5 * iqr)
    high = q3 + (1.5 * iqr)
    filtered = [v for v in cleaned if low <= v <= high]
    return filtered or cleaned


def _weighted_recent_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    total_weight = 0.0
    weighted_sum = 0.0
    for idx, value in enumerate(values):
        ratio = idx / max(1, len(values) - 1)
        weight = 1.0 + (ratio * 1.6)
        total_weight += weight
        weighted_sum += float(value) * weight

    return weighted_sum / max(total_weight, 1e-6)


def _weighted_recent_mean_samples(samples: Sequence[tuple[float, float]]) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return float(samples[0][0])

    total_weight = 0.0
    weighted_sum = 0.0
    for idx, (value, quality_weight) in enumerate(samples):
        ratio = idx / max(1, len(samples) - 1)
        recency_weight = 1.0 + (ratio * 1.6)
        quality = _clamp(float(quality_weight), 0.05, 1.0)
        weight = recency_weight * quality
        total_weight += weight
        weighted_sum += float(value) * weight

    return weighted_sum / max(total_weight, 1e-6)


def _robust_recent_center(values: Sequence[float], fallback: float) -> float:
    cleaned = _trim_iqr_outliers(values)
    if not cleaned:
        return float(fallback)

    median_value = float(statistics.median(cleaned))
    weighted_value = float(_weighted_recent_mean(cleaned))
    center = (median_value * 0.6) + (weighted_value * 0.4)
    return float(center)


def _robust_recent_center_weighted(samples: Sequence[tuple[float, float]], fallback: float) -> float:
    if not samples:
        return float(fallback)

    valid_samples: List[tuple[float, float]] = []
    for value, weight in samples:
        try:
            v = float(value)
            w = _clamp(float(weight), 0.05, 1.0)
            valid_samples.append((v, w))
        except (TypeError, ValueError):
            continue

    if not valid_samples:
        return float(fallback)

    values = [v for v, _ in valid_samples]
    if len(values) >= 5:
        sorted_vals = sorted(values)
        q1 = sorted_vals[len(sorted_vals) // 4]
        q3 = sorted_vals[(len(sorted_vals) * 3) // 4]
        iqr = q3 - q1
        if iqr > 1e-9:
            low = q1 - (1.5 * iqr)
            high = q3 + (1.5 * iqr)
            filtered = [(v, w) for v, w in valid_samples if low <= v <= high]
            if filtered:
                valid_samples = filtered

    median_value = float(statistics.median([v for v, _ in valid_samples]))
    weighted_value = float(_weighted_recent_mean_samples(valid_samples))
    center = (median_value * 0.55) + (weighted_value * 0.45)
    return float(center)


def _ema(values: Sequence[float], alpha: float = 0.35) -> float:
    if not values:
        return 0.0

    ema_value = float(values[0])
    for value in values[1:]:
        ema_value = (alpha * float(value)) + ((1.0 - alpha) * ema_value)
    return ema_value


def compute_personalization_weight(session_count: int) -> float:
    """
    Convert session count to hybrid personalization weight.

    Fallback stages:
    - <3 sessions: mostly defaults
    - 3-7 sessions: hybrid
    - >7 sessions: mostly personalized
    """
    n = max(0, int(session_count))

    if n < 3:
        return 0.08 + (n / 3.0) * 0.12

    if n <= 7:
        return 0.28 + ((n - 3) / 4.0) * 0.42

    return min(0.92, 0.72 + ((n - 7) * 0.03))


def personalization_stage(session_count: int) -> str:
    n = max(0, int(session_count))
    if n < 3:
        return "cold_start"
    if n <= 7:
        return "hybrid"
    return "personalized"


@dataclass
class UserBaseline:
    profile_name: str
    session_count: int = 0
    blink_rate_baseline: float = 12.0
    avg_ear_baseline: float = 0.25
    eye_closure_ratio_baseline: float = 0.12
    perclos_baseline: float = 0.08
    average_focus_score_baseline: float = 75.0
    average_distraction_density: float = 3.0
    average_fatigue_onset_minutes: float = 35.0
    focus_score_decay_per_hour: float = 0.0
    recommended_work_minutes: int = 25
    recommended_break_minutes: int = 5
    last_quality_score: float = 0.0
    personalization_weight: float = 0.0
    updated_at: int = 0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["updated_at"] = int(self.updated_at or time.time())
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any], profile_name: str) -> "UserBaseline":
        if not isinstance(data, dict):
            return cls(profile_name=profile_name, updated_at=int(time.time()))

        return cls(
            profile_name=str(data.get("profile_name", profile_name)).strip() or profile_name,
            session_count=max(0, int(data.get("session_count", 0) or 0)),
            blink_rate_baseline=float(data.get("blink_rate_baseline", 12.0) or 12.0),
            avg_ear_baseline=float(data.get("avg_ear_baseline", 0.25) or 0.25),
            eye_closure_ratio_baseline=float(data.get("eye_closure_ratio_baseline", 0.12) or 0.12),
            perclos_baseline=float(data.get("perclos_baseline", 0.08) or 0.08),
            average_focus_score_baseline=float(data.get("average_focus_score_baseline", 75.0) or 75.0),
            average_distraction_density=float(data.get("average_distraction_density", 3.0) or 3.0),
            average_fatigue_onset_minutes=float(data.get("average_fatigue_onset_minutes", 35.0) or 35.0),
            focus_score_decay_per_hour=float(data.get("focus_score_decay_per_hour", 0.0) or 0.0),
            recommended_work_minutes=int(data.get("recommended_work_minutes", 25) or 25),
            recommended_break_minutes=int(data.get("recommended_break_minutes", 5) or 5),
            last_quality_score=float(data.get("last_quality_score", 0.0) or 0.0),
            personalization_weight=float(data.get("personalization_weight", 0.0) or 0.0),
            updated_at=int(data.get("updated_at", int(time.time())) or int(time.time())),
        )


@dataclass
class PersonalizedThresholds:
    profile_name: str
    source_session_count: int
    adaptation_stage: str
    personalization_weight: float

    ear_threshold: float
    drowsy_ear_threshold: float
    drowsy_closure_ratio: float
    perclos_threshold: float

    blink_rate_low_screen_max: float
    blink_rate_high_fatigue_min: float
    fatigue_head_down_min_duration: float
    phone_eye_down_min_duration: float

    score_drop_rate: float
    score_recover_rate: float
    score_target_uncertain: float
    refocus_validation_seconds: float

    def to_focus_engine_overrides(self) -> Dict[str, float]:
        return {
            "drowsy_ear_threshold": float(self.drowsy_ear_threshold),
            "drowsy_closure_ratio": float(self.drowsy_closure_ratio),
            "perclos_threshold": float(self.perclos_threshold),
            "blink_rate_low_screen_max": float(self.blink_rate_low_screen_max),
            "blink_rate_high_fatigue_min": float(self.blink_rate_high_fatigue_min),
            "fatigue_head_down_min_duration": float(self.fatigue_head_down_min_duration),
            "phone_eye_down_min_duration": float(self.phone_eye_down_min_duration),
            "score_drop_rate": float(self.score_drop_rate),
            "score_recover_rate": float(self.score_recover_rate),
            "score_target_uncertain": float(self.score_target_uncertain),
            "refocus_validation_seconds": float(self.refocus_validation_seconds),
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["focus_engine_overrides"] = self.to_focus_engine_overrides()
        return payload


class UserBaselineStore:
    """Persist and update per-user baselines using recent sessions."""

    def __init__(self, base_dir: Optional[Path] = None, max_source_sessions: int = 20):
        self.base_dir = base_dir or Path("analytics") / "baselines"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_source_sessions = max(5, int(max_source_sessions))

    @staticmethod
    def sanitize_profile_name(profile_name: str) -> str:
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

    def _baseline_path(self, profile_name: str) -> Path:
        safe = self.sanitize_profile_name(profile_name)
        return self.base_dir / f"{safe}.json"

    @staticmethod
    def _session_quality_weight(session: Dict[str, Any], duration_seconds: float) -> float:
        """Estimate per-session data quality for personalization updates."""
        quality = _safe_float(session.get("session_quality_weight"))
        if quality is None:
            quality = _safe_float(session.get("analytics_quality_score"))

        if quality is not None:
            return _clamp(float(quality), 0.12, 1.0)

        face_ratio = _safe_float(session.get("face_presence_ratio"))
        if face_ratio is None:
            face_ratio = 0.8

        uncertain_raw = _safe_float(session.get("uncertain_seconds_raw"))
        uncertain_cleaned = _safe_float(session.get("uncertain_seconds_cleaned"))
        if uncertain_raw is None:
            states = session.get("state_seconds", {}) or {}
            uncertain_raw = _safe_float(states.get("UNCERTAIN")) or 0.0
        if uncertain_cleaned is None:
            uncertain_cleaned = uncertain_raw

        uncertain_ratio = float(uncertain_raw or 0.0) / max(1.0, duration_seconds)
        recovered_ratio = 0.0
        if (uncertain_raw or 0.0) > 1e-6:
            recovered_ratio = max(0.0, float(uncertain_raw - uncertain_cleaned)) / max(1e-6, float(uncertain_raw))

        duration_score = _clamp(duration_seconds / 900.0, 0.0, 1.0)
        quality_score = (
            (duration_score * 0.28)
            + (_clamp(float(face_ratio), 0.0, 1.0) * 0.42)
            + ((1.0 - _clamp(uncertain_ratio, 0.0, 1.0)) * 0.22)
            + (_clamp(recovered_ratio, 0.0, 1.0) * 0.08)
        )
        return _clamp(quality_score, 0.12, 1.0)

    def load_baseline(self, profile_name: str) -> UserBaseline:
        path = self._baseline_path(profile_name)
        if not path.exists():
            return UserBaseline(profile_name=profile_name or "default", updated_at=int(time.time()))

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return UserBaseline.from_dict(data, profile_name=profile_name or "default")
        except Exception as exc:
            logger.warning("Failed to load baseline for '%s': %s", profile_name, exc)
            return UserBaseline(profile_name=profile_name or "default", updated_at=int(time.time()))

    def save_baseline(self, baseline: UserBaseline) -> None:
        baseline.updated_at = int(time.time())
        path = self._baseline_path(baseline.profile_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(baseline.to_dict(), f, ensure_ascii=False, indent=2)

    def merge_remote_baseline(self, profile_name: str, remote_data: Dict[str, Any]) -> UserBaseline:
        local = self.load_baseline(profile_name)
        remote = UserBaseline.from_dict(remote_data, profile_name=profile_name)

        if remote.updated_at >= local.updated_at:
            merged = remote
        else:
            merged = local

        self.save_baseline(merged)
        return merged

    def update_from_sessions(
        self,
        profile_name: str,
        sessions: Sequence[Dict[str, Any]],
        default_work: int = 25,
        default_break: int = 5,
    ) -> UserBaseline:
        profile_name = profile_name or "default"
        previous = self.load_baseline(profile_name)

        valid_sessions: List[Dict[str, Any]] = []
        for session in sessions:
            duration = _safe_float(session.get("session_seconds"))
            if duration is not None and duration >= 60.0:
                valid_sessions.append(session)

        if not valid_sessions:
            if previous.session_count <= 0:
                previous.recommended_work_minutes = int(_clamp(default_work, 15, 60))
                previous.recommended_break_minutes = int(_clamp(default_break, 3, 20))
                previous.personalization_weight = compute_personalization_weight(0)
                self.save_baseline(previous)
            return previous

        recent = list(valid_sessions[-self.max_source_sessions:])
        session_count = len(recent)
        weight = compute_personalization_weight(session_count)

        blink_rate_samples: List[tuple[float, float]] = []
        avg_ear_samples: List[tuple[float, float]] = []
        closure_ratio_samples: List[tuple[float, float]] = []
        perclos_samples: List[tuple[float, float]] = []
        avg_score_samples: List[tuple[float, float]] = []
        distraction_density_samples: List[tuple[float, float]] = []
        fatigue_onset_samples: List[tuple[float, float]] = []
        score_decay_samples: List[tuple[float, float]] = []
        work_minutes_used_samples: List[tuple[float, float]] = []
        break_minutes_used_samples: List[tuple[float, float]] = []
        quality_weights: List[float] = []

        for session in recent:
            duration = max(1.0, float(session.get("session_seconds", 1.0) or 1.0))
            quality_weight = self._session_quality_weight(session, duration)
            quality_weights.append(quality_weight)

            blink_rate = _safe_float(session.get("blink_rate_per_min"))
            if blink_rate is not None and blink_rate >= 0.0:
                blink_rate_samples.append((blink_rate, quality_weight))

            avg_ear = _safe_float(session.get("avg_ear"))
            if avg_ear is not None and avg_ear > 0.0:
                avg_ear_samples.append((avg_ear, quality_weight))

            closure = _safe_float(session.get("eye_closure_ratio"))
            if closure is not None and closure >= 0.0:
                closure_ratio_samples.append((_clamp(closure, 0.0, 1.0), quality_weight))

            perclos = _safe_float(session.get("perclos"))
            if perclos is not None and perclos >= 0.0:
                perclos_samples.append((_clamp(perclos, 0.0, 1.0), quality_weight))

            avg_score = _safe_float(session.get("avg_score_cleaned"))
            if avg_score is None:
                avg_score = _safe_float(session.get("avg_score"))
            if avg_score is not None:
                avg_score_samples.append((_clamp(avg_score, 0.0, 100.0), quality_weight))

            distractions = _safe_float(session.get("distraction_count_cleaned"))
            if distractions is None:
                distractions = _safe_float(session.get("distraction_count"))
            distractions = max(0.0, float(distractions or 0.0))
            distraction_density_samples.append((distractions / (duration / 3600.0), quality_weight))

            fatigue_onset = _safe_float(session.get("fatigue_onset_minutes"))
            if fatigue_onset is not None and fatigue_onset > 0.0:
                fatigue_onset_samples.append((fatigue_onset, quality_weight))
            else:
                states = session.get("state_seconds", {}) or {}
                drowsy_seconds = _safe_float(states.get("DROWSY_FATIGUE")) or 0.0
                if drowsy_seconds >= 8.0 and duration >= 300.0:
                    estimated = max(5.0, (duration - (drowsy_seconds * 0.5)) / 60.0)
                    fatigue_onset_samples.append((estimated, quality_weight * 0.9))

            score_decay = _safe_float(session.get("score_drop_per_hour_cleaned"))
            if score_decay is None:
                score_decay = _safe_float(session.get("score_drop_per_hour"))
            if score_decay is not None:
                score_decay_samples.append((score_decay, quality_weight))
            else:
                score_start = _safe_float(session.get("focus_score_start"))
                score_end = _safe_float(session.get("focus_score_end"))
                if score_start is not None and score_end is not None and duration >= 30.0:
                    decay = (score_start - score_end) / (duration / 3600.0)
                    score_decay_samples.append((decay, quality_weight * 0.85))

            work_used = _safe_float(session.get("work_interval_minutes_used"))
            if work_used is not None and work_used > 0.0:
                work_minutes_used_samples.append((work_used, quality_weight))

            break_used = _safe_float(session.get("break_duration_minutes_used"))
            if break_used is not None and break_used > 0.0:
                break_minutes_used_samples.append((break_used, quality_weight))

        avg_quality_score = statistics.fmean(quality_weights) if quality_weights else previous.last_quality_score
        effective_count = max(0, int(round(session_count * max(0.35, float(avg_quality_score or 0.0)))))
        weight = compute_personalization_weight(effective_count)

        blink_baseline = _robust_recent_center_weighted(
            blink_rate_samples,
            fallback=max(8.0, previous.blink_rate_baseline),
        )
        avg_ear_baseline = _robust_recent_center_weighted(
            avg_ear_samples,
            fallback=previous.avg_ear_baseline if previous.avg_ear_baseline > 0 else 0.25,
        )
        closure_baseline = _robust_recent_center_weighted(
            closure_ratio_samples,
            fallback=previous.eye_closure_ratio_baseline if previous.eye_closure_ratio_baseline > 0 else 0.12,
        )
        perclos_baseline = _robust_recent_center_weighted(
            perclos_samples,
            fallback=previous.perclos_baseline if previous.perclos_baseline > 0 else 0.08,
        )
        score_baseline = _robust_recent_center_weighted(
            avg_score_samples,
            fallback=previous.average_focus_score_baseline if previous.average_focus_score_baseline > 0 else 75.0,
        )
        distraction_baseline = _robust_recent_center_weighted(
            distraction_density_samples,
            fallback=previous.average_distraction_density if previous.average_distraction_density > 0 else 3.0,
        )
        fatigue_baseline = _robust_recent_center_weighted(
            fatigue_onset_samples,
            fallback=previous.average_fatigue_onset_minutes if previous.average_fatigue_onset_minutes > 0 else 35.0,
        )
        score_decay_baseline = _robust_recent_center_weighted(
            score_decay_samples,
            fallback=previous.focus_score_decay_per_hour,
        )

        smoothed_blink = _blend(previous.blink_rate_baseline, blink_baseline, max(0.35, weight))
        smoothed_ear = _blend(previous.avg_ear_baseline, avg_ear_baseline, max(0.35, weight))
        smoothed_closure = _blend(previous.eye_closure_ratio_baseline, closure_baseline, max(0.35, weight))
        smoothed_perclos = _blend(previous.perclos_baseline, perclos_baseline, max(0.35, weight))
        smoothed_score = _blend(previous.average_focus_score_baseline, score_baseline, max(0.35, weight))
        smoothed_distraction = _blend(previous.average_distraction_density, distraction_baseline, max(0.35, weight))
        smoothed_fatigue = _blend(previous.average_fatigue_onset_minutes, fatigue_baseline, max(0.35, weight))

        work_default = int(_clamp(default_work, 15, 60))
        break_default = int(_clamp(default_break, 3, 20))

        work_personal = _robust_recent_center_weighted(
            work_minutes_used_samples,
            fallback=float(work_default),
        )
        if fatigue_baseline > 0.0:
            work_personal = min(work_personal, fatigue_baseline * 0.82)
        work_minutes = int(round(_clamp(_blend(work_default, work_personal, max(0.25, weight)), 15, 60)))

        break_personal = _robust_recent_center_weighted(
            break_minutes_used_samples,
            fallback=float(break_default),
        )
        if smoothed_distraction > 4.0 or smoothed_score < 68.0:
            break_personal += 1.5
        break_minutes = int(round(_clamp(_blend(break_default, break_personal, max(0.25, weight)), 3, 20)))

        baseline = UserBaseline(
            profile_name=profile_name,
            session_count=len(valid_sessions),
            blink_rate_baseline=_clamp(smoothed_blink, 4.0, 45.0),
            avg_ear_baseline=_clamp(smoothed_ear, 0.15, 0.42),
            eye_closure_ratio_baseline=_clamp(smoothed_closure, 0.01, 0.75),
            perclos_baseline=_clamp(smoothed_perclos, 0.01, 0.75),
            average_focus_score_baseline=_clamp(smoothed_score, 0.0, 100.0),
            average_distraction_density=_clamp(smoothed_distraction, 0.0, 40.0),
            average_fatigue_onset_minutes=_clamp(smoothed_fatigue, 5.0, 180.0),
            focus_score_decay_per_hour=_clamp(score_decay_baseline, -90.0, 90.0),
            recommended_work_minutes=work_minutes,
            recommended_break_minutes=break_minutes,
            last_quality_score=_clamp(float(avg_quality_score or 0.0), 0.0, 1.0),
            personalization_weight=weight,
            updated_at=int(time.time()),
        )
        self.save_baseline(baseline)
        return baseline


class PersonalizationManager:
    """Map per-user baseline into adaptive threshold overrides."""

    def __init__(self, baseline_store: Optional[UserBaselineStore] = None):
        self.baseline_store = baseline_store or UserBaselineStore()

    def build_thresholds(
        self,
        profile_name: str,
        baseline: UserBaseline,
        focus_defaults: Optional[Dict[str, Any]] = None,
    ) -> PersonalizedThresholds:
        defaults = focus_defaults or {}

        session_count = max(0, int(baseline.session_count))
        weight = compute_personalization_weight(session_count)
        stage = personalization_stage(session_count)

        default_ear_threshold = float(defaults.get("ear_threshold", 0.21) or 0.21)
        default_drowsy_ear_threshold = float(defaults.get("drowsy_ear_threshold", 0.18) or 0.18)
        default_drowsy_closure = float(defaults.get("drowsy_closure_ratio", 0.30) or 0.30)
        default_perclos = float(defaults.get("perclos_threshold", 0.15) or 0.15)
        default_blink_low = float(defaults.get("blink_rate_low_screen_max", 10.0) or 10.0)
        default_blink_high = float(defaults.get("blink_rate_high_fatigue_min", 22.0) or 22.0)
        default_fatigue_head_down = float(defaults.get("fatigue_head_down_min_duration", 15.0) or 15.0)
        default_phone_eye_down = float(defaults.get("phone_eye_down_min_duration", 45.0) or 45.0)
        default_score_drop = float(defaults.get("score_drop_rate", 4.0) or 4.0)
        default_score_recover = float(defaults.get("score_recover_rate", 5.5) or 5.5)
        default_score_target_uncertain = float(defaults.get("score_target_uncertain", 78.0) or 78.0)
        default_refocus_validation = float(defaults.get("refocus_validation_seconds", 2.5) or 2.5)

        # Reuse existing EAR calibration helper instead of duplicating threshold math.
        blink_detector = BlinkDetector(
            BlinkConfig(
                ear_threshold=default_ear_threshold,
                drowsy_ear_threshold=default_drowsy_ear_threshold,
            )
        )
        open_ear = _clamp(baseline.avg_ear_baseline, 0.18, 0.38)
        blink_detector.calibrate_threshold(open_ear)
        calibrated_ear_threshold = float(blink_detector.config.ear_threshold)
        calibrated_drowsy_ear_threshold = float(blink_detector.config.drowsy_ear_threshold)

        personalized_blink_low = _clamp(baseline.blink_rate_baseline * 0.72, 5.0, 18.0)
        personalized_blink_high = _clamp(
            max(
                baseline.blink_rate_baseline * 1.35,
                personalized_blink_low + 6.0,
            ),
            12.0,
            40.0,
        )

        personalized_closure_threshold = _clamp(
            baseline.eye_closure_ratio_baseline + max(0.08, baseline.eye_closure_ratio_baseline * 0.75),
            0.12,
            0.62,
        )
        personalized_perclos_threshold = _clamp(
            baseline.perclos_baseline + max(0.05, baseline.perclos_baseline * 0.70),
            0.08,
            0.52,
        )

        personalized_fatigue_head_down = _clamp(
            8.0 + (baseline.average_fatigue_onset_minutes * 0.22),
            8.0,
            30.0,
        )
        personalized_phone_eye_down = _clamp(
            25.0 + (baseline.average_fatigue_onset_minutes * 0.45),
            18.0,
            90.0,
        )

        sensitivity = (
            (baseline.average_distraction_density * 0.18)
            + (max(0.0, baseline.focus_score_decay_per_hour) * 0.02)
        )
        personalized_score_drop = _clamp(3.0 + sensitivity, 2.0, 7.5)

        recovery_bias = (
            ((baseline.average_focus_score_baseline - 65.0) * 0.04)
            - (baseline.average_distraction_density * 0.06)
        )
        personalized_score_recover = _clamp(5.3 + recovery_bias, 3.8, 8.5)

        personalized_uncertain_target = _clamp(
            72.0 + ((baseline.average_focus_score_baseline - 70.0) * 0.24),
            66.0,
            90.0,
        )
        personalized_refocus_validation = _clamp(
            2.9 + (baseline.average_distraction_density * 0.07) - ((baseline.average_focus_score_baseline - 70.0) * 0.01),
            1.0,
            4.8,
        )

        return PersonalizedThresholds(
            profile_name=profile_name,
            source_session_count=session_count,
            adaptation_stage=stage,
            personalization_weight=weight,
            ear_threshold=_clamp(_blend(default_ear_threshold, calibrated_ear_threshold, weight), 0.15, 0.32),
            drowsy_ear_threshold=_clamp(
                _blend(default_drowsy_ear_threshold, calibrated_drowsy_ear_threshold, weight),
                0.12,
                0.28,
            ),
            drowsy_closure_ratio=_clamp(
                _blend(default_drowsy_closure, personalized_closure_threshold, weight),
                0.10,
                0.65,
            ),
            perclos_threshold=_clamp(_blend(default_perclos, personalized_perclos_threshold, weight), 0.06, 0.6),
            blink_rate_low_screen_max=_clamp(_blend(default_blink_low, personalized_blink_low, weight), 4.0, 20.0),
            blink_rate_high_fatigue_min=_clamp(_blend(default_blink_high, personalized_blink_high, weight), 10.0, 45.0),
            fatigue_head_down_min_duration=_clamp(
                _blend(default_fatigue_head_down, personalized_fatigue_head_down, weight),
                6.0,
                30.0,
            ),
            phone_eye_down_min_duration=_clamp(
                _blend(default_phone_eye_down, personalized_phone_eye_down, weight),
                15.0,
                120.0,
            ),
            score_drop_rate=_clamp(_blend(default_score_drop, personalized_score_drop, weight), 1.5, 8.0),
            score_recover_rate=_clamp(_blend(default_score_recover, personalized_score_recover, weight), 2.5, 9.5),
            score_target_uncertain=_clamp(
                _blend(default_score_target_uncertain, personalized_uncertain_target, weight),
                60.0,
                92.0,
            ),
            refocus_validation_seconds=_clamp(
                _blend(default_refocus_validation, personalized_refocus_validation, weight),
                0.6,
                6.0,
            ),
        )

    def get_or_build_baseline(
        self,
        profile_name: str,
        sessions: Sequence[Dict[str, Any]],
        default_work: int = 25,
        default_break: int = 5,
    ) -> UserBaseline:
        return self.baseline_store.update_from_sessions(
            profile_name=profile_name,
            sessions=sessions,
            default_work=default_work,
            default_break=default_break,
        )
