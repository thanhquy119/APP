"""Go/No-Go helpers for Focus Reset Game."""

from __future__ import annotations

from .config import GoNoGoConfig
from .game_logic import build_trials
from .metrics import compute_summary
from .models import MetricSummary, TrialResult, TrialSpec


def build_gonogo_trials(cfg: GoNoGoConfig, duration_s: int | None = None) -> list[TrialSpec]:
    """Build trial timeline for a Go/No-Go phase."""
    phase_seconds = int(duration_s if duration_s is not None else cfg.round_duration_s)
    return build_trials(
        duration_seconds=max(10, phase_seconds),
        stimulus_duration_ms=int(cfg.stimulus_duration_ms),
        inter_stimulus_ms=int(cfg.inter_stimulus_ms),
        target_probability=float(cfg.target_probability),
    )


def summarize_gonogo(
    results: list[TrialResult],
    extra_commissions: int = 0,
    baseline_avg_rt_ms: float | None = None,
) -> MetricSummary:
    """Compute standard Go/No-Go session metrics."""
    return compute_summary(
        results=results,
        extra_commissions=extra_commissions,
        baseline_avg_rt_ms=baseline_avg_rt_ms,
    )


def gonogo_focus_score(summary: MetricSummary) -> float:
    """Project Go/No-Go metrics into one 0-100 game score."""
    weighted = (summary.focus_stability * 0.7) + (summary.accuracy * 0.3)
    return max(0.0, min(100.0, weighted))
