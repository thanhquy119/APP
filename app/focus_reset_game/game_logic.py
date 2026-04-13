"""Core trial generation and evaluation logic for Focus Reset Game."""

from __future__ import annotations

import random
from typing import Optional

from .models import TrialResult, TrialSpec


def build_trials(
    duration_seconds: int,
    stimulus_duration_ms: int,
    inter_stimulus_ms: int,
    target_probability: float,
) -> list[TrialSpec]:
    """Generate a deterministic timeline of go/no-go trials."""
    slot_ms = stimulus_duration_ms + inter_stimulus_ms
    total_ms = max(slot_ms, int(duration_seconds * 1000))
    trial_count = max(1, total_ms // slot_ms)

    p_target = max(0.6, min(0.85, float(target_probability)))

    trials: list[TrialSpec] = []
    for idx in range(trial_count):
        is_target = random.random() < p_target
        start = idx * slot_ms
        end = start + stimulus_duration_ms
        trials.append(TrialSpec(index=idx, is_target=is_target, start_ms=start, end_ms=end))

    # Keep at least one target and one no-go for meaningful metrics.
    if trials and all(t.is_target for t in trials):
        trials[-1] = TrialSpec(
            index=trials[-1].index,
            is_target=False,
            start_ms=trials[-1].start_ms,
            end_ms=trials[-1].end_ms,
        )
    if trials and all(not t.is_target for t in trials):
        trials[0] = TrialSpec(
            index=trials[0].index,
            is_target=True,
            start_ms=trials[0].start_ms,
            end_ms=trials[0].end_ms,
        )

    return trials


def active_trial_at(
    elapsed_ms: int,
    trials: list[TrialSpec],
    stimulus_duration_ms: int,
    trial_slot_ms: int,
) -> tuple[Optional[int], Optional[int]]:
    """Return active trial index and within-trial reaction time if visible."""
    if elapsed_ms < 0 or not trials:
        return None, None

    idx = elapsed_ms // trial_slot_ms
    if idx < 0 or idx >= len(trials):
        return None, None

    within = elapsed_ms - idx * trial_slot_ms
    if within >= stimulus_duration_ms:
        return None, None

    return int(idx), int(within)


def evaluate_trials(
    trials: list[TrialSpec],
    responses_by_trial_ms: dict[int, int],
) -> list[TrialResult]:
    """Convert raw responses into per-trial outcomes."""
    results: list[TrialResult] = []

    for trial in trials:
        rt = responses_by_trial_ms.get(trial.index)
        omission = trial.is_target and rt is None
        commission = (not trial.is_target) and rt is not None
        inhibition = (not trial.is_target) and rt is None
        results.append(
            TrialResult(
                is_target=trial.is_target,
                reaction_time_ms=float(rt) if rt is not None else None,
                commission_error=commission,
                omission_error=omission,
                correct_inhibition=inhibition,
            )
        )

    return results
