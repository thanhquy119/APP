"""Sequence Memory helpers for Focus Reset Game."""

from __future__ import annotations

import random
from statistics import fmean

from .config import SequenceConfig
from .models import SequenceRoundResult, SequenceSummary


def build_round_lengths(cfg: SequenceConfig) -> list[int]:
    """Build round lengths with gradual difficulty increase."""
    rounds = max(1, int(cfg.rounds))
    start = int(cfg.start_length)
    min_len = int(cfg.min_length)
    max_len = max(start, int(cfg.max_length))

    lengths: list[int] = []
    current = max(min_len, min(start, max_len))
    for idx in range(rounds):
        if idx > 0 and idx % 2 == 0 and current < max_len:
            current += 1
        lengths.append(current)

    return lengths


def build_sequence(symbols: tuple[str, ...], length: int) -> list[str]:
    pool = list(symbols) or ["A", "S", "D", "F"]
    return [random.choice(pool) for _ in range(max(1, int(length)))]


def evaluate_sequence(results: list[SequenceRoundResult]) -> SequenceSummary:
    """Aggregate sequence-memory performance."""
    if not results:
        return SequenceSummary(
            accuracy=0.0,
            max_sequence_length=0,
            average_response_time_ms=0.0,
            focus_consistency_score=0.0,
            rounds=0,
        )

    rounds = len(results)
    correct_rounds = sum(1 for r in results if r.correct)
    accuracy = (correct_rounds / max(1, rounds)) * 100.0
    max_len = max((r.sequence_length for r in results if r.correct), default=0)
    avg_response = fmean(max(0.0, r.response_time_ms) for r in results)
    mistakes = sum(max(0, int(r.mistakes)) for r in results)

    speed_score = max(0.0, 100.0 - (avg_response / 45.0))
    mistake_penalty = min(35.0, mistakes * 4.0)
    consistency = (accuracy * 0.65) + (speed_score * 0.35) - mistake_penalty

    return SequenceSummary(
        accuracy=accuracy,
        max_sequence_length=max_len,
        average_response_time_ms=avg_response,
        focus_consistency_score=max(0.0, min(100.0, consistency)),
        rounds=rounds,
    )


def sequence_focus_score(summary: SequenceSummary) -> float:
    return max(0.0, min(100.0, summary.focus_consistency_score))
