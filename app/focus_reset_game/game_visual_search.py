"""Visual Search helpers for Focus Reset Game."""

from __future__ import annotations

import random
from statistics import fmean

from .config import VisualSearchConfig
from .models import VisualRoundResult, VisualRoundSpec, VisualSummary


_SYMBOL_PAIRS: tuple[tuple[str, str], ...] = (
    ("E", "F"),
    ("P", "R"),
    ("8", "B"),
    ("X", "K"),
)


def build_visual_specs(cfg: VisualSearchConfig) -> list[VisualRoundSpec]:
    """Build visual-search rounds with increasing grid complexity."""
    rounds = max(1, int(cfg.rounds))
    start = int(cfg.grid_start)
    end = max(start, int(cfg.grid_max))

    specs: list[VisualRoundSpec] = []
    for idx in range(rounds):
        ratio = idx / max(1, rounds - 1)
        size = int(round(start + (end - start) * ratio))
        rows = max(3, size)
        cols = max(3, size)
        total = rows * cols

        target, distractor = random.choice(_SYMBOL_PAIRS)
        specs.append(
            VisualRoundSpec(
                round_index=idx,
                rows=rows,
                cols=cols,
                target_index=random.randrange(total),
                target_symbol=target,
                distractor_symbol=distractor,
            )
        )

    return specs


def evaluate_visual(results: list[VisualRoundResult]) -> VisualSummary:
    """Aggregate visual-search performance into one summary."""
    if not results:
        return VisualSummary(
            accuracy=0.0,
            miss_click_count=0,
            average_completion_time_ms=0.0,
            visual_focus_score=0.0,
            rounds=0,
        )

    rounds = len(results)
    correct_rounds = sum(1 for r in results if r.correct)
    accuracy = (correct_rounds / max(1, rounds)) * 100.0
    misses = sum(max(0, int(r.miss_clicks)) for r in results)

    valid_times = [max(0.0, r.search_time_ms) for r in results if r.search_time_ms > 0]
    average_time = fmean(valid_times) if valid_times else 0.0

    timeout_penalty = sum(1 for r in results if r.timeout) * 8.0
    miss_penalty = min(30.0, misses * 2.0)
    speed_score = max(0.0, 100.0 - (average_time / 55.0))
    score = (accuracy * 0.65) + (speed_score * 0.35) - timeout_penalty - miss_penalty

    return VisualSummary(
        accuracy=accuracy,
        miss_click_count=misses,
        average_completion_time_ms=average_time,
        visual_focus_score=max(0.0, min(100.0, score)),
        rounds=rounds,
    )


def visual_focus_score(summary: VisualSummary) -> float:
    return max(0.0, min(100.0, summary.visual_focus_score))
