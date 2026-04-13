"""Persistence layer for Focus Reset Game session history."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import MetricSummary, SequenceSummary, SessionSummary, VisualSummary


class SessionStorage:
    """Simple JSON storage for session history with optional CSV export."""

    def __init__(self, history_path: Path):
        self.history_path = history_path
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []

        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass

        return []

    def append(self, session_record: dict[str, Any]) -> None:
        rows = self.load()
        rows.append(session_record)

        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(rows[-1000:], f, indent=2, ensure_ascii=False)

    def export_csv(self, target_path: Path | None = None) -> Path:
        records = self.load()
        out = target_path or self.history_path.with_suffix(".csv")
        out.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "timestamp",
            "baseline_rt_ms",
            "gonogo_rt_ms",
            "reaction_variability_ms",
            "accuracy",
            "commission_errors",
            "omission_errors",
            "focus_stability",
            "comparison",
            "best_game",
            "weakest_game",
            "score_gonogo",
            "score_sequence",
            "score_visual",
            "sequence_accuracy",
            "sequence_max_length",
            "visual_accuracy",
            "visual_miss_clicks",
        ]

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in records:
                writer.writerow({k: item.get(k, "") for k in fieldnames})

        return out


def build_session_record(
    session_summary: SessionSummary,
    baseline_summary: MetricSummary | None = None,
    gonogo_summary: MetricSummary | None = None,
    sequence_summary: SequenceSummary | None = None,
    visual_summary: VisualSummary | None = None,
) -> dict[str, Any]:
    """Create one normalized history row."""
    game_scores = session_summary.game_scores or {}
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "baseline_rt_ms": round(float((baseline_summary.average_reaction_ms if baseline_summary else 0.0)), 2),
        "gonogo_rt_ms": round(float((gonogo_summary.average_reaction_ms if gonogo_summary else 0.0)), 2),
        "reaction_variability_ms": round(float(session_summary.reaction_variability_ms), 2),
        "accuracy": round(float(session_summary.accuracy), 2),
        "commission_errors": int(session_summary.commission_errors),
        "omission_errors": int(session_summary.omission_errors),
        "focus_stability": round(float(session_summary.focus_stability), 2),
        "comparison": str(session_summary.comparison),
        "best_game": str(session_summary.best_game),
        "weakest_game": str(session_summary.weakest_game),
        "score_gonogo": round(float(game_scores.get("Go/No-Go", 0.0)), 2),
        "score_sequence": round(float(game_scores.get("Sequence Memory", 0.0)), 2),
        "score_visual": round(float(game_scores.get("Visual Search", 0.0)), 2),
        "sequence_accuracy": round(float((sequence_summary.accuracy if sequence_summary else 0.0)), 2),
        "sequence_max_length": int((sequence_summary.max_sequence_length if sequence_summary else 0)),
        "visual_accuracy": round(float((visual_summary.accuracy if visual_summary else 0.0)), 2),
        "visual_miss_clicks": int((visual_summary.miss_click_count if visual_summary else 0)),
    }
