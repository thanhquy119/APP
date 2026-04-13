"""Data models for Focus Reset Game."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrialSpec:
    index: int
    is_target: bool
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class TrialResult:
    is_target: bool
    reaction_time_ms: float | None
    commission_error: bool
    omission_error: bool
    correct_inhibition: bool


@dataclass(frozen=True)
class MetricSummary:
    average_reaction_ms: float
    reaction_variability_ms: float
    accuracy: float
    commission_errors: int
    omission_errors: int
    focus_stability: float
    total_trials: int
    hit_count: int
    correct_inhibition_count: int


@dataclass(frozen=True)
class SequenceRoundResult:
    round_index: int
    sequence_length: int
    correct: bool
    response_time_ms: float
    mistakes: int


@dataclass(frozen=True)
class SequenceSummary:
    accuracy: float
    max_sequence_length: int
    average_response_time_ms: float
    focus_consistency_score: float
    rounds: int


@dataclass(frozen=True)
class VisualRoundSpec:
    round_index: int
    rows: int
    cols: int
    target_index: int
    target_symbol: str
    distractor_symbol: str


@dataclass(frozen=True)
class VisualRoundResult:
    round_index: int
    correct: bool
    search_time_ms: float
    miss_clicks: int
    timeout: bool


@dataclass(frozen=True)
class VisualSummary:
    accuracy: float
    miss_click_count: int
    average_completion_time_ms: float
    visual_focus_score: float
    rounds: int


@dataclass
class SessionSummary:
    average_reaction_ms: float
    reaction_variability_ms: float
    accuracy: float
    commission_errors: int
    omission_errors: int
    focus_stability: float
    comparison: str
    feedback: str
    best_game: str
    weakest_game: str
    game_scores: dict[str, float] = field(default_factory=dict)
