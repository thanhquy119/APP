"""Metrics and scoring helpers for Focus Reset Game."""

from __future__ import annotations

from statistics import fmean, pstdev

from .models import MetricSummary, SequenceSummary, SessionSummary, TrialResult, VisualSummary


def compute_summary(
    results: list[TrialResult],
    extra_commissions: int = 0,
    baseline_avg_rt_ms: float | None = None,
) -> MetricSummary:
    """Compute aggregate metrics for one block or full session."""
    if not results:
        return MetricSummary(
            average_reaction_ms=0.0,
            reaction_variability_ms=0.0,
            accuracy=0.0,
            commission_errors=0,
            omission_errors=0,
            focus_stability=0.0,
            total_trials=0,
            hit_count=0,
            correct_inhibition_count=0,
        )

    target_trials = [r for r in results if r.is_target]
    hit_rts = [r.reaction_time_ms for r in target_trials if r.reaction_time_ms is not None]

    hit_count = len(hit_rts)
    omission_errors = sum(1 for r in results if r.omission_error)
    commission_errors = sum(1 for r in results if r.commission_error) + max(0, int(extra_commissions))
    correct_inhibitions = sum(1 for r in results if r.correct_inhibition)

    avg_rt = fmean(hit_rts) if hit_rts else 0.0
    rt_var = pstdev(hit_rts) if len(hit_rts) > 1 else 0.0

    total_trials = len(results)
    correct_actions = hit_count + correct_inhibitions
    accuracy = (correct_actions / max(1, total_trials)) * 100.0

    stability = compute_focus_stability(
        average_rt_ms=avg_rt,
        rt_variability_ms=rt_var,
        commission_errors=commission_errors,
        omission_errors=omission_errors,
        total_trials=total_trials,
        baseline_avg_rt_ms=baseline_avg_rt_ms,
    )

    return MetricSummary(
        average_reaction_ms=avg_rt,
        reaction_variability_ms=rt_var,
        accuracy=accuracy,
        commission_errors=commission_errors,
        omission_errors=omission_errors,
        focus_stability=stability,
        total_trials=total_trials,
        hit_count=hit_count,
        correct_inhibition_count=correct_inhibitions,
    )


def compute_focus_stability(
    average_rt_ms: float,
    rt_variability_ms: float,
    commission_errors: int,
    omission_errors: int,
    total_trials: int,
    baseline_avg_rt_ms: float | None,
) -> float:
    """
    Compute a 0-100 focus stability score.

    Heuristic combines error pressure, RT variability, and slowdown vs baseline.
    """
    if total_trials <= 0:
        return 0.0

    error_rate = (commission_errors + omission_errors) / max(1, total_trials)
    error_penalty = min(55.0, error_rate * 120.0)

    variability_penalty = min(28.0, rt_variability_ms / 10.0)

    shift_penalty = 0.0
    if baseline_avg_rt_ms is not None and baseline_avg_rt_ms > 0 and average_rt_ms > 0:
        positive_shift = max(0.0, average_rt_ms - baseline_avg_rt_ms)
        shift_penalty = min(17.0, positive_shift / 12.0)

    score = 100.0 - error_penalty - variability_penalty - shift_penalty
    return max(0.0, min(100.0, score))


def compare_baseline(baseline: MetricSummary, session: MetricSummary) -> str:
    """Return Better / Similar / Worse comparison label."""
    if baseline.total_trials <= 0 or session.total_trials <= 0:
        return "Similar"

    baseline_index = _performance_index(baseline)
    session_index = _performance_index(session)
    delta = session_index - baseline_index

    if delta >= 2.5:
        return "Better"
    if delta <= -2.5:
        return "Worse"
    return "Similar"


def build_feedback(summary: MetricSummary, comparison: str) -> str:
    """Build short user-friendly final feedback sentence."""
    if summary.focus_stability >= 78 and comparison in {"Better", "Similar"}:
        return "Mức tập trung của bạn đang ổn định."
    if summary.focus_stability >= 60:
        return "Bạn có dấu hiệu mệt nhẹ về chú ý."
    return "Nên nghỉ thêm một chút trước khi tiếp tục."


def build_session_summary(
    baseline: MetricSummary | None,
    gonogo: MetricSummary | None,
    sequence: SequenceSummary | None,
    visual: VisualSummary | None,
) -> SessionSummary:
    """Combine game-level metrics into one final session summary."""
    game_scores: dict[str, float] = {}

    if gonogo is not None and gonogo.total_trials > 0:
        game_scores["Go/No-Go"] = _gonogo_score(gonogo)
    if sequence is not None and sequence.rounds > 0:
        game_scores["Sequence Memory"] = _sequence_score(sequence)
    if visual is not None and visual.rounds > 0:
        game_scores["Visual Search"] = _visual_score(visual)

    if gonogo is not None and gonogo.total_trials > 0:
        avg_rt = gonogo.average_reaction_ms
        rt_var = gonogo.reaction_variability_ms
        accuracy = gonogo.accuracy
        commission = gonogo.commission_errors
        omission = gonogo.omission_errors
        comparison = compare_baseline(baseline or compute_summary([]), gonogo)
    else:
        avg_rt = 0.0
        rt_var = 0.0
        accuracy_candidates = []
        if sequence is not None:
            accuracy_candidates.append(sequence.accuracy)
        if visual is not None:
            accuracy_candidates.append(visual.accuracy)
        accuracy = fmean(accuracy_candidates) if accuracy_candidates else 0.0
        commission = 0
        omission = 0
        comparison = "Similar"

    focus_stability = fmean(game_scores.values()) if game_scores else 0.0

    best_game = "-"
    weakest_game = "-"
    if game_scores:
        best_game = max(game_scores, key=game_scores.get)
        weakest_game = min(game_scores, key=game_scores.get)

    summary = SessionSummary(
        average_reaction_ms=avg_rt,
        reaction_variability_ms=rt_var,
        accuracy=accuracy,
        commission_errors=commission,
        omission_errors=omission,
        focus_stability=max(0.0, min(100.0, focus_stability)),
        comparison=comparison,
        feedback="",
        best_game=best_game,
        weakest_game=weakest_game,
        game_scores=game_scores,
    )
    summary.feedback = build_session_feedback(summary)
    return summary


def build_session_feedback(summary: SessionSummary) -> str:
    """Generate final guidance text from aggregate session metrics."""
    if summary.focus_stability >= 80.0:
        return "Phục hồi rất tốt. Bạn có thể tiếp tục học/làm việc."

    if summary.focus_stability >= 65.0:
        if summary.weakest_game != "-":
            weakest = _game_name_vn(summary.weakest_game)
            return f"Phục hồi khá ổn. Nên bổ sung nghỉ ngắn cho {weakest}."
        return "Phục hồi khá ổn. Nghỉ thêm 1-2 phút để ổn định hơn."

    if summary.comparison == "Worse":
        return "Hiệu suất thấp hơn baseline. Nên nghỉ dài hơn trước khi quay lại."

    return "Tập trung chưa ổn định. Thử uống nước, chỉnh tư thế, rồi làm thêm 1 lượt phục hồi ngắn."


def _performance_index(summary: MetricSummary) -> float:
    error_total = summary.commission_errors + summary.omission_errors
    error_rate_pct = (error_total / max(1, summary.total_trials)) * 100.0

    rt_component = 0.0 if summary.average_reaction_ms <= 0 else max(0.0, 320.0 - summary.average_reaction_ms) / 40.0
    variability_component = max(0.0, 120.0 - summary.reaction_variability_ms) / 35.0

    return (
        (summary.accuracy / 20.0)
        + rt_component
        + variability_component
        - (error_rate_pct / 20.0)
    )


def _gonogo_score(summary: MetricSummary) -> float:
    weighted = (summary.focus_stability * 0.7) + (summary.accuracy * 0.3)
    return max(0.0, min(100.0, weighted))


def _sequence_score(summary: SequenceSummary) -> float:
    speed_score = max(0.0, 100.0 - (summary.average_response_time_ms / 45.0))
    weighted = (summary.accuracy * 0.65) + (speed_score * 0.15) + (summary.focus_consistency_score * 0.2)
    return max(0.0, min(100.0, weighted))


def _visual_score(summary: VisualSummary) -> float:
    speed_score = max(0.0, 100.0 - (summary.average_completion_time_ms / 55.0))
    weighted = (summary.accuracy * 0.7) + (speed_score * 0.1) + (summary.visual_focus_score * 0.2)
    return max(0.0, min(100.0, weighted))


def _game_name_vn(value: str) -> str:
    mapping = {
        "Go/No-Go": "Go/No-Go",
        "Sequence Memory": "Ghi nhớ chuỗi",
        "Visual Search": "Tìm kiếm thị giác",
    }
    return mapping.get(value, value)
