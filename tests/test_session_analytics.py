from __future__ import annotations

from datetime import datetime

from app.logic.session_analytics import SessionAnalyticsStore


def _ts(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp())


def test_work_rhythm_summary_builds_day_week_month_periods(tmp_path):
    store = SessionAnalyticsStore(base_dir=tmp_path)
    profile = store._default_profile("tester")
    profile["sessions"] = [
        {
            "timestamp": _ts("2026-05-08T09:10:00"),
            "session_seconds_cleaned": 1800,
            "focus_seconds_cleaned": 1500,
            "avg_score_cleaned": 86,
            "distraction_count_cleaned": 1,
            "break_count": 1,
            "state_seconds": {
                "ON_SCREEN_READING": 1400,
                "OFFSCREEN_WRITING": 100,
                "PHONE_DISTRACTION": 60,
                "DROWSY_FATIGUE": 0,
                "AWAY": 0,
                "UNCERTAIN": 240,
            },
        },
        {
            "timestamp": _ts("2026-05-07T15:30:00"),
            "session_seconds_cleaned": 1200,
            "focus_seconds_cleaned": 720,
            "avg_score_cleaned": 70,
            "distraction_count_cleaned": 3,
            "break_count": 0,
            "state_seconds": {
                "ON_SCREEN_READING": 720,
                "PHONE_DISTRACTION": 180,
                "DROWSY_FATIGUE": 120,
                "AWAY": 0,
                "UNCERTAIN": 180,
            },
        },
    ]
    store.save_profile("tester", profile)

    summary = store.build_work_rhythm_summary(
        "tester",
        now_ts=_ts("2026-05-08T18:00:00"),
    )

    day = summary["periods"]["day"]
    week = summary["periods"]["week"]
    month = summary["periods"]["month"]

    assert day["session_count"] == 1
    assert day["focus_seconds"] == 1500
    assert day["focus_ratio"] == 1500 / 1800
    assert day["best_bucket_label"] == "09h"
    assert week["session_count"] == 2
    assert month["session_count"] == 2
    assert day["state_distribution"]["focused"]["seconds"] == 1500
    assert day["insights"]


def test_work_rhythm_summary_can_include_live_session(tmp_path):
    store = SessionAnalyticsStore(base_dir=tmp_path)
    store.save_profile("tester", store._default_profile("tester"))

    summary = store.build_work_rhythm_summary(
        "tester",
        live_session={
            "timestamp": _ts("2026-05-08T10:00:00"),
            "session_seconds": 300,
            "focus_seconds": 240,
            "avg_score": 82,
            "distraction_count": 0,
            "state_seconds": {"ON_SCREEN_READING": 240, "UNCERTAIN": 60},
        },
        now_ts=_ts("2026-05-08T10:05:00"),
    )

    day = summary["periods"]["day"]
    assert day["session_count"] == 1
    assert day["live_session_included"] is True
    assert day["focus_seconds"] == 240


def test_work_rhythm_summary_combines_saved_day_and_live_session(tmp_path):
    store = SessionAnalyticsStore(base_dir=tmp_path)
    profile = store._default_profile("tester")
    profile["sessions"] = [
        {
            "timestamp": _ts("2026-05-08T08:00:00"),
            "session_seconds_cleaned": 4 * 3600,
            "focus_seconds_cleaned": 3 * 3600,
            "avg_score_cleaned": 84,
            "distraction_count_cleaned": 2,
            "break_count": 3,
            "state_seconds": {"ON_SCREEN_READING": 3 * 3600, "UNCERTAIN": 3600},
        }
    ]
    store.save_profile("tester", profile)

    summary = store.build_work_rhythm_summary(
        "tester",
        live_session={
            "timestamp": _ts("2026-05-08T14:00:00"),
            "session_seconds": 1800,
            "focus_seconds": 1200,
            "avg_score": 90,
            "distraction_count": 1,
            "break_count": 1,
            "state_seconds": {"ON_SCREEN_READING": 1200, "UNCERTAIN": 600},
        },
        now_ts=_ts("2026-05-08T14:30:00"),
    )

    day = summary["periods"]["day"]

    assert day["session_count"] == 2
    assert day["live_session_included"] is True
    assert day["total_seconds"] == (4 * 3600) + 1800
    assert day["focus_seconds"] == (3 * 3600) + 1200
    assert day["distraction_count"] == 3
    assert day["break_count"] == 4


def test_reset_profile_baseline_keeps_history_but_excludes_old_sessions(tmp_path):
    store = SessionAnalyticsStore(base_dir=tmp_path)
    profile = store._default_profile("tester")
    profile["sessions"] = [
        {
            "timestamp": _ts("2026-05-01T09:00:00"),
            "session_seconds_cleaned": 1800,
            "focus_seconds_cleaned": 1500,
            "avg_score_cleaned": 84,
            "analytics_quality_score": 0.9,
            "face_presence_ratio": 0.95,
        }
    ]
    store.save_profile("tester", profile)

    status = store.reset_profile_baseline("tester", default_work=30, default_break=5)
    reloaded = store.load_profile("tester")

    assert len(reloaded["sessions"]) == 1
    assert reloaded["baseline_reset_at"] > _ts("2026-05-01T09:00:00")
    assert status["eligible_sessions"] == 0
    assert status["label"] == "Chưa đủ dữ liệu"
    assert reloaded["recommendation"]["based_on_sessions"] == 0
