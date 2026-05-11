from app.logic.task_context import TaskContextClassifier, TaskContextSample


def make_sample(title: str, process_name: str, *, window_handle: int = 0) -> TaskContextSample:
    return TaskContextSample(
        timestamp=1000.0,
        window_title=title,
        process_name=process_name,
        process_id=123,
        app_id=process_name.lower() or "unknown",
        window_handle=window_handle,
    )


def test_arc_tiktok_tab_title_is_distracting():
    classifier = TaskContextClassifier()

    sample = classifier.annotate(make_sample("TikTok - For You - Arc", "arc.exe"))

    assert sample.category == "distracting"
    assert sample.reason == "distracting keyword"


def test_default_game_app_is_distracting_even_with_empty_saved_apps():
    classifier = TaskContextClassifier()
    classifier.update_from_app_config({"task_context_distracting_apps": ""})

    sample = classifier.annotate(make_sample("", "steam.exe"))

    assert sample.category == "distracting"
    assert sample.reason == "distracting app"


def test_default_keywords_survive_saved_custom_lists():
    classifier = TaskContextClassifier()
    classifier.update_from_app_config({
        "task_context_task_apps": "customwork.exe",
        "task_context_distracting_keywords": "custombadsite",
    })

    codex = classifier.annotate(make_sample("Codex", "codex.exe"))
    tiktok = classifier.annotate(make_sample("TikTok - Make Your Day", "arc.exe"))

    assert codex.category == "task_related"
    assert tiktok.category == "distracting"


def test_common_work_app_defaults_are_task_related():
    classifier = TaskContextClassifier()
    classifier.update_from_app_config({"task_context_task_apps": ""})

    sample = classifier.annotate(make_sample("", "code.exe"))

    assert sample.category == "task_related"
    assert sample.reason == "task app"


def test_word_foreground_is_task_related_with_low_risk():
    classifier = TaskContextClassifier()

    sample = classifier.annotate(make_sample("Document1 - Word", "winword.exe"))
    stats = classifier.compute_stats(now=sample.timestamp)

    assert sample.category == "task_related"
    assert stats.risk_score <= 0.14


def test_task_manager_foreground_is_neutral_not_background_scan():
    classifier = TaskContextClassifier()

    sample = classifier.annotate(make_sample("Task Manager", "taskmgr.exe"))
    stats = classifier.compute_stats(now=sample.timestamp)

    assert sample.category == "neutral"
    assert stats.risk_score <= 0.16


def test_codex_defaults_to_task_related_context():
    classifier = TaskContextClassifier()
    classifier.update_from_app_config({"task_context_task_apps": ""})

    sample = classifier.annotate(make_sample("Codex", "codex.exe"))

    assert sample.category == "task_related"
    assert sample.reason == "task app"


def test_focusguardian_excluded_context_does_not_create_risk():
    classifier = TaskContextClassifier()

    sample = classifier.annotate(make_sample("FocusGuardian", "focusguardian.exe"))
    stats = classifier.compute_stats(now=sample.timestamp)

    assert sample.category == "excluded"
    assert stats.risk_score == 0.0
    assert stats.total_samples == 0


def test_startup_unknown_context_has_low_risk():
    classifier = TaskContextClassifier()

    sample = classifier.annotate(make_sample("Untitled", "unknownapp.exe"))
    stats = classifier.compute_stats(now=sample.timestamp)

    assert sample.category == "unknown"
    assert stats.risk_score <= 0.18


def test_current_tiktok_raises_risk_after_work_context():
    classifier = TaskContextClassifier()

    work = make_sample("Codex", "codex.exe")
    work.timestamp = 1000.0
    classifier.annotate(work)
    work_stats = classifier.compute_stats(now=1000.0)

    tiktok = make_sample("TikTok - Make Your Day", "arc.exe")
    tiktok.timestamp = 1005.0
    classifier.annotate(tiktok)
    tiktok_stats = classifier.compute_stats(now=1005.0)

    assert work_stats.risk_score <= 0.14
    assert tiktok_stats.risk_score >= 0.86
    assert tiktok_stats.risk_score > work_stats.risk_score


def test_expanded_social_and_game_titles_are_distracting():
    classifier = TaskContextClassifier()

    shorts = classifier.annotate(make_sample("YouTube Shorts - Arc", "arc.exe"))
    game = classifier.annotate(make_sample("Genshin Impact Daily Login", "chrome.exe"))

    assert shorts.category == "distracting"
    assert game.category == "distracting"


def test_auxiliary_browser_context_text_can_mark_arc_distracting():
    classifier = TaskContextClassifier()
    sample = make_sample("New Tab", "arc.exe")
    sample.context_text = "(12)TikTok - Make Your Day"

    annotated = classifier.annotate(sample)

    assert annotated.category == "distracting"
    assert annotated.reason == "distracting keyword"


def test_privacy_safe_context_does_not_persist_title_or_window_handle():
    sample = make_sample("Private Browser Title", "arc.exe", window_handle=98765)
    sample.context_text = "(12)TikTok - Make Your Day"

    payload = sample.to_privacy_safe_dict()

    assert "window_title" not in payload
    assert "window_handle" not in payload
    assert "context_text" not in payload
    assert payload["process_name"] == "arc.exe"
