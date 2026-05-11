from app.ui.context_dialogs import (
    FOCUS_AIRPORT_DATA,
    FOCUS_ROUTE_PRESETS,
    FOCUS_ROUTE_SCHEDULE,
    FOCUS_ROUTE_SLOT_MINUTES,
    _nearest_focus_routes,
    _round_minutes_to_five,
)


def test_focus_route_durations_are_rounded_to_five_minutes():
    assert _round_minutes_to_five(56) == 55
    assert _round_minutes_to_five(58) == 60
    assert all(int(route["duration_minutes"]) % 5 == 0 for route in FOCUS_ROUTE_PRESETS)


def test_nearest_focus_routes_display_in_ascending_duration_order():
    routes = _nearest_focus_routes(47, limit=3, from_code="DAD")
    durations = [int(route["duration_minutes"]) for route in routes]
    assert durations == sorted(durations)
    assert all(duration % 5 == 0 for duration in durations)


def test_focus_route_schedule_covers_every_five_minute_slot_for_each_origin():
    expected = set(FOCUS_ROUTE_SLOT_MINUTES)
    for origin in FOCUS_AIRPORT_DATA:
        slots = {
            int(route["duration_minutes"])
            for route in FOCUS_ROUTE_SCHEDULE
            if route["from_code"] == origin
        }
        assert expected <= slots


def test_nearest_focus_routes_include_exact_slot_for_current_origin():
    for target in FOCUS_ROUTE_SLOT_MINUTES:
        routes = _nearest_focus_routes(target, limit=3, from_code="DAD")
        assert any(int(route["duration_minutes"]) == target for route in routes)
