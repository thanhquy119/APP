from __future__ import annotations

from app.logic.supabase_sync import SupabaseConfig, SupabaseSessionSync, supabase_missing_config_message


class FakeClient:
    def __init__(self):
        self.insert_calls = []
        self.upsert_calls = []
        self.rows = []

    def insert(self, table_name, payload):
        self.insert_calls.append((table_name, payload))
        return True

    def upsert(self, table_name, payload, *, on_conflict):
        self.upsert_calls.append((table_name, payload, on_conflict))
        return True

    def select(self, table_name, *, select="*", filters=None, limit=None):
        return list(self.rows)


def _sync_with_fake_client() -> tuple[SupabaseSessionSync, FakeClient]:
    sync = SupabaseSessionSync(
        SupabaseConfig(
            enabled=True,
            url="https://example.supabase.co",
            api_key="eyJfake",
        )
    )
    fake = FakeClient()
    sync._client = fake
    return sync, fake


def test_append_session_maps_cloud_row_to_supabase_payload():
    sync, fake = _sync_with_fake_client()

    ok = sync.append_session(
        {
            "timestamp": 1778172418,
            "profile_name": "thanhquy",
            "session_seconds": 120,
            "focus_seconds": 98,
            "avg_score": 82.5,
            "state_seconds": {"ON_SCREEN_READING": 80, "UNCERTAIN": 5},
            "session_exit": {"focus_rating": ""},
        }
    )

    assert ok is True
    table_name, payload = fake.insert_calls[0]
    assert table_name == "focusguardian_sessions"
    assert payload["profile_name"] == "thanhquy"
    assert payload["session_seconds"] == 120
    assert payload["state_on_screen"] == 80
    assert payload["session_exit_focus_rating"] is None
    assert payload["raw_payload"]["profile_name"] == "thanhquy"


def test_profile_settings_upsert_adds_json_payload_and_conflict_key():
    sync, fake = _sync_with_fake_client()

    ok = sync.upsert_profile_settings(
        "thanhquy",
        {"theme_mode": "dark", "enable_notifications": "true", "volume": "55"},
    )

    assert ok is True
    table_name, payload, on_conflict = fake.upsert_calls[0]
    assert table_name == "focusguardian_profile_settings"
    assert on_conflict == "profile_name"
    assert payload["profile_name"] == "thanhquy"
    assert payload["settings"]["theme_mode"] == "dark"
    assert payload["settings"]["enable_notifications"] is True
    assert payload["settings"]["volume"] == 55


def test_load_profile_settings_reads_jsonb_settings_column():
    sync, fake = _sync_with_fake_client()
    fake.rows = [
        {
            "profile_name": "thanhquy",
            "settings": {"theme_mode": "dark", "enable_sounds": "false"},
        }
    ]

    loaded = sync.load_profile_settings("thanhquy")

    assert loaded == {"theme_mode": "dark", "enable_sounds": False}


def test_missing_supabase_config_message_points_to_api_key_field():
    config = SupabaseConfig(
        enabled=True,
        url="https://jprbgccjztseypgzcmea.supabase.co",
        api_key="",
    )

    assert "supabase_publishable_key" in supabase_missing_config_message(config)
