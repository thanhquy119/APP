from __future__ import annotations

from app.logic.auth import UserAccount, hash_password, now_ts, timestamp_to_iso
from app.logic.auth_manager import AuthManager


class UnavailableStore:
    availability_error = "Supabase offline"

    def configure_from_app_config(self, config) -> None:
        self.config = dict(config or {})

    def ensure_available(self) -> bool:
        return False


class MemoryStore:
    availability_error = ""

    def __init__(self, users=None):
        self.users = list(users or [])
        self.last_login_updates = []

    def configure_from_app_config(self, config) -> None:
        self.config = dict(config or {})

    def ensure_available(self) -> bool:
        return True

    def find_by_identity(self, identity: str):
        key = str(identity or "").strip().lower()
        for user in self.users:
            if user.username.lower() == key:
                return user
        return None

    def find_by_username(self, username: str):
        return self.find_by_identity(username)

    def find_by_user_id(self, user_id: str):
        key = str(user_id or "").strip().lower()
        for user in self.users:
            if user.user_id.lower() == key:
                return user
        return None

    def create_user(self, user: UserAccount):
        if self.find_by_username(user.username) is not None:
            return False, "Username da ton tai"
        self.users.append(user)
        return True, "Dang ky thanh cong"

    def update_last_login(self, user_id: str, timestamp=None) -> bool:
        self.last_login_updates.append((user_id, timestamp))
        return True

    def create_user_account(self, *, username: str, password_hash: str, profile_name: str) -> UserAccount:
        ts = now_ts()
        ts_iso = timestamp_to_iso(ts)
        return UserAccount(
            user_id=f"id_{username}",
            username=username,
            password_hash=password_hash,
            created_at=ts,
            created_at_iso=ts_iso,
            last_login_at=ts,
            last_login_at_iso=ts_iso,
            is_active=True,
            profile_name=profile_name,
        )


def _user(username: str, password: str = "12345678") -> UserAccount:
    ts = now_ts()
    ts_iso = timestamp_to_iso(ts)
    return UserAccount(
        user_id=f"id_{username}",
        username=username,
        password_hash=hash_password(password),
        created_at=ts,
        created_at_iso=ts_iso,
        last_login_at=ts,
        last_login_at_iso=ts_iso,
        is_active=True,
        profile_name=username,
    )


def test_login_fails_when_supabase_is_unavailable():
    manager = AuthManager({}, store=UnavailableStore())

    result = manager.login(username="offline_user", password="12345678")

    assert result.success is False
    assert result.session is None
    assert manager.is_authenticated() is False
    assert result.message == "Supabase offline"


def test_register_fails_when_supabase_is_unavailable():
    manager = AuthManager({}, store=UnavailableStore())

    result = manager.register(
        username="new_user",
        password="12345678",
        confirm_password="12345678",
    )

    assert result.success is False
    assert result.session is None
    assert result.message == "Supabase offline"


def test_login_requires_existing_database_user_and_matching_password():
    store = MemoryStore([_user("thanhquy", "correct-password")])
    manager = AuthManager({}, store=store)

    wrong = manager.login(username="thanhquy", password="wrong-password")
    right = manager.login(username="thanhquy", password="correct-password")

    assert wrong.success is False
    assert wrong.session is None
    assert right.success is True
    assert right.session is not None
    assert right.session.user.username == "thanhquy"
    assert store.last_login_updates


def test_restore_cached_session_must_verify_database_user():
    manager = AuthManager({}, store=UnavailableStore())

    result = manager.restore_cached_session(
        user_id="id_thanhquy",
        username="thanhquy",
        verify_backend=False,
    )

    assert result.success is False
    assert result.session is None

    store = MemoryStore([_user("thanhquy")])
    manager = AuthManager({}, store=store)
    result = manager.restore_cached_session(
        user_id="id_thanhquy",
        username="thanhquy",
        verify_backend=True,
    )

    assert result.success is True
    assert result.session is not None
    assert result.session.user.username == "thanhquy"
