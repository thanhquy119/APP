"""Authentication domain models and secure password helpers."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime

PBKDF2_ALGORITHM = "pbkdf2_sha256"
PBKDF2_DIGEST = "sha256"
PBKDF2_ITERATIONS = 260_000

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9._-]{3,32}$")


@dataclass(slots=True)
class UserAccount:
    """Persistent account record stored in Google Sheets."""

    user_id: str
    username: str
    password_hash: str
    created_at: int
    created_at_iso: str
    last_login_at: int
    last_login_at_iso: str
    is_active: bool
    profile_name: str

    @classmethod
    def from_record(cls, record: dict[str, object]) -> "UserAccount":
        created_at = _safe_int(record.get("created_at"), default=0)
        last_login_at = _safe_int(record.get("last_login_at"), default=0)
        return cls(
            user_id=str(record.get("user_id", "") or "").strip(),
            username=str(record.get("username", "") or "").strip(),
            password_hash=str(record.get("password_hash", "") or "").strip(),
            created_at=created_at,
            created_at_iso=str(record.get("created_at_iso", "") or "").strip() or timestamp_to_iso(created_at),
            last_login_at=last_login_at,
            last_login_at_iso=(
                str(record.get("last_login_at_iso", "") or "").strip()
                or timestamp_to_iso(last_login_at)
            ),
            is_active=_safe_bool(record.get("is_active"), default=True),
            profile_name=normalize_profile_name(
                str(record.get("profile_name", "") or "")
                or str(record.get("username", "") or "")
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "password_hash": self.password_hash,
            "created_at": int(self.created_at),
            "created_at_iso": self.created_at_iso,
            "last_login_at": int(self.last_login_at),
            "last_login_at_iso": self.last_login_at_iso,
            "is_active": bool(self.is_active),
            "profile_name": self.profile_name,
        }


@dataclass(slots=True)
class CurrentUserSession:
    """In-memory authenticated session."""

    user: UserAccount
    login_at: int
    login_at_iso: str


def now_ts() -> int:
    return int(time.time())


def timestamp_to_iso(value: int) -> str:
    if value <= 0:
        return ""
    return datetime.fromtimestamp(int(value)).isoformat(sep=" ", timespec="seconds")


def normalize_username(username: str) -> str:
    return str(username or "").strip()


def normalize_profile_name(profile_name: str) -> str:
    base = str(profile_name or "").strip().lower()
    if not base:
        return "default"
    cleaned = []
    for ch in base:
        if ch.isalnum() or ch in {"_", "-"}:
            cleaned.append(ch)
        elif ch in {".", " "}:
            cleaned.append("_")
    normalized = "".join(cleaned).strip("_")
    return normalized or "default"


def is_valid_username(username: str) -> bool:
    return bool(_USERNAME_RE.match(normalize_username(username)))


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 (bcrypt-equivalent safety class)."""
    secret = str(password or "")
    if not secret:
        raise ValueError("Password cannot be empty")

    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        PBKDF2_DIGEST,
        secret.encode("utf-8"),
        salt,
        int(iterations),
    )
    return f"{PBKDF2_ALGORITHM}${int(iterations)}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded_hash: str) -> bool:
    """Verify PBKDF2 password hash in constant time."""
    if not password or not encoded_hash:
        return False

    try:
        scheme, iterations_text, salt_hex, digest_hex = encoded_hash.split("$", 3)
        if scheme != PBKDF2_ALGORITHM:
            return False
        iterations = int(iterations_text)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except (ValueError, TypeError):
        return False

    actual = hashlib.pbkdf2_hmac(
        PBKDF2_DIGEST,
        str(password).encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(actual, expected)


def _safe_int(value: object, *, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _safe_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return bool(default)
