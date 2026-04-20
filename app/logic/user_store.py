"""Google Sheets-backed user account storage."""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from .auth import UserAccount, now_ts, timestamp_to_iso

logger = logging.getLogger(__name__)


class GoogleSheetsUserStore:
    """Store user accounts in a dedicated worksheet inside the existing spreadsheet."""

    USERS_HEADER = [
        "user_id",
        "username",
        "password_hash",
        "created_at",
        "created_at_iso",
        "last_login_at",
        "last_login_at_iso",
        "is_active",
        "profile_name",
    ]

    _LEGACY_OPTIONAL_COLUMNS = ["email", "display_name"]

    def __init__(self, app_config: Optional[dict[str, Any]] = None):
        self.enabled = False
        self.spreadsheet_id = ""
        self.users_worksheet_name = "focusguardian_users"
        self.service_account_path = "google_service_account.json"

        self._spreadsheet = None
        self._worksheet = None
        self._availability_error = ""
        self._header_fields: list[str] = []

        if app_config is not None:
            self.configure_from_app_config(app_config)

    def configure_from_app_config(self, app_config: dict[str, Any]) -> None:
        enabled = bool(app_config.get("enable_google_sheets_sync", False))
        spreadsheet_id = str(app_config.get("google_sheets_id", "") or "").strip()
        users_ws = str(app_config.get("google_sheets_users_worksheet", "focusguardian_users") or "").strip()
        service_account = str(app_config.get("google_service_account_path", "google_service_account.json") or "").strip()

        changed = (
            enabled != self.enabled
            or spreadsheet_id != self.spreadsheet_id
            or users_ws != self.users_worksheet_name
            or service_account != self.service_account_path
        )

        self.enabled = enabled
        self.spreadsheet_id = spreadsheet_id
        self.users_worksheet_name = users_ws or "focusguardian_users"
        self.service_account_path = service_account or "google_service_account.json"

        if changed:
            self._spreadsheet = None
            self._worksheet = None
            self._availability_error = ""
            self._header_fields = []

    @property
    def availability_error(self) -> str:
        return self._availability_error

    def ensure_available(self) -> bool:
        worksheet = self._get_or_create_users_worksheet()
        return worksheet is not None

    def find_by_identity(self, identity: str) -> Optional[UserAccount]:
        # Keep method name for compatibility with existing callers.
        key = str(identity or "").strip().lower()
        if not key:
            return None

        for user in self._load_all_users():
            if user.username.lower() == key:
                return user
        return None

    def find_by_username(self, username: str) -> Optional[UserAccount]:
        return self.find_by_identity(username)

    def find_by_user_id(self, user_id: str) -> Optional[UserAccount]:
        key = str(user_id or "").strip().lower()
        if not key:
            return None

        for user in self._load_all_users():
            if str(user.user_id or "").strip().lower() == key:
                return user
        return None

    def create_user(self, user: UserAccount) -> tuple[bool, str]:
        worksheet = self._get_or_create_users_worksheet()
        if worksheet is None:
            return False, self._availability_error or "Không thể truy cập worksheet users"

        if self.find_by_username(user.username) is not None:
            return False, "Username đã tồn tại"

        header = self._get_header_fields(worksheet)
        record = user.to_record()
        # Legacy compatibility for worksheets that still include these columns.
        record.setdefault("email", "")
        record.setdefault("display_name", user.username)

        row = [record.get(column, "") for column in header]

        try:
            worksheet.append_row(row, value_input_option="USER_ENTERED")
            return True, "Đăng ký thành công"
        except Exception as exc:
            logger.warning("Failed to append user row: %s", exc)
            self._availability_error = "Không thể ghi dữ liệu user lên Google Sheets"
            return False, self._availability_error

    def update_last_login(self, user_id: str, timestamp: Optional[int] = None) -> bool:
        worksheet = self._get_or_create_users_worksheet()
        if worksheet is None:
            return False

        row_index = self._find_row_index_by_user_id(worksheet, user_id)
        if row_index is None:
            return False

        ts = int(timestamp or now_ts())
        ts_iso = timestamp_to_iso(ts)

        col_last_login = self._column_index("last_login_at")
        col_last_login_iso = self._column_index("last_login_at_iso")
        if col_last_login is None or col_last_login_iso is None:
            return False

        try:
            worksheet.update_cell(row_index, col_last_login, ts)
            worksheet.update_cell(row_index, col_last_login_iso, ts_iso)
            return True
        except Exception as exc:
            logger.warning("Failed to update last_login_at for user '%s': %s", user_id, exc)
            return False

    def create_user_account(
        self,
        *,
        username: str,
        password_hash: str,
        profile_name: str,
    ) -> UserAccount:
        created_at = now_ts()
        created_at_iso = timestamp_to_iso(created_at)
        user_id = uuid.uuid4().hex
        return UserAccount(
            user_id=user_id,
            username=username,
            password_hash=password_hash,
            created_at=created_at,
            created_at_iso=created_at_iso,
            last_login_at=created_at,
            last_login_at_iso=created_at_iso,
            is_active=True,
            profile_name=profile_name,
        )

    def _load_all_users(self) -> list[UserAccount]:
        worksheet = self._get_or_create_users_worksheet()
        if worksheet is None:
            return []

        try:
            rows = worksheet.get_all_records()
        except Exception as exc:
            logger.warning("Failed to read users worksheet: %s", exc)
            self._availability_error = "Không đọc được dữ liệu users từ Google Sheets"
            return []

        users: list[UserAccount] = []
        for row in rows:
            try:
                users.append(UserAccount.from_record(row))
            except Exception as exc:
                logger.debug("Skip malformed user row: %s", exc)
        return users

    def _get_or_create_users_worksheet(self):
        if not self.enabled:
            self._availability_error = "Google Sheets sync đang tắt"
            return None
        if not self.spreadsheet_id:
            self._availability_error = "Thiếu Google Sheets ID"
            return None

        if self._worksheet is not None:
            return self._worksheet

        spreadsheet = self._open_spreadsheet()
        if spreadsheet is None:
            return None

        try:
            worksheet = spreadsheet.worksheet(self.users_worksheet_name)
        except Exception:
            try:
                worksheet = spreadsheet.add_worksheet(
                    title=self.users_worksheet_name,
                    rows="3000",
                    cols=str(len(self.USERS_HEADER) + 4),
                )
            except Exception as exc:
                logger.warning("Failed to create users worksheet '%s': %s", self.users_worksheet_name, exc)
                self._availability_error = "Không tạo được worksheet users trên Google Sheets"
                return None

        self._ensure_header(worksheet)
        self._worksheet = worksheet
        return worksheet

    def _open_spreadsheet(self):
        if self._spreadsheet is not None:
            return self._spreadsheet

        try:
            import gspread
            from google.oauth2.service_account import Credentials
        except Exception as exc:
            logger.warning("Google Sheets auth unavailable (dependency): %s", exc)
            self._availability_error = "Thiếu thư viện gspread/google-auth"
            return None

        credentials_path = self._resolve_credentials_path()
        if not credentials_path.exists():
            self._availability_error = f"Không tìm thấy file service account: {credentials_path}"
            return None

        try:
            scopes = ["https://www.googleapis.com/auth/spreadsheets"]
            creds = Credentials.from_service_account_file(str(credentials_path), scopes=scopes)
            client = gspread.authorize(creds)
            self._spreadsheet = client.open_by_key(self.spreadsheet_id)
            return self._spreadsheet
        except Exception as exc:
            logger.warning("Failed to initialize users spreadsheet: %s", exc)
            self._availability_error = "Không kết nối được Google Sheets"
            return None

    def _resolve_credentials_path(self) -> Path:
        raw_path = Path(self.service_account_path)
        if raw_path.is_absolute():
            return raw_path

        candidates: list[Path] = []
        seen: set[Path] = set()

        def add_candidate(path: Path) -> None:
            key = path
            try:
                key = path.resolve()
            except Exception:
                pass
            if key in seen:
                return
            seen.add(key)
            candidates.append(path)

        if getattr(sys, "frozen", False):
            exe_dir = Path(sys.executable).resolve().parent
            add_candidate(exe_dir / raw_path)
            add_candidate(exe_dir / "_internal" / raw_path)

            meipass = getattr(sys, "_MEIPASS", "")
            if meipass:
                add_candidate(Path(meipass) / raw_path)

        add_candidate(Path.cwd() / raw_path)
        add_candidate(Path(__file__).resolve().parents[2] / raw_path)

        for path in candidates:
            if path.exists():
                return path

        return candidates[0] if candidates else raw_path

    def _ensure_header(self, worksheet) -> None:
        try:
            first_row = worksheet.row_values(1)
        except Exception:
            first_row = []

        normalized: list[str] = []
        seen: set[str] = set()
        for raw in first_row:
            field = str(raw or "").strip()
            if not field or field in seen:
                continue
            normalized.append(field)
            seen.add(field)

        if not normalized:
            final_header = list(self.USERS_HEADER)
        else:
            final_header = list(normalized)
            for field in self.USERS_HEADER:
                if field not in seen:
                    final_header.append(field)
                    seen.add(field)

            for legacy in self._LEGACY_OPTIONAL_COLUMNS:
                if legacy in seen and legacy not in final_header:
                    final_header.append(legacy)

        if normalized != final_header:
            worksheet.update("A1", [final_header])

        self._header_fields = final_header

    def _get_header_fields(self, worksheet) -> list[str]:
        if self._header_fields:
            return self._header_fields

        self._ensure_header(worksheet)
        return list(self._header_fields)

    def _column_index(self, field_name: str) -> Optional[int]:
        try:
            index = self._header_fields.index(str(field_name))
            return index + 1
        except ValueError:
            return None

    def _find_row_index_by_user_id(self, worksheet, user_id: str) -> Optional[int]:
        key = str(user_id or "").strip().lower()
        if not key:
            return None

        col_user_id = self._column_index("user_id") or 1

        try:
            col = worksheet.col_values(col_user_id)
        except Exception:
            return None

        for idx, value in enumerate(col[1:], start=2):
            if str(value or "").strip().lower() == key:
                return idx
        return None
