"""Manual helper to test Zalo bot outbound messaging from terminal.

Usage examples:
- python zalo_send_test.py --text "FocusGuardian test message"
- python zalo_send_test.py --token "<bot_token>" --chat-id "<chat_id>" --text "Hello"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from app.logic.zalo_bot import ZaloBotClient, ZaloBotConfig


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a test message via Zalo Bot API")
    parser.add_argument("--token", default="", help="Zalo Bot token (optional, fallback from config.json)")
    parser.add_argument("--chat-id", default="", help="Target chat_id (optional, fallback from config.json)")
    parser.add_argument(
        "--text",
        default="FocusGuardian: Tin nhan test ket noi Zalo tu terminal.",
        help="Message text",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config JSON (default: config.json)",
    )

    args = parser.parse_args()

    app_config = _load_config(Path(args.config))
    if args.token:
        app_config["zalo_bot_token"] = args.token
    if args.chat_id:
        app_config["zalo_chat_id"] = args.chat_id

    app_config["enable_zalo_alerts"] = True

    client = ZaloBotClient(ZaloBotConfig.from_app_config(app_config))
    success, detail, _ = client.send_message(app_config.get("zalo_chat_id"), args.text)

    if success:
        print("[OK]", detail)
        return 0

    print("[ERROR]", detail)
    return 1


if __name__ == "__main__":
    sys.exit(main())
