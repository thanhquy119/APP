"""Configuration and theme models for Focus Reset Game."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GoNoGoConfig:
    rounds: int = 3
    round_duration_s: int = 55
    target_probability: float = 0.75
    stimulus_duration_ms: int = 850
    inter_stimulus_ms: int = 600


@dataclass
class SequenceConfig:
    rounds: int = 6
    start_length: int = 3
    min_length: int = 2
    max_length: int = 6
    show_item_ms: int = 700
    gap_ms: int = 240
    input_timeout_s: int = 12
    symbols: tuple[str, ...] = ("A", "S", "D", "F")


@dataclass
class VisualSearchConfig:
    rounds: int = 8
    grid_start: int = 4
    grid_max: int = 6
    round_timeout_s: int = 12


@dataclass
class FocusResetConfig:
    app_name: str = "Focus Reset"
    subtitle: str = "A short attention reset game with micro-breaks"

    baseline_duration_s: int = 24
    micro_break_s: int = 12
    final_breathing_break_s: int = 40

    inhale_seconds: float = 4.0
    exhale_seconds: float = 6.0

    response_key_name: str = "Space"
    sound_enabled: bool = False

    gonogo: GoNoGoConfig = field(default_factory=GoNoGoConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    visual: VisualSearchConfig = field(default_factory=VisualSearchConfig)

    history_path: Path = field(default_factory=lambda: Path("analytics") / "focus_reset_history.json")
    settings_path: Path = field(default_factory=lambda: Path("analytics") / "focus_reset_settings.json")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["history_path"] = str(self.history_path)
        data["settings_path"] = str(self.settings_path)
        data["sequence"]["symbols"] = list(self.sequence.symbols)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FocusResetConfig":
        cfg = cls()
        if not isinstance(data, dict):
            return cfg

        cfg.app_name = str(data.get("app_name", cfg.app_name))
        cfg.subtitle = str(data.get("subtitle", cfg.subtitle))

        cfg.baseline_duration_s = _clamp_int(data.get("baseline_duration_s", cfg.baseline_duration_s), 10, 60)
        cfg.micro_break_s = _clamp_int(data.get("micro_break_s", cfg.micro_break_s), 5, 30)
        cfg.final_breathing_break_s = _clamp_int(
            data.get("final_breathing_break_s", cfg.final_breathing_break_s),
            20,
            90,
        )

        cfg.inhale_seconds = _clamp_float(data.get("inhale_seconds", cfg.inhale_seconds), 2.0, 8.0)
        cfg.exhale_seconds = _clamp_float(data.get("exhale_seconds", cfg.exhale_seconds), 3.0, 10.0)

        cfg.response_key_name = str(data.get("response_key_name", cfg.response_key_name))
        cfg.sound_enabled = bool(data.get("sound_enabled", cfg.sound_enabled))

        gonogo = data.get("gonogo", {})
        if isinstance(gonogo, dict):
            cfg.gonogo.rounds = _clamp_int(gonogo.get("rounds", cfg.gonogo.rounds), 1, 8)
            cfg.gonogo.round_duration_s = _clamp_int(
                gonogo.get("round_duration_s", cfg.gonogo.round_duration_s),
                30,
                120,
            )
            cfg.gonogo.target_probability = _clamp_float(
                gonogo.get("target_probability", cfg.gonogo.target_probability),
                0.60,
                0.85,
            )
            cfg.gonogo.stimulus_duration_ms = _clamp_int(
                gonogo.get("stimulus_duration_ms", cfg.gonogo.stimulus_duration_ms),
                500,
                1200,
            )
            cfg.gonogo.inter_stimulus_ms = _clamp_int(
                gonogo.get("inter_stimulus_ms", cfg.gonogo.inter_stimulus_ms),
                300,
                1000,
            )

        sequence = data.get("sequence", {})
        if isinstance(sequence, dict):
            cfg.sequence.rounds = _clamp_int(sequence.get("rounds", cfg.sequence.rounds), 3, 12)
            cfg.sequence.start_length = _clamp_int(sequence.get("start_length", cfg.sequence.start_length), 2, 5)
            cfg.sequence.min_length = _clamp_int(sequence.get("min_length", cfg.sequence.min_length), 2, 4)
            cfg.sequence.max_length = _clamp_int(sequence.get("max_length", cfg.sequence.max_length), 4, 8)
            cfg.sequence.show_item_ms = _clamp_int(sequence.get("show_item_ms", cfg.sequence.show_item_ms), 350, 1200)
            cfg.sequence.gap_ms = _clamp_int(sequence.get("gap_ms", cfg.sequence.gap_ms), 120, 600)
            cfg.sequence.input_timeout_s = _clamp_int(sequence.get("input_timeout_s", cfg.sequence.input_timeout_s), 6, 25)

            symbols = sequence.get("symbols", list(cfg.sequence.symbols))
            if isinstance(symbols, list) and len(symbols) >= 4:
                cfg.sequence.symbols = tuple(str(x)[:2] for x in symbols[:4])

            if cfg.sequence.max_length < cfg.sequence.start_length:
                cfg.sequence.max_length = cfg.sequence.start_length
            if cfg.sequence.start_length < cfg.sequence.min_length:
                cfg.sequence.start_length = cfg.sequence.min_length

        visual = data.get("visual", {})
        if isinstance(visual, dict):
            cfg.visual.rounds = _clamp_int(visual.get("rounds", cfg.visual.rounds), 4, 15)
            cfg.visual.grid_start = _clamp_int(visual.get("grid_start", cfg.visual.grid_start), 3, 6)
            cfg.visual.grid_max = _clamp_int(visual.get("grid_max", cfg.visual.grid_max), 4, 8)
            cfg.visual.round_timeout_s = _clamp_int(
                visual.get("round_timeout_s", cfg.visual.round_timeout_s),
                6,
                25,
            )

            if cfg.visual.grid_max < cfg.visual.grid_start:
                cfg.visual.grid_max = cfg.visual.grid_start

        history_path = data.get("history_path")
        settings_path = data.get("settings_path")

        if isinstance(history_path, str) and history_path.strip():
            cfg.history_path = Path(history_path)
        if isinstance(settings_path, str) and settings_path.strip():
            cfg.settings_path = Path(settings_path)

        return cfg


@dataclass(frozen=True)
class Theme:
    mode: str = "dark"
    background: str = "#050b18"
    panel: str = "#0f172a"
    panel_alt: str = "#111c2f"
    panel_soft: str = "#0a1426"
    border: str = "#233347"
    text_primary: str = "#e2e8f0"
    text_muted: str = "#94a3b8"
    accent: str = "#38bdf8"
    accent_hover: str = "#0ea5e9"
    accent_text: str = "#082032"
    accent_border: str = "#8ddcff"
    hero_bg: str = "#0b1323"
    progress_bg: str = "#08101f"
    table_bg: str = "#0a1426"
    table_header_bg: str = "#111f35"
    table_grid: str = "#1f334d"
    selection_bg: str = "#123354"
    success_text: str = "#86efac"
    error_text: str = "#fca5a5"
    info_text: str = "#93c5fd"
    interactive_bg: str = "#101e32"
    interactive_border: str = "#304860"
    interactive_hover: str = "#17304f"
    target_color: str = "#22c55e"
    nogo_color: str = "#ef4444"

    @classmethod
    def for_mode(cls, mode: str) -> "Theme":
        normalized = str(mode or "dark").strip().lower()
        if normalized != "light":
            return cls(mode="dark")

        return cls(
            mode="light",
            background="#eff4fb",
            panel="#ffffff",
            panel_alt="#edf3fb",
            panel_soft="#f6f9ff",
            border="#c9d7e6",
            text_primary="#263648",
            text_muted="#607488",
            accent="#3ea99a",
            accent_hover="#319688",
            accent_text="#ffffff",
            accent_border="#2f9687",
            hero_bg="#f4f8ff",
            progress_bg="#e3edf8",
            table_bg="#f9fbff",
            table_header_bg="#e8f0fa",
            table_grid="#c7d6e6",
            selection_bg="#d4ebe6",
            success_text="#1f8b5d",
            error_text="#bc4b4b",
            info_text="#2f6fa8",
            interactive_bg="#e8eff9",
            interactive_border="#b8cada",
            interactive_hover="#dde8f5",
            target_color="#1b9f62",
            nogo_color="#c75a52",
        )


def load_focus_reset_config(path: Path | None = None) -> FocusResetConfig:
    """Load user config from JSON with safe fallback defaults."""
    cfg = FocusResetConfig()
    config_path = path or cfg.settings_path

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            cfg = FocusResetConfig.from_dict(raw)
        except Exception:
            cfg = FocusResetConfig()

    cfg.settings_path = config_path
    cfg.history_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.settings_path.parent.mkdir(parents=True, exist_ok=True)
    return cfg


def save_focus_reset_config(cfg: FocusResetConfig, path: Path | None = None) -> None:
    """Persist user config to JSON."""
    target = path or cfg.settings_path
    target.parent.mkdir(parents=True, exist_ok=True)

    with open(target, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)


def _clamp_int(value: Any, lo: int, hi: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = lo
    return max(lo, min(hi, numeric))


def _clamp_float(value: Any, lo: float, hi: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = lo
    return max(lo, min(hi, numeric))
