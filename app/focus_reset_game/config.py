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
    symbols: tuple[str, ...] = ("1", "2", "3", "4", "5")


@dataclass
class VisualSearchConfig:
    rounds: int = 8
    grid_start: int = 4
    grid_max: int = 6
    round_timeout_s: int = 12


@dataclass
class FocusResetConfig:
    app_name: str = "Attention Probe"
    subtitle: str = "A short post-break attention probe with micro-breaks"

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
                cfg.sequence.symbols = tuple(str(x)[:2] for x in symbols[:5])
                if tuple(symbol.upper() for symbol in cfg.sequence.symbols[:4]) == ("A", "S", "D", "F"):
                    cfg.sequence.symbols = ("1", "2", "3", "4", "5")
                elif cfg.sequence.symbols == ("1", "2", "3", "4"):
                    cfg.sequence.symbols = ("1", "2", "3", "4", "5")

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
    background: str = "#0b131d"
    panel: str = "#131f2d"
    panel_alt: str = "#111c2b"
    panel_soft: str = "#111c2b"
    border: str = "#2a394b"
    text_primary: str = "#edf4fd"
    text_muted: str = "#9baec5"
    accent: str = "#59d5c0"
    accent_hover: str = "#4abdaa"
    accent_text: str = "#07251f"
    accent_border: str = "#71e3d0"
    hero_bg: str = "#111c2b"
    progress_bg: str = "#162334"
    table_bg: str = "#101a27"
    table_header_bg: str = "#182637"
    table_grid: str = "#2a394b"
    selection_bg: str = "#1d4d4a"
    success_text: str = "#7ef4d4"
    error_text: str = "#f7b3b3"
    info_text: str = "#9fd6ff"
    interactive_bg: str = "#223449"
    interactive_border: str = "#2f465e"
    interactive_hover: str = "#293f58"
    target_color: str = "#59d5c0"
    nogo_color: str = "#ef9d95"
    titlebar_dot_close: str = "#ff5f57"
    titlebar_dot_close_hover: str = "#ff736d"
    titlebar_dot_close_pressed: str = "#e14f49"
    titlebar_dot_min: str = "#febc2e"
    titlebar_dot_min_hover: str = "#ffca4c"
    titlebar_dot_min_pressed: str = "#dea225"
    titlebar_dot_max: str = "#28c840"
    titlebar_dot_max_hover: str = "#42d95a"
    titlebar_dot_max_pressed: str = "#1faa36"

    @classmethod
    def for_mode(cls, mode: str) -> "Theme":
        normalized = str(mode or "dark").strip().lower()
        if normalized != "light":
            return cls(mode="dark")

        return cls(
            mode="light",
            background="#f2f8fe",
            panel="#ffffff",
            panel_alt="#f7fbff",
            panel_soft="#eef6fd",
            border="#c5d6e8",
            text_primary="#182c41",
            text_muted="#435d76",
            accent="#2f9f90",
            accent_hover="#268f82",
            accent_text="#ffffff",
            accent_border="#2f9f90",
            hero_bg="#f2f8fe",
            progress_bg="#dfeaf6",
            table_bg="#ffffff",
            table_header_bg="#eaf3fb",
            table_grid="#c5d6e8",
            selection_bg="#d7eee9",
            success_text="#16775f",
            error_text="#b74747",
            info_text="#286f9c",
            interactive_bg="#e7f0fa",
            interactive_border="#b8ccdf",
            interactive_hover="#dbe8f5",
            target_color="#2f9f90",
            nogo_color="#c75a52",
            titlebar_dot_close="#ff5f57",
            titlebar_dot_close_hover="#ff736d",
            titlebar_dot_close_pressed="#e14f49",
            titlebar_dot_min="#febc2e",
            titlebar_dot_min_hover="#ffd159",
            titlebar_dot_min_pressed="#dea225",
            titlebar_dot_max="#2fca46",
            titlebar_dot_max_hover="#4ddc63",
            titlebar_dot_max_pressed="#24ab39",
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
