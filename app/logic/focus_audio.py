"""Focus audio playback manager for optional concentration background sounds."""

from __future__ import annotations

import logging
import random
import tempfile
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QUrl

try:
    from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
    _AUDIO_BACKEND_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on local Qt multimedia installation
    QAudioOutput = None  # type: ignore[assignment]
    QMediaPlayer = None  # type: ignore[assignment]
    _AUDIO_BACKEND_IMPORT_ERROR = exc


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FocusAudioTrack:
    key: str
    label: str
    filename: str


AUDIO_ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets" / "audio"
DEFAULT_FOCUS_AUDIO_TRACK = "rain_light"
DEFAULT_FOCUS_AUDIO_VOLUME = 30

FOCUS_AUDIO_TRACKS: tuple[FocusAudioTrack, ...] = (
    FocusAudioTrack("rain_light", "Mưa nhẹ", "mixkit-light-rain-loop-2393.wav"),
    FocusAudioTrack("forest", "Rừng", "mixkit-night-crickets-near-the-swamp-1782.wav"),
    FocusAudioTrack("stream", "Suối", "mixkit-birds-chirping-near-the-river-2473.wav"),
    FocusAudioTrack("white_noise", "White noise", "white_noise.wav"),
    FocusAudioTrack("pink_noise", "Pink noise", "pink_noise.wav"),
    FocusAudioTrack("brown_noise", "Brown noise", "brown_noise.wav"),
)

_TRACK_BY_KEY = {track.key: track for track in FOCUS_AUDIO_TRACKS}
_GENERATED_NOISE_KEYS = {"white_noise", "pink_noise", "brown_noise"}


class FocusAudioManager(QObject):
    """Handle background focus audio with looped playback and safe error handling."""

    def __init__(self, config: Optional[dict] = None, parent=None):
        super().__init__(parent)

        self._enabled = False
        self._preview_active = False
        self._track_key = DEFAULT_FOCUS_AUDIO_TRACK
        self._volume = DEFAULT_FOCUS_AUDIO_VOLUME

        self._audio_available = QAudioOutput is not None and QMediaPlayer is not None
        self._audio_output = None
        self._player = None

        if self._audio_available:
            self._audio_output = QAudioOutput(self)
            self._player = QMediaPlayer(self)
            self._player.setAudioOutput(self._audio_output)
            self._player.errorOccurred.connect(self._on_player_error)
            self._player.mediaStatusChanged.connect(self._on_media_status_changed)
        else:
            logger.warning(
                "QtMultimedia is unavailable. Focus audio is disabled (%s)",
                _AUDIO_BACKEND_IMPORT_ERROR,
            )

        if config:
            self.load_from_config(config)
        else:
            self.set_volume(DEFAULT_FOCUS_AUDIO_VOLUME)

    def is_available(self) -> bool:
        return bool(self._audio_available)

    def is_enabled(self) -> bool:
        return bool(self._enabled)

    def is_previewing(self) -> bool:
        return bool(self._preview_active)

    def current_track_key(self) -> str:
        return self._track_key

    def current_volume(self) -> int:
        return self._volume

    def to_config(self) -> dict:
        return {
            "enable_focus_audio": bool(self._enabled),
            "focus_audio_track": self._track_key,
            "focus_audio_volume": int(self._volume),
        }

    def load_from_config(self, config: Optional[dict]) -> None:
        data = dict(config or {})
        self._track_key = self._normalize_track_key(data.get("focus_audio_track"))
        self.set_volume(data.get("focus_audio_volume", DEFAULT_FOCUS_AUDIO_VOLUME))

        enabled = bool(data.get("enable_focus_audio", False))
        if enabled:
            ok, message = self.play(self._track_key)
            if not ok:
                logger.warning("Could not start focus audio from config: %s", message)
        else:
            self.set_enabled(False)

    def set_enabled(self, enabled: bool) -> tuple[bool, str]:
        self._enabled = bool(enabled)
        if not self.is_available():
            if self._enabled:
                return False, "Âm thanh không khả dụng trong môi trường hiện tại."
            return True, ""

        if self._enabled:
            return self.play(self._track_key)

        self.stop()
        return True, ""

    def set_track(self, track_key: str) -> None:
        self._track_key = self._normalize_track_key(track_key)

    def set_volume(self, value) -> int:
        self._volume = self._normalize_volume(value)
        if self._audio_output is not None:
            self._audio_output.setVolume(self._volume / 100.0)
        return self._volume

    def play(self, track_key: Optional[str] = None) -> tuple[bool, str]:
        if track_key:
            self._track_key = self._normalize_track_key(track_key)
        ok, message = self._start_playback(self._track_key, preview=False)
        self._enabled = bool(ok)
        return ok, message

    def preview(self, track_key: Optional[str] = None) -> tuple[bool, str]:
        if track_key:
            self._track_key = self._normalize_track_key(track_key)
        ok, message = self._start_playback(self._track_key, preview=True)
        if not ok:
            self._preview_active = False
        return ok, message

    def stop(self) -> None:
        self._preview_active = False
        if self._player is None:
            return
        self._player.stop()

    def _start_playback(self, track_key: str, preview: bool) -> tuple[bool, str]:
        if not self.is_available():
            return False, "Âm thanh không khả dụng trong môi trường hiện tại."

        if self._player is None:
            return False, "Media player chưa được khởi tạo."

        audio_path = self._resolve_track_path(track_key)
        if audio_path is None:
            self._preview_active = False
            return (
                False,
                f"Không tìm thấy file audio cho lựa chọn '{track_key}'.",
            )

        try:
            self.set_volume(self._volume)
            self._player.setSource(QUrl.fromLocalFile(str(audio_path.resolve())))
            self._configure_looping()
            self._player.play()
            self._preview_active = bool(preview)

            mode = "preview" if preview else "background"
            logger.info("Playing %s focus audio: %s", mode, audio_path)
            return True, ""
        except Exception as exc:
            logger.exception("Failed to start focus audio playback: %s", exc)
            self._preview_active = False
            return False, f"Không thể phát audio: {exc}"

    def _configure_looping(self) -> None:
        if self._player is None:
            return

        try:
            loops_enum = getattr(QMediaPlayer, "Loops", None)
            if loops_enum is not None and hasattr(loops_enum, "Infinite"):
                self._player.setLoops(int(loops_enum.Infinite))
                return
            self._player.setLoops(-1)
        except Exception:
            # Some backends may not support setLoops reliably; fallback handled in mediaStatus callback.
            return

    def _resolve_track_path(self, track_key: str) -> Optional[Path]:
        track = _TRACK_BY_KEY.get(track_key)
        if track is None:
            track = _TRACK_BY_KEY[DEFAULT_FOCUS_AUDIO_TRACK]
            track_key = track.key

        preferred = AUDIO_ASSETS_DIR / track.filename
        if preferred.exists():
            return preferred

        if track_key in _GENERATED_NOISE_KEYS:
            generated = self._ensure_noise_file(track_key, preferred)
            if generated is not None and generated.exists():
                return generated

        stem = preferred.stem
        for ext in (".wav", ".mp3", ".ogg", ".m4a"):
            candidate = AUDIO_ASSETS_DIR / f"{stem}{ext}"
            if candidate.exists():
                return candidate

        logger.error("Missing focus audio file for track '%s': expected %s", track_key, preferred)
        return None

    def _ensure_noise_file(self, track_key: str, target_path: Path) -> Optional[Path]:
        candidate_paths = [target_path]
        cache_path = Path(tempfile.gettempdir()) / "focusguardian_audio_cache" / target_path.name
        if cache_path not in candidate_paths:
            candidate_paths.append(cache_path)

        for destination in candidate_paths:
            generated = self._try_generate_noise_file(track_key, destination)
            if generated is not None and generated.exists():
                return generated
        logger.warning("Unable to generate noise file for track '%s'", track_key)
        return None

    def _try_generate_noise_file(self, track_key: str, destination: Path) -> Optional[Path]:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)

            sample_rate = 22050
            duration_seconds = 10
            total_samples = sample_rate * duration_seconds
            rng = random.Random(f"focus-audio-{track_key}")
            samples = array("h")

            if track_key == "white_noise":
                for _ in range(total_samples):
                    noise = rng.uniform(-1.0, 1.0) * 0.42
                    samples.append(self._to_pcm16(noise))
            elif track_key == "pink_noise":
                b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0
                for _ in range(total_samples):
                    white = rng.uniform(-1.0, 1.0)
                    b0 = (0.99886 * b0) + (white * 0.0555179)
                    b1 = (0.99332 * b1) + (white * 0.0750759)
                    b2 = (0.96900 * b2) + (white * 0.1538520)
                    b3 = (0.86650 * b3) + (white * 0.3104856)
                    b4 = (0.55000 * b4) + (white * 0.5329522)
                    b5 = (-0.7616 * b5) - (white * 0.0168980)
                    pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + (white * 0.5362)
                    b6 = white * 0.115926
                    samples.append(self._to_pcm16(pink * 0.10))
            elif track_key == "brown_noise":
                last = 0.0
                for _ in range(total_samples):
                    white = rng.uniform(-1.0, 1.0)
                    last = (last + (0.022 * white)) / 1.022
                    samples.append(self._to_pcm16(last * 3.2))
            else:
                return None

            with wave.open(str(destination), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples.tobytes())

            logger.info("Generated focus noise file: %s", destination)
            return destination
        except Exception as exc:
            logger.debug("Failed generating noise file at %s: %s", destination, exc)
            return None

    @staticmethod
    def _to_pcm16(sample: float) -> int:
        clipped = max(-1.0, min(1.0, float(sample)))
        return int(clipped * 32767.0)

    @staticmethod
    def _normalize_track_key(value) -> str:
        key = str(value or "").strip().lower()
        if key in _TRACK_BY_KEY:
            return key
        return DEFAULT_FOCUS_AUDIO_TRACK

    @staticmethod
    def _normalize_volume(value) -> int:
        try:
            numeric = int(float(value))
        except (TypeError, ValueError):
            numeric = DEFAULT_FOCUS_AUDIO_VOLUME
        return max(0, min(100, numeric))

    def _on_media_status_changed(self, status) -> None:
        if self._player is None:
            return

        if status != QMediaPlayer.MediaStatus.EndOfMedia:
            return

        if not (self._enabled or self._preview_active):
            return

        # Fallback loop for backends where setLoops may be ignored.
        try:
            self._player.setPosition(0)
            self._player.play()
        except Exception as exc:
            logger.warning("Failed to restart loop playback: %s", exc)

    def _on_player_error(self, error, error_string: str) -> None:
        _ = error
        logger.error("Focus audio player error: %s", error_string)
