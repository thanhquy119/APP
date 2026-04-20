"""
Focus Engine - Core state machine for focus tracking.

Implements state classification using temporal windows and hysteresis
to distinguish between different focus states:
- ON_SCREEN_READING: Looking at screen, engaged
- OFFSCREEN_WRITING: Head down but writing/note-taking (focused)
- PHONE_DISTRACTION: Using phone (distracted)
- DROWSY_FATIGUE: Signs of tiredness
- AWAY: Not at desk / no face detected
- UNCERTAIN: Cannot determine state confidently
"""

import copy
import math
import time
import logging
from enum import Enum, auto
from typing import Optional, NamedTuple, Dict, Any, List
from dataclasses import dataclass, field
from collections import deque

from ..utils.ring_buffer import RingBuffer, MultiFieldBuffer

logger = logging.getLogger(__name__)


class FocusState(Enum):
    """Possible focus states."""
    ON_SCREEN_READING = auto()   # Looking at screen, engaged
    OFFSCREEN_WRITING = auto()   # Head down but writing (still focused)
    PHONE_DISTRACTION = auto()   # Using phone (distracted)
    DROWSY_FATIGUE = auto()      # Signs of tiredness
    AWAY = auto()                # Not at desk / no face
    UNCERTAIN = auto()           # Cannot determine


class FrameFeatures(NamedTuple):
    """Features extracted from a single frame."""
    timestamp: float

    # Face detection
    face_detected: bool

    # Head pose (degrees)
    head_pitch: Optional[float]  # Negative = looking down
    head_yaw: Optional[float]    # Positive = looking right
    head_roll: Optional[float]

    # Eye metrics
    ear_avg: Optional[float]     # Eye aspect ratio (0-0.4)
    is_eye_closed: bool
    blink_detected: bool

    # Hand metrics
    hand_present: bool
    hand_write_score: float      # 0-1, higher = likely writing
    hand_region: str             # "upper", "middle", "lower"

    # Phone detection
    phone_present: bool

    # System idle
    idle_seconds: float

    # Eye gaze signals from blendshapes (optional)
    eye_look_down: Optional[float] = None  # 0-1, higher = looking down
    eye_look_up: Optional[float] = None    # 0-1, higher = looking up
    eye_closure_level: Optional[float] = None  # 0-1 from blendshapes if available


@dataclass
class WindowStats:
    """Statistics computed over a time window."""
    window_seconds: float
    sample_count: int

    # Face presence
    face_ratio: float            # Ratio of frames with face detected

    # Head pose stats
    head_down_ratio: float       # Ratio of frames with head looking down
    max_continuous_head_down: float  # Longest continuous head-down period (seconds)
    avg_pitch: float
    avg_yaw: float
    num_glances_up: int          # Count of upward glances (pitch transitions)

    # Eye stats
    avg_ear: float
    eye_closure_ratio: float     # Ratio of frames with eyes closed
    blink_count: int
    blink_rate_per_min: float
    avg_eye_closure_level: float
    perclos_ratio: float
    avg_eye_look_down: float
    avg_eye_look_up: float
    eye_down_ratio: float

    # Hand stats
    hand_present_ratio: float
    avg_write_score: float
    hand_lower_ratio: float      # Ratio of time hands in lower region

    # Phone stats
    phone_ratio: float

    # Idle stats
    avg_idle: float
    max_idle: float


@dataclass
class FocusEngineConfig:
    """Configuration for focus engine."""

    # Window sizes for analysis (seconds)
    short_window: float = 10.0
    long_window: float = 30.0

    # Head pose thresholds
    head_down_pitch_threshold: float = -15.0  # Degrees
    deep_head_down_pitch_threshold: float = -34.0  # Strongly lowered head
    deep_head_down_min_duration: float = 2.0       # Must persist before counting as distraction
    deep_head_down_eye_missing_ear_threshold: float = 0.16
    deep_head_down_eye_closure_ratio_min: float = 0.45
    head_away_yaw_threshold: float = 30.0     # Degrees
    on_screen_yaw_grace_degrees: float = 6.0  # Extra tolerance for slightly off-center camera setups
    glance_up_pitch_threshold: float = -5.0   # Pitch must rise above this for glance

    # Eye gaze thresholds from MediaPipe face blendshapes (0-1)
    eye_look_down_threshold: float = 0.35
    eye_look_up_threshold: float = 0.30

    # State thresholds
    away_no_face_seconds: float = 10.0        # No face for this long = AWAY

    # OFFSCREEN_WRITING thresholds
    write_head_down_ratio_min: float = 0.5    # At least 50% head down
    write_score_threshold: float = 0.4        # hand_write_score threshold
    write_glances_min: int = 1                # Minimum glances up in window

    # PHONE_DISTRACTION thresholds
    phone_head_down_continuous_max: float = 15.0  # Continuous head down (seconds)
    phone_glances_max: int = 1                    # Very few glances = phone
    phone_write_score_max: float = 0.25           # Low write score
    phone_evidence_threshold: float = 0.72        # Evidence score to classify as phone
    phone_head_down_ratio_min: float = 0.65       # Head-down ratio support for phone pattern
    phone_eye_down_ratio_min: float = 0.60        # Eye-down ratio support in short window
    phone_eye_down_min_duration: float = 45.0     # Sustained head-down + eye-down to infer phone use
    phone_evidence_window: float = 2.0            # Seconds for sustained evidence averaging
    phone_evidence_avg_threshold: float = 0.68    # Average evidence threshold in window
    phone_evidence_peak_threshold: float = 0.78   # Peak evidence threshold in window

    # Blink-rate interpretation anchors:
    # - Resting spontaneous blink is commonly around 15-20/min (Stern et al., 1984).
    # - Visual display tasks often reduce blink to around 4-10/min (Patel et al., 1991; Tsubota & Nakamori, 1993).
    blink_rate_low_screen_max: float = 10.0       # Low blink while still looking at screen
    blink_rate_high_fatigue_min: float = 22.0     # Elevated blink -> visual fatigue/discomfort
    fatigue_head_down_min_duration: float = 15.0  # Require sustained head-down for fatigue inference

    # DROWSY thresholds
    drowsy_ear_threshold: float = 0.18        # Low EAR indicates closed eyes
    drowsy_closure_ratio: float = 0.3         # 30% eyes closed
    perclos_threshold: float = 0.15           # High perclos indicates prolonged closure
    drowsy_idle_threshold: float = 10.0       # Seconds of keyboard/mouse idle

    # Hysteresis (seconds to maintain state before switching)
    hysteresis_enter: float = 1.2             # Time to enter a new state
    hysteresis_exit: float = 2.4              # Time to exit current state
    focused_state_hold_seconds: float = 2.2   # Hold focused states during short noisy spans
    uncertain_short_soft_seconds: float = 2.0 # Short uncertainty should be treated as measurement noise
    uncertain_behavior_window_seconds: float = 3.5  # Sustained uncertainty before stronger penalty

    # Focus score weights
    score_on_screen_weight: float = 1.0
    score_writing_weight: float = 0.9         # Writing is still focused
    score_uncertain_penalty: float = 0.02     # Very mild penalty per second
    score_distraction_penalty: float = 0.5    # Large penalty per second
    score_drowsy_penalty: float = 0.3
    score_away_penalty: float = 0.2

    # Focus score smoothing (EMA alpha)
    score_smoothing: float = 0.1

    # Focus score target model (0-100)
    score_target_on_screen: float = 100.0
    score_target_writing: float = 95.0
    score_target_uncertain: float = 78.0
    score_target_distraction: float = 22.0
    score_target_drowsy: float = 30.0
    score_target_away: float = 38.0

    # Max movement speed for score updates (points/second)
    score_recover_rate: float = 5.5
    score_drop_rate: float = 4.0

    # Research-aligned scoring dynamics controls
    score_noise_softening_seconds: float = 2.6
    score_confidence_floor_focused: float = 0.58
    score_confidence_floor_uncertain: float = 0.33
    score_recover_rate_focused_stable: float = 4.4
    score_recover_rate_focused_unstable: float = 1.8
    score_drop_rate_distraction_strong: float = 8.2
    score_drop_rate_distraction_soft: float = 3.6
    score_drop_rate_drowsy_strong: float = 6.8
    score_uncertain_soft_penalty: float = 0.24
    time_on_task_drift_start_minutes: float = 22.0
    time_on_task_drift_per_minute: float = 0.08
    break_recovery_boost_window_seconds: float = 45.0

    # Re-focus validation after distraction:
    # score recovery is limited until focus is stable for a short period.
    refocus_validation_seconds: float = 2.5
    refocus_confidence_min: float = 0.72
    refocus_face_ratio_min: float = 0.72
    refocus_recover_rate_locked: float = 0.45
    refocus_recover_ramp_seconds: float = 3.0

    # Score update safety
    max_score_delta_time: float = 0.25            # Clamp dt to avoid jumps on timestamp gaps


@dataclass
class StateTransition:
    """Record of a state transition."""
    timestamp: float
    from_state: FocusState
    to_state: FocusState
    reason: str
    confidence: float


class FocusEngine:
    """
    Core focus tracking engine with state machine.

    Processes frame features and maintains focus state using
    temporal windows and hysteresis to avoid jitter.
    """

    def __init__(self, config: Optional[FocusEngineConfig] = None, profile_name: str = "default"):
        self.config = config or FocusEngineConfig()
        self._base_config = copy.deepcopy(self.config)

        # Personalization metadata
        self._profile_name = (profile_name or "default").strip() or "default"
        self._personalization_weight: float = 0.0
        self._personalization_stage: str = "cold_start"
        self._personalized_thresholds: Dict[str, Any] = {}
        self._personalization_context: Dict[str, Any] = {}

        # Feature buffer (stores ~60 seconds at 30fps)
        self._feature_buffer: RingBuffer[FrameFeatures] = RingBuffer(max_size=1800)

        # Current state
        self._current_state = FocusState.UNCERTAIN
        self._state_start_time = time.time()
        self._state_confidence = 0.0

        # Pending state (for hysteresis)
        self._pending_state: Optional[FocusState] = None
        self._pending_since: float = 0.0

        # Focus score
        self._focus_score = 100.0  # Start at full score
        self._raw_focus_score = 100.0

        # Transition history
        self._transitions: deque[StateTransition] = deque(maxlen=100)

        # Track phone-evidence over time to avoid single-frame false positives
        self._phone_evidence_buffer: RingBuffer[float] = RingBuffer(max_size=600, max_age_seconds=60.0)

        # Glance detection state
        self._last_pitch: Optional[float] = None
        self._was_head_down = False
        self._glance_timestamps: deque[float] = deque(maxlen=50)

        # Continuous head-down tracking
        self._head_down_start: Optional[float] = None
        self._eye_down_start: Optional[float] = None

        # Statistics cache
        self._last_stats_time = 0.0
        self._cached_short_stats: Optional[WindowStats] = None
        self._cached_long_stats: Optional[WindowStats] = None

        # Track last frame timestamp for proper time handling
        self._last_frame_time: float = time.time()

        # Track score update timing for time-based scoring
        self._last_score_update_time: float = self._last_frame_time

        # Session start for reporting
        self._session_start: float = self._last_frame_time

        # Latest classification internals for debugging/UI
        self._last_reason: str = "Initializing"
        self._last_intended_state: FocusState = FocusState.UNCERTAIN
        self._last_intended_confidence: float = 0.0

        # Focused-state hold and uncertainty diagnostics
        self._last_focused_state: Optional[FocusState] = None
        self._last_focused_time: float = self._last_frame_time
        self._focused_hold_until: float = self._last_frame_time
        self._uncertain_reason_type: str = "none"
        self._focused_hold_active: bool = False
        self._uncertain_grace_remaining: float = 0.0
        self._uncertain_clean_candidate: bool = False

        # Time-on-task tracking for fatigue drift and break-aware recovery
        self._continuous_work_seconds: float = 0.0
        self._nonfocused_since: Optional[float] = None
        self._last_significant_break_time: float = self._session_start
        self._last_time_tracking_timestamp: float = self._session_start
        self._break_recovery_boost_until: float = self._session_start

        # Smoothed evidence channels to avoid frame-level score jitter
        self._smoothed_focus_stability: float = 0.78
        self._smoothed_distraction_severity: float = 0.0
        self._smoothed_drowsiness_severity: float = 0.0
        self._smoothed_uncertainty_severity: float = 0.0
        self._smoothed_time_drift: float = 0.0

        # Post-distraction recovery gate:
        # require sustained focused evidence before allowing normal score recovery.
        self._needs_refocus_validation: bool = False
        self._refocus_candidate_since: Optional[float] = None
        self._refocus_validated_since: Optional[float] = None

    @property
    def current_state(self) -> FocusState:
        """Get current focus state."""
        return self._current_state

    @property
    def focus_score(self) -> float:
        """Get smoothed focus score (0-100)."""
        return self._focus_score

    @property
    def raw_focus_score(self) -> float:
        """Get raw (unsmoothed) focus score."""
        return self._raw_focus_score

    @property
    def state_duration(self) -> float:
        """Seconds in current state."""
        return time.time() - self._state_start_time

    @property
    def session_duration(self) -> float:
        """Total session duration in seconds."""
        return time.time() - self._session_start

    @property
    def state_confidence(self) -> float:
        """Current state confidence after hysteresis."""
        return self._state_confidence

    @property
    def last_reason(self) -> str:
        """Last classification reason text."""
        return self._last_reason

    @property
    def profile_name(self) -> str:
        """Current profile identifier used for personalization metadata."""
        return self._profile_name

    def set_profile(self, profile_name: str) -> None:
        """Set active profile for diagnostics and personalization context."""
        profile = (profile_name or "default").strip() or "default"
        self._profile_name = profile

    def capture_base_config(self) -> None:
        """Persist current runtime config as baseline before applying overrides."""
        self._base_config = copy.deepcopy(self.config)

    @staticmethod
    def _coerce_threshold_overrides(personalized_thresholds: Any) -> Dict[str, Any]:
        if personalized_thresholds is None:
            return {}

        if isinstance(personalized_thresholds, dict):
            nested = personalized_thresholds.get("focus_engine_overrides")
            if isinstance(nested, dict):
                return dict(nested)
            return dict(personalized_thresholds)

        if hasattr(personalized_thresholds, "to_focus_engine_overrides"):
            try:
                payload = personalized_thresholds.to_focus_engine_overrides()
                if isinstance(payload, dict):
                    return dict(payload)
            except Exception:
                return {}

        return {}

    def set_personalized_thresholds(
        self,
        personalized_thresholds: Optional[Any] = None,
        profile_name: Optional[str] = None,
        user_baseline: Optional[Dict[str, Any]] = None,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Apply profile-specific threshold overrides on top of captured base config.

        The state machine logic remains unchanged; this only updates config values.
        """
        if profile_name is not None:
            self.set_profile(profile_name)

        overrides = self._coerce_threshold_overrides(personalized_thresholds)

        self.config = copy.deepcopy(self._base_config)
        for key, value in overrides.items():
            if not hasattr(self.config, key):
                continue

            current_value = getattr(self.config, key)
            if isinstance(current_value, bool):
                setattr(self.config, key, bool(value))
            elif isinstance(current_value, int) and not isinstance(current_value, bool):
                try:
                    setattr(self.config, key, int(value))
                except (TypeError, ValueError):
                    continue
            elif isinstance(current_value, float):
                try:
                    setattr(self.config, key, float(value))
                except (TypeError, ValueError):
                    continue
            else:
                setattr(self.config, key, value)

        self._personalized_thresholds = overrides

        if hasattr(personalized_thresholds, "personalization_weight"):
            try:
                self._personalization_weight = float(personalized_thresholds.personalization_weight)
            except (TypeError, ValueError):
                self._personalization_weight = 0.0

        if isinstance(personalized_thresholds, dict):
            weight = personalized_thresholds.get("personalization_weight")
            stage = personalized_thresholds.get("adaptation_stage")
            try:
                if weight is not None:
                    self._personalization_weight = float(weight)
            except (TypeError, ValueError):
                pass
            if stage is not None:
                stage_text = str(stage).strip()
                if stage_text:
                    self._personalization_stage = stage_text

        if hasattr(personalized_thresholds, "adaptation_stage"):
            stage = str(getattr(personalized_thresholds, "adaptation_stage", "") or "").strip()
            self._personalization_stage = stage or self._personalization_stage

        if isinstance(user_baseline, dict):
            weight = user_baseline.get("personalization_weight")
            try:
                if weight is not None:
                    self._personalization_weight = float(weight)
            except (TypeError, ValueError):
                pass

        if isinstance(session_context, dict):
            self._personalization_context = dict(session_context)

    def clear_personalization(self) -> None:
        """Reset runtime config to captured global defaults."""
        self.config = copy.deepcopy(self._base_config)
        self._personalized_thresholds = {}
        self._personalization_weight = 0.0
        self._personalization_stage = "cold_start"
        self._personalization_context = {}

    def process_frame(self, features: FrameFeatures) -> FocusState:
        """
        Process a new frame and update state.

        Args:
            features: Extracted features from frame

        Returns:
            Current focus state (may not change every frame)
        """
        # Update last frame time
        self._last_frame_time = features.timestamp

        # Normalize pitch to avoid 180-degree wrap artifacts from pose estimation.
        normalized_pitch = self._normalize_pitch_angle(features.head_pitch)
        if normalized_pitch is not None and features.head_pitch is not None:
            if abs(normalized_pitch - features.head_pitch) > 1e-6:
                features = features._replace(head_pitch=normalized_pitch)

        # Add to buffer
        self._feature_buffer.push(features, features.timestamp)

        # Update glance detection
        self._update_glance_detection(features)

        # Update continuous head-down tracking
        self._update_head_down_tracking(features)

        # Track sustained vertical eye-down pattern
        self._update_eye_down_tracking(features)

        # Compute window statistics using frame timestamp
        short_stats = self._compute_stats(self.config.short_window, features.timestamp)
        long_stats = self._compute_stats(self.config.long_window, features.timestamp)

        # Determine intended state
        intended_state, confidence, reason = self._classify_state(
            features, short_stats, long_stats
        )
        self._refresh_focused_hold(intended_state, confidence, features.timestamp)
        self._last_intended_state = intended_state
        self._last_intended_confidence = confidence
        self._last_reason = reason

        # Apply hysteresis using frame timestamp
        new_state = self._apply_hysteresis(intended_state, confidence, features.timestamp)

        # Update continuous-work timeline before score update so drift uses latest context.
        self._update_time_tracking(new_state, features.timestamp)

        # Update focus score
        self._update_focus_score(
            new_state,
            features.timestamp,
            features,
            short_stats,
            long_stats,
            self._state_confidence,
        )

        return self._current_state

    @staticmethod
    def _normalize_pitch_angle(pitch: Optional[float]) -> Optional[float]:
        """Normalize pitch into a stable range to avoid +/-180 discontinuities."""
        if pitch is None:
            return None

        angle = ((float(pitch) + 180.0) % 360.0) - 180.0
        if angle > 90.0:
            angle = 180.0 - angle
        elif angle < -90.0:
            angle = -180.0 - angle
        return angle

    def _update_glance_detection(self, features: FrameFeatures) -> None:
        """Detect upward glances (looking up from desk)."""
        if features.head_pitch is None:
            return

        pitch = features.head_pitch
        is_head_down = pitch < self.config.head_down_pitch_threshold

        # Detect transition from down to up
        if self._was_head_down and pitch > self.config.glance_up_pitch_threshold:
            self._glance_timestamps.append(features.timestamp)

        self._was_head_down = is_head_down
        self._last_pitch = pitch

    def _update_head_down_tracking(self, features: FrameFeatures) -> None:
        """Track continuous head-down duration."""
        if features.head_pitch is None:
            self._head_down_start = None
            return

        is_down = features.head_pitch < self.config.head_down_pitch_threshold

        if is_down:
            if self._head_down_start is None:
                self._head_down_start = features.timestamp
        else:
            self._head_down_start = None

    def _is_eye_down(self, features: FrameFeatures) -> bool:
        """Return True when eye gaze indicates looking down."""
        if features.eye_look_down is None:
            return False
        return features.eye_look_down >= self.config.eye_look_down_threshold

    def _is_eye_looking_screen(self, features: FrameFeatures) -> bool:
        """Heuristic for still looking toward screen despite a slight head-down posture."""
        look_up = features.eye_look_up or 0.0
        look_down = features.eye_look_down or 0.0
        return look_up >= self.config.eye_look_up_threshold and look_down < self.config.eye_look_down_threshold

    def _update_eye_down_tracking(self, features: FrameFeatures) -> None:
        """Track continuous head-down + eye-down duration."""
        if features.head_pitch is None or features.eye_look_down is None:
            self._eye_down_start = None
            return

        head_down = features.head_pitch < self.config.head_down_pitch_threshold
        eye_down = self._is_eye_down(features)

        if head_down and eye_down:
            if self._eye_down_start is None:
                self._eye_down_start = features.timestamp
        else:
            self._eye_down_start = None

    def _get_continuous_head_down_seconds(self, now: float) -> float:
        """Get current continuous head-down duration."""
        if self._head_down_start is None:
            return 0.0
        return now - self._head_down_start

    def _get_continuous_eye_down_seconds(self, now: float) -> float:
        """Get current continuous head-down + eye-down duration."""
        if self._eye_down_start is None:
            return 0.0
        return now - self._eye_down_start

    def _count_glances_in_window(self, window_seconds: float,
                                  end_time: float) -> int:
        """Count glances up in time window."""
        start_time = end_time - window_seconds
        return sum(1 for t in self._glance_timestamps
                   if start_time <= t <= end_time)

    def _compute_stats(self, window_seconds: float,
                        end_time: Optional[float] = None) -> WindowStats:
        """Compute statistics over time window."""
        now = end_time if end_time is not None else self._last_frame_time

        # Get features in window
        items = self._feature_buffer.get_window(window_seconds, now)

        if not items:
            return self._empty_stats(window_seconds)

        features = [item.data for item in items]
        n = len(features)

        # Face presence
        face_count = sum(1 for f in features if f.face_detected)
        face_ratio = face_count / n

        # Head pose stats
        pitches = [f.head_pitch for f in features if f.head_pitch is not None]
        yaws = [f.head_yaw for f in features if f.head_yaw is not None]
        head_down_threshold = self.config.head_down_pitch_threshold

        if pitches:
            avg_pitch = sum(pitches) / len(pitches)
            head_down_count = sum(1 for p in pitches
                                  if p < head_down_threshold)
            head_down_ratio = head_down_count / len(pitches)
        else:
            avg_pitch = 0.0
            head_down_ratio = 0.0

        avg_yaw = sum(yaws) / len(yaws) if yaws else 0.0

        # Continuous head down calculation
        max_continuous = self._calculate_max_continuous_head_down(features)

        # Glance count
        glances = self._count_glances_in_window(window_seconds, now)

        # Eye stats
        ears = [f.ear_avg for f in features if f.ear_avg is not None]
        avg_ear = sum(ears) / len(ears) if ears else 0.25

        eye_closed_count = sum(1 for f in features if f.is_eye_closed)
        eye_closure_ratio = eye_closed_count / n

        blink_count = sum(1 for f in features if f.blink_detected)
        blink_rate_per_min = blink_count * (60.0 / max(window_seconds, 1e-6))

        closure_levels = [f.eye_closure_level for f in features if f.eye_closure_level is not None]
        if closure_levels:
            avg_eye_closure_level = (
                sum(float(v) for v in closure_levels) / len(closure_levels)
            )
            perclos_ratio = (
                sum(1 for v in closure_levels if float(v) >= 0.8) / len(closure_levels)
            )
        else:
            avg_eye_closure_level = eye_closure_ratio
            perclos_ratio = eye_closure_ratio

        eye_down_values = [f.eye_look_down for f in features if f.eye_look_down is not None]
        eye_up_values = [f.eye_look_up for f in features if f.eye_look_up is not None]
        avg_eye_look_down = (
            sum(float(v) for v in eye_down_values) / len(eye_down_values)
            if eye_down_values else 0.0
        )
        avg_eye_look_up = (
            sum(float(v) for v in eye_up_values) / len(eye_up_values)
            if eye_up_values else 0.0
        )
        eye_down_ratio = (
            sum(1 for v in eye_down_values if float(v) >= self.config.eye_look_down_threshold)
            / len(eye_down_values)
            if eye_down_values else 0.0
        )

        # Hand stats
        hand_count = sum(1 for f in features if f.hand_present)
        hand_present_ratio = hand_count / n

        write_scores = [f.hand_write_score for f in features if f.hand_present]
        avg_write_score = sum(write_scores) / len(write_scores) if write_scores else 0.0

        lower_count = sum(1 for f in features
                          if f.hand_present and f.hand_region == "lower")
        hand_lower_ratio = lower_count / n

        # Phone stats
        phone_count = sum(1 for f in features if f.phone_present)
        phone_ratio = phone_count / n

        # Idle stats
        idles = [f.idle_seconds for f in features]
        avg_idle = sum(idles) / len(idles) if idles else 0.0
        max_idle = max(idles) if idles else 0.0

        return WindowStats(
            window_seconds=window_seconds,
            sample_count=n,
            face_ratio=face_ratio,
            head_down_ratio=head_down_ratio,
            max_continuous_head_down=max_continuous,
            avg_pitch=avg_pitch,
            avg_yaw=avg_yaw,
            num_glances_up=glances,
            avg_ear=avg_ear,
            eye_closure_ratio=eye_closure_ratio,
            blink_count=blink_count,
            blink_rate_per_min=blink_rate_per_min,
            avg_eye_closure_level=avg_eye_closure_level,
            perclos_ratio=perclos_ratio,
            avg_eye_look_down=avg_eye_look_down,
            avg_eye_look_up=avg_eye_look_up,
            eye_down_ratio=eye_down_ratio,
            hand_present_ratio=hand_present_ratio,
            avg_write_score=avg_write_score,
            hand_lower_ratio=hand_lower_ratio,
            phone_ratio=phone_ratio,
            avg_idle=avg_idle,
            max_idle=max_idle
        )

    def _calculate_max_continuous_head_down(self,
                                            features: List[FrameFeatures]) -> float:
        """Calculate maximum continuous head-down duration in feature list."""
        if not features:
            return 0.0

        max_duration = 0.0
        current_start: Optional[float] = None

        for f in features:
            if f.head_pitch is not None:
                is_down = f.head_pitch < self.config.head_down_pitch_threshold

                if is_down:
                    if current_start is None:
                        current_start = f.timestamp
                else:
                    if current_start is not None:
                        duration = f.timestamp - current_start
                        max_duration = max(max_duration, duration)
                        current_start = None

        # Check if still head down at end
        if current_start is not None and features:
            duration = features[-1].timestamp - current_start
            max_duration = max(max_duration, duration)

        return max_duration

    def _empty_stats(self, window_seconds: float) -> WindowStats:
        """Return empty statistics."""
        return WindowStats(
            window_seconds=window_seconds,
            sample_count=0,
            face_ratio=0.0,
            head_down_ratio=0.0,
            max_continuous_head_down=0.0,
            avg_pitch=0.0,
            avg_yaw=0.0,
            num_glances_up=0,
            avg_ear=0.25,
            eye_closure_ratio=0.0,
            blink_count=0,
            blink_rate_per_min=0.0,
            avg_eye_closure_level=0.0,
            perclos_ratio=0.0,
            avg_eye_look_down=0.0,
            avg_eye_look_up=0.0,
            eye_down_ratio=0.0,
            hand_present_ratio=0.0,
            avg_write_score=0.0,
            hand_lower_ratio=0.0,
            phone_ratio=0.0,
            avg_idle=0.0,
            max_idle=0.0
        )

    def _classify_state(self, features: FrameFeatures,
                        short_stats: WindowStats,
                        long_stats: WindowStats) -> tuple[FocusState, float, str]:
        """
        Classify focus state based on current features and window stats.

        Logic chính:
        - Nhìn màn hình (face detected, head thẳng) = ON_SCREEN_READING (tập trung)
        - Head cúi xuống + viết thực sự (write_score cao) = OFFSCREEN_WRITING (tập trung)
        - Head cúi xuống + không viết = UNCERTAIN hoặc PHONE_DISTRACTION
        - Mắt nhắm nhiều = DROWSY_FATIGUE
        - Không có mặt = AWAY

        Returns:
            (state, confidence, reason)
        """
        self._uncertain_reason_type = "none"
        self._focused_hold_active = False
        self._uncertain_grace_remaining = 0.0
        self._uncertain_clean_candidate = False

        now = features.timestamp
        session_age = max(0.0, now - self._session_start)
        startup_relaxed_window = max(12.0, self.config.short_window * 1.8)

        # Priority 1: AWAY - no face detected
        if not features.face_detected:
            no_face_duration = self._get_no_face_duration(now)
            if no_face_duration >= self.config.away_no_face_seconds:
                return (FocusState.AWAY, 0.9,
                        f"No face for {no_face_duration:.1f}s")
            elif short_stats.face_ratio < 0.3:
                return (FocusState.AWAY, 0.7,
                        f"Face rarely detected ({short_stats.face_ratio:.0%})")

        # Priority 2: DROWSY_FATIGUE - kiểm tra buồn ngủ trước
        drowsy_state = self._check_drowsy(features, short_stats, long_stats)
        if drowsy_state:
            return drowsy_state

        # Priority 3: PHONE_DISTRACTION (direct signal only)
        phone_state = self._check_phone_distraction(
            features,
            short_stats,
            long_stats,
            allow_direct=True,
        )
        if phone_state and phone_state[1] >= 0.85:
            return phone_state

        # Priority 4: Head-down disambiguation (writing vs phone)
        head_down_threshold = self.config.head_down_pitch_threshold
        if features.head_pitch is not None and features.head_pitch < head_down_threshold:
            # A mild downward pitch is common due to camera placement.
            # If there is no strong writing/phone evidence, keep this as focused reading.
            mild_down_margin = 7.0
            down_severity = head_down_threshold - features.head_pitch
            if (
                down_severity <= mild_down_margin
                and abs(features.head_yaw or 0.0) < self.config.head_away_yaw_threshold * 0.85
                and not features.phone_present
                and short_stats.avg_write_score < max(0.22, self.config.write_score_threshold - 0.12)
                and short_stats.hand_lower_ratio < 0.35
            ):
                eye_down_signal = (
                    features.eye_look_down is not None
                    and features.eye_look_down >= self.config.eye_look_down_threshold
                )
                if not eye_down_signal:
                    return (
                        FocusState.ON_SCREEN_READING,
                        0.72,
                        (
                            "Mild downward neutral posture "
                            f"(pitch={features.head_pitch:.1f}deg, threshold={head_down_threshold:.1f}deg)"
                        ),
                    )

            # Head down but eyes still oriented toward screen with low blink rate
            # is often a focused reading posture, not phone use.
            if (
                self._is_eye_looking_screen(features)
                and short_stats.blink_rate_per_min <= self.config.blink_rate_low_screen_max
            ):
                return (
                    FocusState.ON_SCREEN_READING,
                    0.76,
                    (
                        "Head down but screen gaze preserved "
                        f"(blink={short_stats.blink_rate_per_min:.1f}/min)"
                    ),
                )

            # Kiểm tra có đang viết thực sự không
            writing_state = self._check_writing(features, short_stats, long_stats)
            if writing_state:
                return writing_state

            # Nếu không phải writing thì kiểm tra phone heuristic
            phone_state = self._check_phone_distraction(
                features,
                short_stats,
                long_stats,
                allow_direct=False,
            )
            if phone_state:
                return phone_state

            fatigue_state = self._check_head_down_fatigue(features, short_stats)
            if fatigue_state:
                return fatigue_state

            # Head cúi + không viết = có thể đang xem điện thoại hoặc mất tập trung
            continuous_down = self._get_continuous_head_down_seconds(now)
            if self._is_deep_head_down_unfocused(features, short_stats, continuous_down):
                return (
                    FocusState.PHONE_DISTRACTION,
                    0.78,
                    f"Deep head-down with eyes not visible ({continuous_down:.0f}s)",
                )

            if self._looks_stably_attentive(features, short_stats, continuous_down):
                conf = 0.68 if session_age <= startup_relaxed_window else 0.62
                return (
                    FocusState.ON_SCREEN_READING,
                    conf,
                    (
                        "Head-down but stable attentive cues "
                        f"(down={continuous_down:.0f}s, face={short_stats.face_ratio:.0%})"
                    ),
                )

            return self._resolve_uncertain_state(
                features,
                short_stats,
                long_stats,
                base_confidence=0.5,
                base_reason=f"Head down {continuous_down:.0f}s, no confirmed phone pattern",
            )

        # Priority 5: ON_SCREEN_READING - only when head is not in head-down zone
        if features.face_detected and features.head_pitch is not None:
            head_yaw = abs(features.head_yaw or 0.0)
            head_pitch = features.head_pitch
            yaw_threshold = self.config.head_away_yaw_threshold
            yaw_grace_limit = yaw_threshold + max(0.0, self.config.on_screen_yaw_grace_degrees)

            if (
                head_yaw < yaw_threshold
                and head_pitch >= head_down_threshold + 2.0
            ):
                return (FocusState.ON_SCREEN_READING, 0.85,
                        f"Looking at screen (yaw={head_yaw:.0f}deg)")

            eye_down_signal = (
                features.eye_look_down is not None
                and features.eye_look_down >= self.config.eye_look_down_threshold
            )
            if (
                head_yaw >= yaw_threshold
                and head_yaw < yaw_grace_limit
                and head_pitch >= head_down_threshold + 2.0
                and not eye_down_signal
                and not features.phone_present
            ):
                yaw_excess = max(0.0, head_yaw - yaw_threshold)
                yaw_grace_span = max(1e-6, yaw_grace_limit - yaw_threshold)
                yaw_excess_ratio = min(1.0, yaw_excess / yaw_grace_span)
                # Slightly over the yaw threshold should keep a confidence close to normal on-screen.
                confidence = 0.84 - 0.10 * yaw_excess_ratio
                return (
                    FocusState.ON_SCREEN_READING,
                    confidence,
                    f"Likely on-screen with slight side angle (yaw={head_yaw:.0f}deg)",
                )

        # Fallback: face is stable but head pose is temporarily missing/noisy.
        if (
            features.face_detected
            and features.head_pitch is None
            and short_stats.face_ratio >= 0.72
            and not features.phone_present
            and short_stats.phone_ratio < 0.22
            and short_stats.eye_closure_ratio < self.config.drowsy_closure_ratio
        ):
            return (
                FocusState.ON_SCREEN_READING,
                0.58,
                "Face stable while head-pose is temporarily unavailable",
            )

        # Default: UNCERTAIN
        return self._resolve_uncertain_state(
            features,
            short_stats,
            long_stats,
            base_confidence=0.22,
            base_reason="Insufficient stable pose cues",
        )

    def _has_strong_negative_uncertainty_evidence(
        self,
        features: FrameFeatures,
        short_stats: WindowStats,
        now: float,
    ) -> bool:
        """Return True when evidence is strong enough to avoid focused hold fallback."""
        yaw_limit = (
            self.config.head_away_yaw_threshold
            + max(0.0, self.config.on_screen_yaw_grace_degrees)
            + 8.0
        )

        if not features.face_detected and self._get_no_face_duration(now) >= self.config.away_no_face_seconds * 0.7:
            return True
        if abs(features.head_yaw or 0.0) > yaw_limit:
            return True
        if features.phone_present or short_stats.phone_ratio >= 0.28:
            return True
        if short_stats.eye_closure_ratio >= self.config.drowsy_closure_ratio + 0.08:
            return True
        if short_stats.perclos_ratio >= self.config.perclos_threshold + 0.08:
            return True

        continuous_down = self._get_continuous_head_down_seconds(now)
        low_write_context = short_stats.avg_write_score < max(0.24, self.config.write_score_threshold * 0.72)
        if (
            features.head_pitch is not None
            and features.head_pitch < (self.config.head_down_pitch_threshold - 4.5)
            and continuous_down >= max(2.5, self.config.deep_head_down_min_duration * 1.2)
            and low_write_context
            and (
                (features.hand_present and features.hand_region == "upper")
                or short_stats.hand_lower_ratio < 0.28
            )
        ):
            return True

        return False

    def _has_partial_writing_evidence(self, features: FrameFeatures, short_stats: WindowStats) -> bool:
        """Return True when writing cues are weak but still present."""
        return (
            (features.hand_present and features.hand_region in ("lower", "middle") and features.hand_write_score >= 0.22)
            or short_stats.hand_present_ratio >= 0.25
            or short_stats.hand_lower_ratio >= 0.25
            or short_stats.avg_write_score >= max(0.24, self.config.write_score_threshold * 0.62)
        )

    def _fallback_focused_state_for_noise(
        self,
        features: FrameFeatures,
        short_stats: WindowStats,
        now: float,
    ) -> Optional[FocusState]:
        """Pick a focused fallback state for short measurement-noise windows."""
        focused_states = (FocusState.ON_SCREEN_READING, FocusState.OFFSCREEN_WRITING)
        grace_remaining = max(0.0, self._focused_hold_until - now)

        candidate: Optional[FocusState] = None
        if self._current_state in focused_states:
            candidate = self._current_state
        elif (
            self._last_focused_state in focused_states
            and (now - self._last_focused_time) <= (self.config.focused_state_hold_seconds + 1.0)
        ):
            candidate = self._last_focused_state

        if candidate == FocusState.OFFSCREEN_WRITING and not self._has_partial_writing_evidence(features, short_stats):
            candidate = FocusState.ON_SCREEN_READING

        mild_pose = (
            features.head_pitch is None
            or features.head_pitch >= (self.config.head_down_pitch_threshold - 2.0)
        )
        if candidate is None and short_stats.face_ratio >= 0.76 and short_stats.phone_ratio < 0.2 and mild_pose:
            candidate = FocusState.ON_SCREEN_READING

        self._focused_hold_active = candidate is not None and grace_remaining > 0.0
        self._uncertain_grace_remaining = grace_remaining
        return candidate

    def _resolve_uncertain_state(
        self,
        features: FrameFeatures,
        short_stats: WindowStats,
        long_stats: WindowStats,
        base_confidence: float,
        base_reason: str,
    ) -> tuple[FocusState, float, str]:
        """
        Distinguish short measurement noise from genuine behavioral uncertainty.

        When face cues remain stable and there is no strong negative evidence,
        keep or fallback to a focused state instead of dropping to UNCERTAIN.
        """
        now = features.timestamp
        _ = long_stats  # Reserved for future weighting refinements.

        strong_negative = self._has_strong_negative_uncertainty_evidence(features, short_stats, now)
        continuous_down = self._get_continuous_head_down_seconds(now)
        low_write_context = short_stats.avg_write_score < max(0.24, self.config.write_score_threshold * 0.72)
        suspicious_head_down = (
            features.head_pitch is not None
            and features.head_pitch < (self.config.head_down_pitch_threshold - 4.0)
            and continuous_down >= max(2.4, self.config.deep_head_down_min_duration * 1.2)
            and low_write_context
            and (
                (features.hand_present and features.hand_region == "upper")
                or short_stats.hand_lower_ratio < 0.28
            )
        )
        measurement_noise = (
            features.face_detected
            and short_stats.face_ratio >= 0.7
            and abs(features.head_yaw or 0.0)
            <= (self.config.head_away_yaw_threshold + max(0.0, self.config.on_screen_yaw_grace_degrees) + 6.0)
            and short_stats.phone_ratio < 0.22
            and short_stats.eye_closure_ratio < (self.config.drowsy_closure_ratio + 0.05)
            and short_stats.perclos_ratio < (self.config.perclos_threshold + 0.05)
            and not suspicious_head_down
            and not strong_negative
        )

        self._uncertain_clean_candidate = measurement_noise

        if measurement_noise:
            self._uncertain_reason_type = "measurement_noise"
            fallback_state = self._fallback_focused_state_for_noise(features, short_stats, now)
            if fallback_state is not None:
                confidence = max(0.52, min(0.76, base_confidence + 0.12))
                return (
                    fallback_state,
                    confidence,
                    f"Measurement noise; holding {fallback_state.name}",
                )

            return (
                FocusState.ON_SCREEN_READING,
                max(0.5, min(0.7, base_confidence + 0.08)),
                "Measurement noise fallback to focused state",
            )

        self._uncertain_reason_type = "behavioral_uncertainty"
        self._focused_hold_active = False
        self._uncertain_grace_remaining = 0.0
        return (FocusState.UNCERTAIN, base_confidence, base_reason)

    def _refresh_focused_hold(self, intended_state: FocusState, confidence: float, timestamp: float) -> None:
        """Refresh focused-state grace window to absorb short measurement noise."""
        focused_states = (FocusState.ON_SCREEN_READING, FocusState.OFFSCREEN_WRITING)
        if intended_state in focused_states and confidence >= 0.55:
            self._last_focused_state = intended_state
            self._last_focused_time = timestamp
            hold_seconds = max(0.6, self.config.focused_state_hold_seconds)
            self._focused_hold_until = max(self._focused_hold_until, timestamp + hold_seconds)

    def _looks_stably_attentive(
        self,
        features: FrameFeatures,
        short_stats: WindowStats,
        continuous_down: float,
    ) -> bool:
        """Return True for stable attentive posture that should not be penalized as uncertain."""
        yaw = abs(features.head_yaw or 0.0)
        yaw_limit = (
            self.config.head_away_yaw_threshold
            + max(0.0, self.config.on_screen_yaw_grace_degrees)
            + 4.0
        )

        if yaw > yaw_limit:
            return False
        if features.phone_present or short_stats.phone_ratio >= 0.22:
            return False
        if short_stats.eye_closure_ratio >= self.config.drowsy_closure_ratio:
            return False

        # Long head-down without writing or gaze evidence should not be treated as attentive.
        prolonged_down = continuous_down >= max(
            self.config.phone_head_down_continuous_max * 0.85,
            12.0,
        )
        low_write_context = short_stats.avg_write_score < max(0.24, self.config.write_score_threshold * 0.75)
        no_glances = short_stats.num_glances_up <= self.config.phone_glances_max

        if prolonged_down and low_write_context and no_glances:
            return False

        if (
            features.eye_look_down is None
            and prolonged_down
            and low_write_context
            and short_stats.hand_lower_ratio < 0.3
        ):
            return False

        if (
            features.hand_present
            and features.hand_region == "upper"
            and low_write_context
            and continuous_down >= max(self.config.deep_head_down_min_duration * 1.5, 3.0)
        ):
            return False

        # Keep writing logic ownership when desk-writing evidence is present.
        if (
            short_stats.avg_write_score >= max(self.config.write_score_threshold, 0.42)
            and short_stats.hand_lower_ratio >= 0.4
        ):
            return False

        # Prolonged eye-down + head-down remains suspicious for phone-like behavior.
        if (
            continuous_down >= max(24.0, self.config.phone_eye_down_min_duration * 0.55)
            and short_stats.eye_down_ratio >= self.config.phone_eye_down_ratio_min
        ):
            return False

        return short_stats.face_ratio >= 0.72

    def _is_deep_head_down_unfocused(
        self,
        features: FrameFeatures,
        short_stats: WindowStats,
        continuous_down: float,
    ) -> bool:
        """True when head is strongly down and eye visibility is lost for a sustained period."""
        if features.head_pitch is None:
            return False

        if features.head_pitch > self.config.deep_head_down_pitch_threshold:
            return False

        if continuous_down < self.config.deep_head_down_min_duration:
            return False

        # Preserve writing behavior when desk-writing evidence is strong.
        if (
            short_stats.avg_write_score >= max(self.config.write_score_threshold, 0.45)
            and short_stats.hand_lower_ratio >= 0.45
        ):
            return False

        if features.phone_present:
            return True

        current_eye_missing = (
            features.is_eye_closed
            or (
                features.ear_avg is not None
                and float(features.ear_avg) <= self.config.deep_head_down_eye_missing_ear_threshold
            )
        )
        window_eye_missing = short_stats.eye_closure_ratio >= self.config.deep_head_down_eye_closure_ratio_min
        return current_eye_missing or window_eye_missing

    def _check_phone_distraction(self, features: FrameFeatures,
                                  short_stats: WindowStats,
                                  long_stats: WindowStats,
                                  allow_direct: bool = True) -> Optional[tuple]:
        """Check for phone distraction pattern."""

        # Direct phone detection has highest priority
        if allow_direct:
            if features.phone_present:
                return (FocusState.PHONE_DISTRACTION, 0.95, "Phone detected")

            if short_stats.phone_ratio > 0.3:
                # Prevent sticky phone state after user returns to screen.
                is_currently_head_down = (
                    features.head_pitch is not None
                    and features.head_pitch < self.config.head_down_pitch_threshold + 1.0
                )
                if is_currently_head_down:
                    return (
                        FocusState.PHONE_DISTRACTION,
                        0.88,
                        f"Phone signal in recent window ({short_stats.phone_ratio:.0%})",
                    )
            return None

        continuous_eye_down = self._get_continuous_eye_down_seconds(features.timestamp)
        if continuous_eye_down < self.config.phone_eye_down_min_duration:
            return None

        if short_stats.eye_down_ratio < self.config.phone_eye_down_ratio_min:
            return None

        # Indirect detection via evidence scoring
        evidence, reasons = self._compute_phone_evidence(features, short_stats)

        # Require sustained evidence over a short horizon to reduce flicker
        self._phone_evidence_buffer.push(evidence, features.timestamp)
        evidence_items = self._phone_evidence_buffer.get_window(
            self.config.phone_evidence_window,
            features.timestamp,
        )

        values = [item.data for item in evidence_items]
        evidence_avg = sum(values) / len(values) if values else evidence
        evidence_peak = max(values) if values else evidence

        if (
            evidence_avg >= self.config.phone_evidence_avg_threshold
            and evidence_peak >= self.config.phone_evidence_peak_threshold
        ):
            confidence = min(0.92, 0.58 + evidence_avg * 0.35 + evidence_peak * 0.15)
            reason_text = ", ".join(reasons) if reasons else "sustained phone-like pattern"
            return (
                FocusState.PHONE_DISTRACTION,
                confidence,
                (
                    "Phone-like sustained: "
                    f"eye-down {continuous_eye_down:.0f}s, {reason_text} | avg={evidence_avg:.2f}"
                ),
            )

        # Very high single-frame evidence still qualifies
        if evidence >= self.config.phone_evidence_threshold + 0.15:
            confidence = min(0.88, 0.5 + evidence * 0.45)
            reason_text = ", ".join(reasons) if reasons else "strong phone-like pattern"
            return (
                FocusState.PHONE_DISTRACTION,
                confidence,
                f"Phone-like strong: {reason_text}",
            )

        return None

    def _compute_phone_evidence(self, features: FrameFeatures,
                                short_stats: WindowStats) -> tuple[float, List[str]]:
        """Compute evidence score (0-1+) for phone-like distraction pattern."""
        evidence = 0.0
        reasons: List[str] = []

        continuous_down = self._get_continuous_head_down_seconds(features.timestamp)
        continuous_eye_down = self._get_continuous_eye_down_seconds(features.timestamp)

        if continuous_eye_down >= self.config.phone_eye_down_min_duration:
            overflow = min(
                1.0,
                (continuous_eye_down - self.config.phone_eye_down_min_duration)
                / max(1.0, self.config.phone_eye_down_min_duration),
            )
            evidence += 0.35 + 0.1 * overflow
            reasons.append(f"eye-down {continuous_eye_down:.0f}s")

        if continuous_down > self.config.phone_head_down_continuous_max:
            overflow = min(
                1.0,
                (continuous_down - self.config.phone_head_down_continuous_max) /
                max(1.0, self.config.phone_head_down_continuous_max),
            )
            evidence += 0.35 + 0.15 * overflow
            reasons.append(f"head down {continuous_down:.0f}s")

        if short_stats.eye_down_ratio >= self.config.phone_eye_down_ratio_min:
            evidence += 0.18
            reasons.append(f"eye-down ratio {short_stats.eye_down_ratio:.0%}")
        else:
            evidence -= 0.12

        if self._is_eye_looking_screen(features):
            evidence -= 0.25
            reasons.append("screen gaze retained")

        if short_stats.head_down_ratio >= self.config.phone_head_down_ratio_min:
            evidence += 0.15
            reasons.append(f"head down ratio {short_stats.head_down_ratio:.0%}")

        if short_stats.avg_write_score < self.config.phone_write_score_max:
            write_deficit = (self.config.phone_write_score_max - short_stats.avg_write_score) / max(self.config.phone_write_score_max, 1e-6)
            evidence += 0.2 + 0.1 * min(1.0, write_deficit)
            reasons.append(f"low write score {short_stats.avg_write_score:.2f}")

        if short_stats.hand_lower_ratio < 0.25:
            evidence += 0.12
            reasons.append("hands not on desk")

        if features.hand_present and features.hand_region == "upper":
            evidence += 0.12
            reasons.append("hand in upper region")

        if short_stats.num_glances_up <= self.config.phone_glances_max:
            evidence += 0.15
            reasons.append("few glances")
        else:
            evidence -= 0.08

        if short_stats.blink_rate_per_min >= self.config.blink_rate_high_fatigue_min:
            evidence -= 0.1
            reasons.append("high blink-rate fatigue pattern")

        if abs(features.head_yaw or 0.0) > self.config.head_away_yaw_threshold:
            evidence += 0.08
            reasons.append("large yaw")

        # Reduce false positives for writing-like patterns
        if (
            short_stats.avg_write_score >= max(self.config.write_score_threshold, 0.45)
            and short_stats.hand_lower_ratio > 0.45
        ):
            evidence -= 0.2

        return evidence, reasons

    def _check_head_down_fatigue(self, features: FrameFeatures,
                                 short_stats: WindowStats) -> Optional[tuple]:
        """
        Classify head-down with elevated blink-rate as fatigue, not phone.

        Scientific basis used for thresholds:
        - Resting blink rates are commonly around 15-20/min (Stern et al., 1984).
        - During sustained visual tasks, blink rate often drops to ~4-10/min
          (Patel et al., 1991; Tsubota & Nakamori, 1993).
        - Elevated blink-rate above resting range with prolonged head-down posture
          is treated as visual fatigue/discomfort evidence.
        """
        continuous_down = self._get_continuous_head_down_seconds(features.timestamp)
        if continuous_down < self.config.fatigue_head_down_min_duration:
            return None

        if short_stats.blink_rate_per_min < self.config.blink_rate_high_fatigue_min:
            return None

        # If eye-down is sustained enough to satisfy the phone criterion,
        # phone branch should handle it instead of fatigue.
        if self._get_continuous_eye_down_seconds(features.timestamp) >= self.config.phone_eye_down_min_duration:
            return None

        return (
            FocusState.DROWSY_FATIGUE,
            0.72,
            (
                "Head-down with elevated blink rate "
                f"({short_stats.blink_rate_per_min:.1f}/min) suggests visual fatigue"
            ),
        )

    def _check_drowsy(self, features: FrameFeatures,
                      short_stats: WindowStats,
                      long_stats: WindowStats) -> Optional[tuple]:
        """Check for drowsiness/fatigue."""

        reasons = []
        score = 0.0

        # Eye closure
        if short_stats.eye_closure_ratio > self.config.drowsy_closure_ratio:
            reasons.append(f"eyes closed {short_stats.eye_closure_ratio:.0%}")
            score += 0.4

        if short_stats.perclos_ratio > self.config.perclos_threshold:
            reasons.append(f"perclos {short_stats.perclos_ratio:.0%}")
            score += 0.3

        if short_stats.avg_eye_closure_level > (self.config.drowsy_closure_ratio + 0.08):
            reasons.append(f"closure level {short_stats.avg_eye_closure_level:.2f}")
            score += 0.2

        # Low EAR
        if short_stats.avg_ear < self.config.drowsy_ear_threshold:
            reasons.append(f"low EAR ({short_stats.avg_ear:.2f})")
            score += 0.3

        # High idle + head down
        if (short_stats.avg_idle > self.config.drowsy_idle_threshold and
            short_stats.head_down_ratio > 0.5):
            reasons.append(f"idle + head down")
            score += 0.3

        if score >= 0.5:
            return (FocusState.DROWSY_FATIGUE, score,
                    "Drowsy: " + ", ".join(reasons))

        return None

    def _check_writing(self, features: FrameFeatures,
                       short_stats: WindowStats,
                       long_stats: WindowStats) -> Optional[tuple]:
        """
        Check for OFFSCREEN_WRITING pattern.

        Yêu cầu nghiêm ngặt hơn để tránh false positive:
        1. Phải có tay visible VÀ ở vị trí viết (lower region)
        2. Write score phải cao (có motion viết thực sự)
        3. Phải có glances up (kiểm tra tài liệu)
        """
        # Need sufficient hand context (allow intermittent misses)
        has_hand_context = features.hand_present or short_stats.hand_present_ratio >= 0.35
        if not has_hand_context:
            return None

        # Hands should mostly stay around desk area for writing
        lower_context = (
            (features.hand_present and features.hand_region == "lower")
            or short_stats.hand_lower_ratio >= 0.35
        )
        if not lower_context:
            return None

        score = 0.0
        reasons = []

        # Writing motion evidence
        strong_write_threshold = max(self.config.write_score_threshold, 0.45)

        if short_stats.avg_write_score >= strong_write_threshold:
            score += 0.45
            reasons.append(f"write_score={short_stats.avg_write_score:.2f}")
        elif (
            features.hand_present
            and features.hand_region == "lower"
            and features.hand_write_score >= self.config.write_score_threshold + 0.15
        ):
            score += 0.35
            reasons.append(f"current write={features.hand_write_score:.2f}")
        else:
            return None

        # Glances up (checking notes/screen periodically)
        if short_stats.num_glances_up >= self.config.write_glances_min:
            score += 0.2
            reasons.append(f"{short_stats.num_glances_up} glances up")

        if short_stats.hand_lower_ratio > 0.5:
            score += 0.15
            reasons.append("hands consistently at desk")

        if short_stats.hand_present_ratio >= 0.45:
            score += 0.1

        if short_stats.max_continuous_head_down < self.config.phone_head_down_continuous_max:
            score += 0.1

        if short_stats.avg_write_score >= 0.6:
            score += 0.1

        if score >= 0.65:
            return (FocusState.OFFSCREEN_WRITING, min(0.9, score),
                    "Writing: " + ", ".join(reasons))

        return None

    def _get_no_face_duration(self, now: float) -> float:
        """Get duration of no face detection."""
        items = self._feature_buffer.get_latest(100)

        if not items:
            return 0.0

        # Walk backwards to find last face
        for item in reversed(items):
            if item.data.face_detected:
                return now - item.timestamp

        # No face found in recent history
        if items:
            return now - items[0].timestamp

        return 0.0

    def _apply_hysteresis(self, intended_state: FocusState,
                          confidence: float,
                          now: Optional[float] = None) -> FocusState:
        """
        Apply hysteresis to prevent rapid state switching.

        State only changes if new state is maintained for hysteresis duration.
        """
        now = now if now is not None else self._last_frame_time
        focused_states = (FocusState.ON_SCREEN_READING, FocusState.OFFSCREEN_WRITING)

        # Keep focused states through short measurement-noise uncertainty.
        if (
            self._current_state in focused_states
            and intended_state == FocusState.UNCERTAIN
            and self._uncertain_reason_type == "measurement_noise"
        ):
            grace_remaining = max(0.0, self._focused_hold_until - now)
            if grace_remaining > 0.0:
                self._focused_hold_active = True
                self._uncertain_grace_remaining = grace_remaining
                self._pending_state = None
                self._pending_since = 0.0
                self._state_confidence = max(0.45, min(0.7, confidence))
                return self._current_state

        if intended_state == self._current_state:
            # Same state - reset pending
            self._pending_state = None
            self._pending_since = 0.0
            self._state_confidence = confidence
            return self._current_state

        if intended_state == self._pending_state:
            # Still pending same state - check duration
            pending_duration = now - self._pending_since
            required_duration = self.config.hysteresis_enter

            # Allow faster transitions for high-priority states
            if intended_state in (FocusState.AWAY, FocusState.PHONE_DISTRACTION):
                required_duration *= 0.5

            # Recover quickly from uncertain to focused when evidence is stable.
            if self._current_state == FocusState.UNCERTAIN and intended_state in focused_states:
                required_duration *= 0.22

            # Avoid flipping to UNCERTAIN on short noisy spans while currently focused.
            if self._current_state in focused_states and intended_state == FocusState.UNCERTAIN:
                required_duration *= 2.8

            if intended_state == FocusState.UNCERTAIN and self._uncertain_reason_type == "measurement_noise":
                required_duration *= 2.0

            if intended_state in focused_states and confidence >= 0.75:
                required_duration *= 0.7

            required_duration = max(0.12, required_duration)

            if pending_duration >= required_duration:
                # Transition!
                old_state = self._current_state
                self._current_state = intended_state
                self._state_start_time = now
                self._state_confidence = confidence
                self._pending_state = None

                self._transitions.append(StateTransition(
                    timestamp=now,
                    from_state=old_state,
                    to_state=intended_state,
                    reason=f"Hysteresis complete ({pending_duration:.1f}s)",
                    confidence=confidence
                ))

                if intended_state in focused_states:
                    self._last_focused_state = intended_state
                    self._last_focused_time = now
                    self._focused_hold_until = max(
                        self._focused_hold_until,
                        now + max(0.6, self.config.focused_state_hold_seconds),
                    )

                logger.info(f"State transition: {old_state.name} -> {intended_state.name}")
        else:
            # New pending state
            self._pending_state = intended_state
            self._pending_since = now

        return self._current_state

    def _get_refocus_recovery_cap(
        self,
        state: FocusState,
        timestamp: float,
        short_stats: WindowStats,
        state_confidence: float,
        features: FrameFeatures,
    ) -> Optional[float]:
        """
        Return optional recovery-rate cap (points/second) while validating refocus.

        None means no cap. 0 means recovery should be paused.
        """
        cfg = self.config
        focused_states = (FocusState.ON_SCREEN_READING, FocusState.OFFSCREEN_WRITING)
        distraction_states = (FocusState.PHONE_DISTRACTION, FocusState.DROWSY_FATIGUE, FocusState.AWAY)
        locked_rate = min(max(0.05, cfg.refocus_recover_rate_locked), max(0.05, cfg.score_recover_rate))

        if state in distraction_states:
            self._needs_refocus_validation = True
            self._refocus_candidate_since = None
            self._refocus_validated_since = None
            return None

        if not self._needs_refocus_validation:
            return None

        if state not in focused_states:
            self._refocus_candidate_since = None
            self._refocus_validated_since = None
            return 0.0

        stable_focus = (
            state_confidence >= cfg.refocus_confidence_min
            and short_stats.face_ratio >= cfg.refocus_face_ratio_min
            and not features.phone_present
            and short_stats.phone_ratio < 0.2
            and short_stats.eye_closure_ratio < cfg.drowsy_closure_ratio
        )

        if not stable_focus:
            self._refocus_candidate_since = None
            self._refocus_validated_since = None
            return 0.0

        if self._refocus_candidate_since is None:
            self._refocus_candidate_since = timestamp
            return locked_rate

        held_seconds = timestamp - self._refocus_candidate_since
        if held_seconds < cfg.refocus_validation_seconds:
            return locked_rate

        if self._refocus_validated_since is None:
            self._refocus_validated_since = timestamp

        ramp_seconds = max(0.15, cfg.refocus_recover_ramp_seconds)
        ramp_ratio = min(1.0, (timestamp - self._refocus_validated_since) / ramp_seconds)
        if ramp_ratio >= 1.0:
            self._needs_refocus_validation = False
            self._refocus_candidate_since = None
            self._refocus_validated_since = None
            return None

        return locked_rate + (cfg.score_recover_rate - locked_rate) * ramp_ratio

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _smooth_evidence_value(self, previous: float, current: float, dt: float) -> float:
        """Smooth noisy evidence channels into short-term accumulated signals."""
        tau = max(0.2, self.config.score_noise_softening_seconds)
        alpha = 1.0 - math.exp(-max(0.0, dt) / tau)
        return previous + (current - previous) * self._clamp01(alpha)

    def _compute_distraction_severity(
        self,
        features: FrameFeatures,
        short_stats: WindowStats,
        long_stats: WindowStats,
        state_confidence: float,
    ) -> float:
        """Compute continuous distraction severity in [0, 1]."""
        cfg = self.config

        phone_signal = 1.0 if features.phone_present else self._clamp01(short_stats.phone_ratio / 0.35)
        head_signal = self._clamp01(
            (short_stats.max_continuous_head_down - max(1.0, cfg.deep_head_down_min_duration))
            / max(4.0, cfg.phone_head_down_continuous_max)
        )
        eye_down_signal = self._clamp01(short_stats.eye_down_ratio / max(cfg.phone_eye_down_ratio_min, 1e-6))
        write_absence_signal = self._clamp01(
            (max(cfg.phone_write_score_max, 0.15) + 0.16 - short_stats.avg_write_score)
            / (max(cfg.phone_write_score_max, 0.15) + 0.16)
        )
        glance_absence_signal = 1.0 - self._clamp01(
            short_stats.num_glances_up / float(max(1, cfg.phone_glances_max + 2))
        )
        upper_hand_signal = (
            1.0 if (features.hand_present and features.hand_region == "upper")
            else self._clamp01((0.35 - short_stats.hand_lower_ratio) / 0.35)
        )
        long_phone_signal = self._clamp01(long_stats.phone_ratio / 0.3)
        confidence_signal = self._clamp01((state_confidence - 0.25) / 0.65)

        context_gate = max(phone_signal, head_signal, eye_down_signal, long_phone_signal)
        aux_scale = self._clamp01((context_gate - 0.1) / 0.9)

        severity = (
            (0.32 * phone_signal)
            + (0.23 * head_signal)
            + (0.17 * eye_down_signal)
            + (0.10 * long_phone_signal)
            + (0.10 * write_absence_signal * aux_scale)
            + (0.05 * glance_absence_signal * aux_scale)
            + (0.03 * upper_hand_signal * aux_scale)
            + (0.06 * confidence_signal * aux_scale)
        )

        if phone_signal > 0.8 and head_signal > 0.6 and write_absence_signal > 0.6:
            severity += 0.15

        if short_stats.blink_rate_per_min >= cfg.blink_rate_high_fatigue_min:
            severity -= 0.08

        if (
            short_stats.avg_write_score >= max(cfg.write_score_threshold, 0.45)
            and short_stats.hand_lower_ratio >= 0.45
        ):
            severity -= 0.16

        return self._clamp01(severity)

    def _compute_drowsiness_severity(
        self,
        features: FrameFeatures,
        short_stats: WindowStats,
        long_stats: WindowStats,
        state_confidence: float,
    ) -> float:
        """Compute continuous drowsiness/fatigue severity in [0, 1]."""
        cfg = self.config

        closure_signal = self._clamp01(short_stats.eye_closure_ratio / max(cfg.drowsy_closure_ratio, 1e-6))
        perclos_signal = self._clamp01(short_stats.perclos_ratio / max(cfg.perclos_threshold, 1e-6))
        ear_drop_signal = self._clamp01(
            (cfg.drowsy_ear_threshold - short_stats.avg_ear) / max(cfg.drowsy_ear_threshold, 1e-6)
        )
        idle_signal = self._clamp01(short_stats.avg_idle / max(cfg.drowsy_idle_threshold, 1.0))
        long_perclos_signal = self._clamp01(long_stats.perclos_ratio / max(cfg.perclos_threshold, 1e-6))
        head_down_fatigue_signal = self._clamp01(
            short_stats.max_continuous_head_down / max(cfg.fatigue_head_down_min_duration, 1.0)
        )
        blink_fatigue_signal = self._clamp01(
            (short_stats.blink_rate_per_min - cfg.blink_rate_high_fatigue_min)
            / max(cfg.blink_rate_high_fatigue_min, 1.0)
        )
        severity = (
            (0.26 * closure_signal)
            + (0.24 * perclos_signal)
            + (0.16 * ear_drop_signal)
            + (0.10 * idle_signal)
            + (0.09 * long_perclos_signal)
            + (0.08 * head_down_fatigue_signal)
            + (0.04 * blink_fatigue_signal)
        )

        confidence_signal = self._clamp01((state_confidence - 0.2) / 0.7)
        severity *= 0.7 + (0.3 * confidence_signal)

        if closure_signal > 0.85 and perclos_signal > 0.85:
            severity += 0.18

        if features.phone_present and short_stats.phone_ratio > 0.2:
            severity -= 0.08

        return self._clamp01(severity)

    def _compute_focus_stability(
        self,
        state: FocusState,
        features: FrameFeatures,
        short_stats: WindowStats,
        long_stats: WindowStats,
        state_confidence: float,
    ) -> float:
        """Compute stability of focused evidence in [0, 1]."""
        cfg = self.config

        face_signal = self._clamp01((short_stats.face_ratio - 0.45) / 0.5)
        conf_floor = max(0.1, min(0.9, cfg.score_confidence_floor_focused))
        confidence_signal = self._clamp01((state_confidence - conf_floor) / max(1e-6, 1.0 - conf_floor))
        phone_free_signal = 1.0 - self._clamp01((short_stats.phone_ratio + (0.6 if features.phone_present else 0.0)) / 0.7)
        closure_ok_signal = 1.0 - self._clamp01(short_stats.eye_closure_ratio / max(cfg.drowsy_closure_ratio + 0.1, 1e-6))
        perclos_ok_signal = 1.0 - self._clamp01(short_stats.perclos_ratio / max(cfg.perclos_threshold + 0.1, 1e-6))
        yaw_ok_signal = 1.0 - self._clamp01(abs(short_stats.avg_yaw) / (cfg.head_away_yaw_threshold + 10.0))
        screen_gaze_signal = self._clamp01((short_stats.avg_eye_look_up + (1.0 - short_stats.avg_eye_look_down)) * 0.5)
        writing_signal = self._clamp01(short_stats.avg_write_score / max(cfg.write_score_threshold, 1e-6))
        glance_signal = self._clamp01(short_stats.num_glances_up / 3.0)
        long_face_signal = self._clamp01((long_stats.face_ratio - 0.4) / 0.55)

        stability = (
            (0.20 * face_signal)
            + (0.14 * confidence_signal)
            + (0.14 * phone_free_signal)
            + (0.12 * closure_ok_signal)
            + (0.10 * perclos_ok_signal)
            + (0.10 * yaw_ok_signal)
            + (0.09 * screen_gaze_signal)
            + (0.07 * writing_signal)
            + (0.04 * glance_signal)
            + (0.10 * long_face_signal)
        )

        if state == FocusState.OFFSCREEN_WRITING:
            writing_context = (0.68 * writing_signal) + (0.32 * glance_signal)
            stability = (0.72 * stability) + (0.28 * writing_context)
        elif state == FocusState.ON_SCREEN_READING:
            stability = (0.78 * stability) + (0.22 * screen_gaze_signal)

        if (
            self._uncertain_reason_type == "measurement_noise"
            and self._current_state == FocusState.UNCERTAIN
            and max(0.0, features.timestamp - self._state_start_time)
            <= self.config.score_noise_softening_seconds
        ):
            stability = max(stability, 0.62)

        return self._clamp01(stability)

    def _compute_uncertainty_severity(
        self,
        features: FrameFeatures,
        short_stats: WindowStats,
        long_stats: WindowStats,
        state_confidence: float,
        distraction_severity: float,
        drowsiness_severity: float,
    ) -> float:
        """Compute how behaviorally meaningful UNCERTAIN evidence is in [0, 1]."""
        cfg = self.config

        uncertain_seconds = 0.0
        if self._current_state == FocusState.UNCERTAIN:
            uncertain_seconds = max(0.0, features.timestamp - self._state_start_time)

        conf_floor = max(0.05, min(0.85, cfg.score_confidence_floor_uncertain))
        confidence_low_signal = 1.0 - self._clamp01(
            (state_confidence - conf_floor) / max(1e-6, 1.0 - conf_floor)
        )
        face_loss_signal = 1.0 - self._clamp01((short_stats.face_ratio - 0.35) / 0.55)
        long_face_loss_signal = 1.0 - self._clamp01((long_stats.face_ratio - 0.3) / 0.6)
        duration_signal = self._clamp01(
            (uncertain_seconds - cfg.uncertain_short_soft_seconds)
            / max(0.5, cfg.uncertain_behavior_window_seconds)
        )

        severity = (
            (0.30 * face_loss_signal)
            + (0.12 * long_face_loss_signal)
            + (0.24 * confidence_low_signal)
            + (0.16 * duration_signal)
            + (0.10 * distraction_severity)
            + (0.08 * drowsiness_severity)
        )

        if self._uncertain_reason_type == "measurement_noise":
            if uncertain_seconds <= cfg.score_noise_softening_seconds:
                severity *= 0.2
            else:
                severity *= 0.55

        if self._uncertain_clean_candidate and uncertain_seconds <= cfg.uncertain_behavior_window_seconds:
            severity *= 0.65

        return self._clamp01(severity)

    def _compute_time_on_task_drift(
        self,
        short_stats: WindowStats,
        long_stats: WindowStats,
        focus_stability: float,
        drowsiness_severity: float,
    ) -> float:
        """Compute light fatigue drift penalty (points) from uninterrupted work duration."""
        cfg = self.config
        work_minutes = self._continuous_work_seconds / 60.0
        if work_minutes <= cfg.time_on_task_drift_start_minutes:
            return 0.0

        over_minutes = work_minutes - cfg.time_on_task_drift_start_minutes
        base_penalty = over_minutes * max(0.0, cfg.time_on_task_drift_per_minute)

        blink_pressure = self._clamp01(
            (short_stats.blink_rate_per_min - cfg.blink_rate_low_screen_max)
            / max(cfg.blink_rate_high_fatigue_min, 1.0)
        )
        eye_down_pressure = self._clamp01(short_stats.eye_down_ratio)
        perclos_pressure = self._clamp01(short_stats.perclos_ratio / max(cfg.perclos_threshold + 0.08, 1e-6))
        closure_pressure = self._clamp01(short_stats.eye_closure_ratio / max(cfg.drowsy_closure_ratio + 0.1, 1e-6))
        long_pressure = self._clamp01(long_stats.perclos_ratio / max(cfg.perclos_threshold + 0.06, 1e-6))

        fatigue_pressure = (
            (0.38 * drowsiness_severity)
            + (0.18 * eye_down_pressure)
            + (0.17 * perclos_pressure)
            + (0.14 * closure_pressure)
            + (0.08 * blink_pressure)
            + (0.05 * long_pressure)
        )
        fatigue_pressure = self._clamp01(fatigue_pressure)

        # If focused evidence is still very stable, drift remains mild.
        stability_mod = 0.55 + (0.45 * (1.0 - self._clamp01(focus_stability)))
        penalty = base_penalty * (0.3 + 0.7 * fatigue_pressure) * stability_mod
        return max(0.0, min(12.0, penalty))

    def _compute_target_score(
        self,
        state: FocusState,
        features: FrameFeatures,
        short_stats: WindowStats,
        long_stats: WindowStats,
        state_confidence: float,
        evidence: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute continuous score target (0-100) from state + accumulated evidence."""
        cfg = self.config

        if evidence is None:
            distraction = self._compute_distraction_severity(features, short_stats, long_stats, state_confidence)
            drowsiness = self._compute_drowsiness_severity(features, short_stats, long_stats, state_confidence)
            stability = self._compute_focus_stability(state, features, short_stats, long_stats, state_confidence)
            uncertainty = self._compute_uncertainty_severity(
                features,
                short_stats,
                long_stats,
                state_confidence,
                distraction,
                drowsiness,
            )
            time_drift = self._compute_time_on_task_drift(short_stats, long_stats, stability, drowsiness)
        else:
            distraction = self._clamp01(evidence.get("distraction", 0.0))
            drowsiness = self._clamp01(evidence.get("drowsiness", 0.0))
            stability = self._clamp01(evidence.get("stability", 0.0))
            uncertainty = self._clamp01(evidence.get("uncertainty", 0.0))
            time_drift = max(0.0, float(evidence.get("time_drift", 0.0)))

        if state == FocusState.ON_SCREEN_READING:
            posture_penalty = self._clamp01(short_stats.eye_down_ratio * 0.8) + self._clamp01(
                (abs(features.head_yaw or 0.0) - cfg.head_away_yaw_threshold * 0.6)
                / max(cfg.head_away_yaw_threshold, 1.0)
            )
            target = cfg.score_target_on_screen
            target -= (1.0 - stability) * 18.0
            target -= distraction * 11.0
            target -= drowsiness * 10.0
            target -= time_drift
            target -= posture_penalty * 2.6
            if short_stats.face_ratio < 0.65:
                target -= (0.65 - short_stats.face_ratio) * 14.0

            clean_focus_context = (
                short_stats.face_ratio >= 0.88
                and short_stats.phone_ratio <= 0.05
                and distraction <= 0.12
                and drowsiness <= 0.12
                and stability >= 0.72
                and time_drift <= 0.6
            )
            if clean_focus_context:
                target = max(target, 99.4)

            return max(72.0, min(100.0, target))

        if state == FocusState.OFFSCREEN_WRITING:
            write_support = self._clamp01(short_stats.avg_write_score / max(cfg.write_score_threshold, 1e-6))
            glance_support = self._clamp01(short_stats.num_glances_up / 3.0)
            desk_support = self._clamp01(short_stats.hand_lower_ratio / 0.6)
            target = cfg.score_target_writing
            target += (write_support - 0.55) * 9.0
            target += (glance_support * 4.0)
            target += (desk_support - 0.5) * 3.0
            target -= distraction * 13.0
            target -= drowsiness * 10.0
            target -= time_drift * 0.9
            target -= (1.0 - stability) * 14.0
            return max(68.0, min(98.0, target))

        if state == FocusState.PHONE_DISTRACTION:
            target = cfg.score_target_distraction + (1.0 - distraction) * 13.0
            target -= distraction * 18.0
            target -= drowsiness * 3.0
            if features.phone_present:
                target -= 4.0
            if short_stats.max_continuous_head_down > cfg.phone_head_down_continuous_max:
                overflow = self._clamp01(
                    (short_stats.max_continuous_head_down - cfg.phone_head_down_continuous_max)
                    / max(cfg.phone_head_down_continuous_max, 1.0)
                )
                target -= 4.0 * overflow
            return max(4.0, min(42.0, target))

        if state == FocusState.DROWSY_FATIGUE:
            severity = max(drowsiness, distraction * 0.35)
            target = cfg.score_target_drowsy + (1.0 - severity) * 10.0
            target -= severity * 20.0
            target -= time_drift * 0.8
            if short_stats.perclos_ratio > cfg.perclos_threshold:
                target -= 3.0
            return max(6.0, min(48.0, target))

        if state == FocusState.AWAY:
            no_face_duration = self._get_no_face_duration(features.timestamp)
            away_signal = max(
                self._clamp01((1.0 - short_stats.face_ratio) / 0.8),
                self._clamp01(no_face_duration / max(cfg.away_no_face_seconds, 1.0)),
            )
            target = cfg.score_target_away + (1.0 - away_signal) * 8.0
            target -= away_signal * 18.0
            target -= time_drift * 0.5
            return max(8.0, min(52.0, target))

        # UNCERTAIN should stay near neutral unless uncertainty becomes behaviorally strong.
        uncertain_seconds = 0.0
        if self._current_state == FocusState.UNCERTAIN:
            uncertain_seconds = max(0.0, features.timestamp - self._state_start_time)

        recent_focused_context = (
            self._last_focused_state in (FocusState.ON_SCREEN_READING, FocusState.OFFSCREEN_WRITING)
            and (features.timestamp - self._last_focused_time)
            <= (cfg.focused_state_hold_seconds + 1.8)
        )
        noise_uncertainty = self._uncertain_reason_type == "measurement_noise"

        if self._last_focused_state == FocusState.OFFSCREEN_WRITING:
            focused_anchor = max(cfg.score_target_writing - 6.0, 80.0)
        else:
            focused_anchor = max(cfg.score_target_on_screen - 9.0, 82.0)

        if noise_uncertainty and uncertain_seconds <= cfg.score_noise_softening_seconds:
            base_target = (0.7 * self._raw_focus_score) + (0.3 * focused_anchor)
        elif recent_focused_context and uncertain_seconds <= cfg.uncertain_behavior_window_seconds:
            base_target = (0.65 * focused_anchor) + (0.35 * cfg.score_target_uncertain)
        else:
            base_target = cfg.score_target_uncertain

        target = base_target
        target -= uncertainty * 12.0
        target -= distraction * 6.0
        target -= drowsiness * 7.0
        target -= time_drift * 0.6

        if uncertain_seconds <= cfg.uncertain_short_soft_seconds:
            target += 2.0

        low_bound = 60.0 if recent_focused_context else 55.0
        return max(low_bound, min(90.0, target))

    def _update_focus_score(
        self,
        state: FocusState,
        timestamp: float,
        features: FrameFeatures,
        short_stats: WindowStats,
        long_stats: WindowStats,
        state_confidence: float,
    ) -> None:
        """Update focus score using asymmetric, evidence-accumulated dynamics."""
        cfg = self.config

        # Time-normalized scoring to avoid FPS-dependent behavior.
        dt = timestamp - self._last_score_update_time
        if dt <= 0:
            dt = 1.0 / 30.0
        dt = min(dt, cfg.max_score_delta_time)
        self._last_score_update_time = timestamp

        distraction_now = self._compute_distraction_severity(features, short_stats, long_stats, state_confidence)
        drowsiness_now = self._compute_drowsiness_severity(features, short_stats, long_stats, state_confidence)
        stability_now = self._compute_focus_stability(state, features, short_stats, long_stats, state_confidence)
        uncertainty_now = self._compute_uncertainty_severity(
            features,
            short_stats,
            long_stats,
            state_confidence,
            distraction_now,
            drowsiness_now,
        )
        time_drift_now = self._compute_time_on_task_drift(
            short_stats,
            long_stats,
            stability_now,
            drowsiness_now,
        )

        self._smoothed_distraction_severity = self._smooth_evidence_value(
            self._smoothed_distraction_severity,
            distraction_now,
            dt,
        )
        self._smoothed_drowsiness_severity = self._smooth_evidence_value(
            self._smoothed_drowsiness_severity,
            drowsiness_now,
            dt,
        )
        self._smoothed_focus_stability = self._smooth_evidence_value(
            self._smoothed_focus_stability,
            stability_now,
            dt,
        )
        self._smoothed_uncertainty_severity = self._smooth_evidence_value(
            self._smoothed_uncertainty_severity,
            uncertainty_now,
            dt,
        )
        self._smoothed_time_drift = self._smooth_evidence_value(
            self._smoothed_time_drift,
            time_drift_now,
            dt,
        )

        evidence = {
            "distraction": self._smoothed_distraction_severity,
            "drowsiness": self._smoothed_drowsiness_severity,
            "stability": self._smoothed_focus_stability,
            "uncertainty": self._smoothed_uncertainty_severity,
            "time_drift": self._smoothed_time_drift,
        }

        target_score = self._compute_target_score(
            state,
            features,
            short_stats,
            long_stats,
            state_confidence,
            evidence,
        )

        conf = self._clamp01(state_confidence)
        if state in (FocusState.ON_SCREEN_READING, FocusState.OFFSCREEN_WRITING):
            conf = max(cfg.score_confidence_floor_focused, conf)
            conf_scale = 0.45 + 0.55 * self._clamp01(conf)
        elif state == FocusState.UNCERTAIN:
            conf = max(cfg.score_confidence_floor_uncertain, conf)
            conf_scale = 0.35 + 0.50 * self._clamp01(conf)
        else:
            conf_scale = 0.40 + 0.60 * self._clamp01(conf)

        effective_target = self._raw_focus_score + (target_score - self._raw_focus_score) * conf_scale
        gap = effective_target - self._raw_focus_score

        max_up = cfg.score_recover_rate * dt
        max_down = cfg.score_drop_rate * dt

        focused_states = (FocusState.ON_SCREEN_READING, FocusState.OFFSCREEN_WRITING)
        is_stable_focused = (
            state in focused_states
            and self._smoothed_focus_stability >= 0.72
            and self._smoothed_distraction_severity < 0.35
            and self._smoothed_drowsiness_severity < 0.35
            and conf >= cfg.score_confidence_floor_focused
        )

        if state in focused_states:
            recover_rate = (
                cfg.score_recover_rate_focused_stable
                if is_stable_focused
                else cfg.score_recover_rate_focused_unstable
            )
            max_up = recover_rate * dt
            negative_pressure = max(self._smoothed_distraction_severity, self._smoothed_drowsiness_severity)
            max_down = cfg.score_drop_rate_distraction_soft * (0.18 + 0.55 * negative_pressure) * dt

            if is_stable_focused and timestamp <= self._break_recovery_boost_until:
                max_up *= 1.22

        elif state == FocusState.UNCERTAIN:
            uncertain_seconds = 0.0
            if self._current_state == FocusState.UNCERTAIN:
                uncertain_seconds = max(0.0, timestamp - self._state_start_time)

            if (
                self._uncertain_reason_type == "measurement_noise"
                and uncertain_seconds <= cfg.score_noise_softening_seconds
            ):
                max_up = cfg.score_recover_rate_focused_unstable * 0.6 * dt
                max_down = cfg.score_uncertain_soft_penalty * 0.35 * dt
            else:
                max_up = cfg.score_recover_rate_focused_unstable * (
                    0.45 + 0.35 * (1.0 - self._smoothed_uncertainty_severity)
                ) * dt
                max_down = (
                    cfg.score_uncertain_soft_penalty
                    + self._smoothed_uncertainty_severity * cfg.score_drop_rate_distraction_soft * 0.4
                ) * dt

        elif state == FocusState.PHONE_DISTRACTION:
            strong = self._smoothed_distraction_severity >= 0.72
            drop_rate = (
                cfg.score_drop_rate_distraction_strong
                if strong
                else cfg.score_drop_rate_distraction_soft
            )
            max_down = drop_rate * (0.65 + 0.65 * self._smoothed_distraction_severity) * dt
            max_up = cfg.score_recover_rate_focused_unstable * 0.08 * dt

        elif state == FocusState.DROWSY_FATIGUE:
            drop_rate = cfg.score_drop_rate_drowsy_strong * (0.6 + 0.7 * self._smoothed_drowsiness_severity)
            max_down = drop_rate * dt
            max_up = cfg.score_recover_rate_focused_unstable * 0.10 * dt

        elif state == FocusState.AWAY:
            max_down = cfg.score_drop_rate_distraction_soft * 0.75 * dt
            max_up = cfg.score_recover_rate_focused_unstable * 0.06 * dt

        # After distraction, require sustained and confident refocus before score rises.
        recovery_rate_cap = self._get_refocus_recovery_cap(
            state,
            timestamp,
            short_stats,
            state_confidence,
            features,
        )
        if recovery_rate_cap is not None:
            max_up = min(max_up, max(0.0, recovery_rate_cap) * dt)

        if gap >= 0.0:
            delta = min(gap, max_up)
        else:
            delta = max(gap, -max_down)

        # Keep legacy score weights as mild interpretable multipliers.
        if state == FocusState.ON_SCREEN_READING:
            delta *= max(0.7, cfg.score_on_screen_weight)
        elif state == FocusState.OFFSCREEN_WRITING:
            delta *= max(0.7, cfg.score_writing_weight)
        elif state == FocusState.UNCERTAIN:
            if (
                self._uncertain_reason_type == "measurement_noise"
                and self._smoothed_uncertainty_severity <= 0.3
            ):
                delta *= 0.45
            else:
                delta *= (1.0 + cfg.score_uncertain_penalty * 0.45)
        elif state == FocusState.PHONE_DISTRACTION:
            delta *= (1.0 + cfg.score_distraction_penalty * min(1.4, 0.6 + self._smoothed_distraction_severity))
        elif state == FocusState.DROWSY_FATIGUE:
            delta *= (1.0 + cfg.score_drowsy_penalty * min(1.4, 0.7 + self._smoothed_drowsiness_severity))
        elif state == FocusState.AWAY:
            delta *= (1.0 + cfg.score_away_penalty * 0.8)

        self._raw_focus_score = max(0.0, min(100.0, self._raw_focus_score + delta))

        # Apply time-adjusted EMA smoothing.
        alpha = 1 - (1 - cfg.score_smoothing) ** max(1.0, dt * 30.0)
        self._focus_score = (alpha * self._raw_focus_score) + ((1 - alpha) * self._focus_score)

    def _update_time_tracking(self, state: FocusState, timestamp: float) -> None:
        """Track continuous work time and significant breaks for drift/recovery."""
        cfg = self.config

        dt = timestamp - self._last_time_tracking_timestamp
        if dt <= 0.0:
            dt = 1.0 / 30.0
        dt = min(dt, 1.0)

        focused_states = (FocusState.ON_SCREEN_READING, FocusState.OFFSCREEN_WRITING)

        if state in focused_states:
            if self._nonfocused_since is not None:
                break_duration = max(0.0, timestamp - self._nonfocused_since)

                if break_duration >= 8.0:
                    reduction_scale = min(
                        1.0,
                        break_duration / max(30.0, cfg.time_on_task_drift_start_minutes * 60.0 * 0.5),
                    )
                    self._continuous_work_seconds *= max(0.0, 1.0 - 0.85 * reduction_scale)

                if break_duration >= cfg.break_recovery_boost_window_seconds:
                    self._last_significant_break_time = timestamp
                    self._break_recovery_boost_until = timestamp + cfg.break_recovery_boost_window_seconds

                self._nonfocused_since = None

            self._continuous_work_seconds = min(6.0 * 3600.0, self._continuous_work_seconds + dt)
        else:
            if self._nonfocused_since is None:
                self._nonfocused_since = timestamp

            break_duration = max(0.0, timestamp - self._nonfocused_since)
            if break_duration >= max(180.0, cfg.break_recovery_boost_window_seconds * 4.0):
                self._continuous_work_seconds = 0.0
                self._last_significant_break_time = timestamp
                self._break_recovery_boost_until = timestamp + cfg.break_recovery_boost_window_seconds

        self._last_time_tracking_timestamp = timestamp

    def get_short_stats(self) -> WindowStats:
        """Get short window statistics."""
        return self._compute_stats(self.config.short_window)

    def get_long_stats(self) -> WindowStats:
        """Get long window statistics."""
        return self._compute_stats(self.config.long_window)

    def get_recent_transitions(self, count: int = 10) -> List[StateTransition]:
        """Get recent state transitions."""
        return list(self._transitions)[-count:]

    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information."""
        nonfocused_seconds = 0.0
        if self._nonfocused_since is not None:
            nonfocused_seconds = max(0.0, self._last_frame_time - self._nonfocused_since)

        return {
            "state": self._current_state.name,
            "state_duration": self.state_duration,
            "confidence": self._state_confidence,
            "reason": self._last_reason,
            "intended_state": self._last_intended_state.name,
            "intended_confidence": self._last_intended_confidence,
            "focus_score": self._focus_score,
            "raw_score": self._raw_focus_score,
            "pending_state": self._pending_state.name if self._pending_state else None,
            "session_duration": self.session_duration,
            "profile_name": self._profile_name,
            "personalization_weight": self._personalization_weight,
            "personalization_stage": self._personalization_stage,
            "uncertain_reason_type": self._uncertain_reason_type,
            "focused_hold_active": self._focused_hold_active,
            "uncertain_grace_remaining": self._uncertain_grace_remaining,
            "uncertain_clean_candidate": self._uncertain_clean_candidate,
            "continuous_work_seconds": self._continuous_work_seconds,
            "continuous_work_minutes": self._continuous_work_seconds / 60.0,
            "nonfocused_seconds": nonfocused_seconds,
            "break_recovery_boost_remaining": max(0.0, self._break_recovery_boost_until - self._last_frame_time),
            "evidence_distraction": self._smoothed_distraction_severity,
            "evidence_drowsiness": self._smoothed_drowsiness_severity,
            "evidence_stability": self._smoothed_focus_stability,
            "evidence_uncertainty": self._smoothed_uncertainty_severity,
            "evidence_time_drift": self._smoothed_time_drift,
        }

    def reset(self) -> None:
        """Reset engine state."""
        self._feature_buffer.clear()
        self._current_state = FocusState.UNCERTAIN
        self._state_start_time = time.time()
        self._state_confidence = 0.0
        self._pending_state = None
        self._pending_since = 0.0
        self._focus_score = 100.0
        self._raw_focus_score = 100.0
        self._transitions.clear()
        self._phone_evidence_buffer.clear()
        self._glance_timestamps.clear()
        self._head_down_start = None
        self._eye_down_start = None
        self._session_start = time.time()
        self._last_score_update_time = self._session_start
        self._last_frame_time = self._session_start
        self._last_reason = "Reset"
        self._last_intended_state = FocusState.UNCERTAIN
        self._last_intended_confidence = 0.0
        self._needs_refocus_validation = False
        self._refocus_candidate_since = None
        self._refocus_validated_since = None
        self._last_focused_state = None
        self._last_focused_time = self._session_start
        self._focused_hold_until = self._session_start
        self._uncertain_reason_type = "none"
        self._focused_hold_active = False
        self._uncertain_grace_remaining = 0.0
        self._uncertain_clean_candidate = False
        self._continuous_work_seconds = 0.0
        self._nonfocused_since = None
        self._last_significant_break_time = self._session_start
        self._last_time_tracking_timestamp = self._session_start
        self._break_recovery_boost_until = self._session_start
        self._smoothed_focus_stability = 0.78
        self._smoothed_distraction_severity = 0.0
        self._smoothed_drowsiness_severity = 0.0
        self._smoothed_uncertainty_severity = 0.0
        self._smoothed_time_drift = 0.0

        logger.info("Focus engine reset")


# Convenience function to create features from vision module outputs
def create_frame_features(
    timestamp: Optional[float] = None,
    face_detected: bool = False,
    head_pitch: Optional[float] = None,
    head_yaw: Optional[float] = None,
    head_roll: Optional[float] = None,
    ear_avg: Optional[float] = None,
    is_eye_closed: bool = False,
    blink_detected: bool = False,
    hand_present: bool = False,
    hand_write_score: float = 0.0,
    hand_region: str = "middle",
    phone_present: bool = False,
    idle_seconds: float = 0.0,
    eye_look_down: Optional[float] = None,
    eye_look_up: Optional[float] = None,
    eye_closure_level: Optional[float] = None,
) -> FrameFeatures:
    """Create FrameFeatures with defaults."""
    return FrameFeatures(
        timestamp=timestamp or time.time(),
        face_detected=face_detected,
        head_pitch=head_pitch,
        head_yaw=head_yaw,
        head_roll=head_roll,
        ear_avg=ear_avg,
        is_eye_closed=is_eye_closed,
        blink_detected=blink_detected,
        hand_present=hand_present,
        hand_write_score=hand_write_score,
        hand_region=hand_region,
        phone_present=phone_present,
        idle_seconds=idle_seconds,
        eye_look_down=eye_look_down,
        eye_look_up=eye_look_up,
        eye_closure_level=eye_closure_level,
    )
