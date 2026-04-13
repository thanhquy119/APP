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
    drowsy_idle_threshold: float = 10.0       # Seconds of keyboard/mouse idle

    # Hysteresis (seconds to maintain state before switching)
    hysteresis_enter: float = 1.2             # Time to enter a new state
    hysteresis_exit: float = 2.4              # Time to exit current state

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

    def __init__(self, config: Optional[FocusEngineConfig] = None):
        self.config = config or FocusEngineConfig()

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
        self._last_intended_state = intended_state
        self._last_intended_confidence = confidence
        self._last_reason = reason

        # Apply hysteresis using frame timestamp
        new_state = self._apply_hysteresis(intended_state, confidence, features.timestamp)

        # Update focus score
        self._update_focus_score(
            new_state,
            features.timestamp,
            features,
            short_stats,
            self._state_confidence,
        )

        # Track time
        self._update_time_tracking(new_state)

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

            return (FocusState.UNCERTAIN, 0.5,
                    f"Head down {continuous_down:.0f}s, no confirmed phone pattern")

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
        return (FocusState.UNCERTAIN, 0.22, "Insufficient stable pose cues")

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
            focused_states = (FocusState.ON_SCREEN_READING, FocusState.OFFSCREEN_WRITING)

            # Allow faster transitions for high-priority states
            if intended_state in (FocusState.AWAY, FocusState.PHONE_DISTRACTION):
                required_duration *= 0.5

            # Recover quickly from uncertain to focused when evidence is stable.
            if self._current_state == FocusState.UNCERTAIN and intended_state in focused_states:
                required_duration *= 0.35

            # Avoid flipping to UNCERTAIN on short noisy spans while currently focused.
            if self._current_state in focused_states and intended_state == FocusState.UNCERTAIN:
                required_duration *= 2.2

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

                logger.info(f"State transition: {old_state.name} -> {intended_state.name}")
        else:
            # New pending state
            self._pending_state = intended_state
            self._pending_since = now

        return self._current_state

    def _compute_target_score(self, state: FocusState,
                              features: FrameFeatures,
                              short_stats: WindowStats) -> float:
        """Compute target focus score (0-100) for the current state."""
        cfg = self.config

        if state == FocusState.ON_SCREEN_READING:
            engagement = 1.0
            if short_stats.face_ratio < 0.8:
                engagement -= 0.2
            if abs(features.head_yaw or 0.0) > cfg.head_away_yaw_threshold * 0.8:
                engagement -= 0.2
            target = cfg.score_target_on_screen + (engagement - 0.75) * 20.0
            return max(82.0, min(100.0, target))

        if state == FocusState.OFFSCREEN_WRITING:
            writing_quality = min(1.0, short_stats.avg_write_score)
            glance_bonus = min(4.0, short_stats.num_glances_up * 1.5)
            target = cfg.score_target_writing + writing_quality * 10.0 + glance_bonus
            return max(75.0, min(98.0, target))

        if state == FocusState.PHONE_DISTRACTION:
            severity = 0.0
            if features.phone_present:
                severity += 0.35
            if short_stats.max_continuous_head_down > cfg.phone_head_down_continuous_max:
                severity += 0.3
            if short_stats.eye_down_ratio >= cfg.phone_eye_down_ratio_min:
                severity += 0.2
            if short_stats.avg_write_score < 0.2:
                severity += 0.2
            if short_stats.num_glances_up <= cfg.phone_glances_max:
                severity += 0.15
            # Elevated blink-rate is handled by fatigue branch, so de-emphasize phone severity.
            if short_stats.blink_rate_per_min >= cfg.blink_rate_high_fatigue_min:
                severity -= 0.1
            severity = max(0.0, severity)
            target = cfg.score_target_distraction - severity * 20.0
            return max(5.0, min(35.0, target))

        if state == FocusState.DROWSY_FATIGUE:
            drowsy_factor = min(1.0, short_stats.eye_closure_ratio + max(0.0, cfg.drowsy_ear_threshold - short_stats.avg_ear))
            target = cfg.score_target_drowsy - drowsy_factor * 14.0
            return max(12.0, min(45.0, target))

        if state == FocusState.AWAY:
            away_factor = min(1.0, short_stats.window_seconds / max(cfg.long_window, 1.0))
            target = cfg.score_target_away - away_factor * 10.0
            return max(18.0, min(45.0, target))

        # UNCERTAIN
        uncertainty_drag = 0.0
        if short_stats.head_down_ratio > 0.65 and short_stats.avg_write_score < 0.2:
            uncertainty_drag += 3.0
        if short_stats.face_ratio < 0.5:
            uncertainty_drag += 2.0
        if short_stats.phone_ratio > 0.2:
            uncertainty_drag += 2.0
        target = cfg.score_target_uncertain - uncertainty_drag
        return max(68.0, min(84.0, target))

    def _update_focus_score(self, state: FocusState, timestamp: float,
                            features: FrameFeatures,
                            short_stats: WindowStats,
                            state_confidence: float) -> None:
        """Update focus score using bounded movement toward a state-dependent target."""
        cfg = self.config

        # Time-normalized scoring to avoid FPS-dependent behavior
        dt = timestamp - self._last_score_update_time
        if dt <= 0:
            dt = 1.0 / 30.0
        dt = min(dt, cfg.max_score_delta_time)
        self._last_score_update_time = timestamp

        target_score = self._compute_target_score(state, features, short_stats)

        # Confidence-aware damping: low confidence pulls target closer to current score.
        conf = max(0.0, min(1.0, state_confidence))
        if state in (FocusState.ON_SCREEN_READING, FocusState.OFFSCREEN_WRITING):
            conf_scale = 0.55 + 0.45 * conf
        elif state in (FocusState.PHONE_DISTRACTION, FocusState.DROWSY_FATIGUE, FocusState.AWAY):
            conf_scale = 0.45 + 0.55 * conf
        else:
            conf_scale = 0.35 + 0.65 * conf

        effective_target = self._raw_focus_score + (target_score - self._raw_focus_score) * conf_scale
        gap = effective_target - self._raw_focus_score

        max_up = cfg.score_recover_rate * dt
        max_down = cfg.score_drop_rate * dt

        if state in (FocusState.ON_SCREEN_READING, FocusState.OFFSCREEN_WRITING):
            # Once user refocuses, recover score faster and prevent extra downward drift.
            max_up *= 1.8 if self._raw_focus_score < 88.0 else 1.35
            max_down *= 0.45
        elif state == FocusState.UNCERTAIN:
            # UNCERTAIN should be a cautious state, not a heavy penalty state.
            max_down *= 0.28
            max_up *= 1.1

        if gap >= 0:
            delta = min(gap, max_up)
        else:
            delta = max(gap, -max_down)

        # Keep existing score weights meaningful as mild multipliers.
        if state == FocusState.ON_SCREEN_READING:
            delta *= max(0.7, cfg.score_on_screen_weight)
        elif state == FocusState.OFFSCREEN_WRITING:
            delta *= max(0.7, cfg.score_writing_weight)
        elif state == FocusState.UNCERTAIN:
            delta *= (1.0 + cfg.score_uncertain_penalty)
        elif state == FocusState.PHONE_DISTRACTION:
            delta *= (1.0 + cfg.score_distraction_penalty)
        elif state == FocusState.DROWSY_FATIGUE:
            delta *= (1.0 + cfg.score_drowsy_penalty)
        elif state == FocusState.AWAY:
            delta *= (1.0 + cfg.score_away_penalty)

        self._raw_focus_score = max(0.0, min(100.0, self._raw_focus_score + delta))

        # Apply time-adjusted EMA smoothing
        alpha = 1 - (1 - cfg.score_smoothing) ** max(1.0, dt * 30.0)
        self._focus_score = (alpha * self._raw_focus_score +
                             (1 - alpha) * self._focus_score)

    def _update_time_tracking(self, state: FocusState) -> None:
        """Track time in focused vs distracted states."""
        # This would be called each frame, but we accumulate
        # based on frame timing. Simplified here.
        pass

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
            "session_duration": self.session_duration
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
        self._last_reason = "Reset"
        self._last_intended_state = FocusState.UNCERTAIN
        self._last_intended_confidence = 0.0

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
    )
