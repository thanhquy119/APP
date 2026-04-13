"""
Pytest tests for FocusEngine state machine.
Tests critical state transitions, especially head_down + hand_write_score scenarios.

Run with: pytest tests/test_focus_engine.py -v
"""

import pytest
import time
from typing import List

from app.logic.focus_engine import (
    FocusEngine,
    FocusEngineConfig,
    FocusState,
    FrameFeatures,
    create_frame_features
)
from app.utils.ring_buffer import RingBuffer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def engine() -> FocusEngine:
    """Create a fresh FocusEngine with default config."""
    config = FocusEngineConfig(
        short_window=10.0,
        long_window=30.0,
        hysteresis_enter=0.5,  # Faster for tests
        hysteresis_exit=0.5,
    )
    return FocusEngine(config)


@pytest.fixture
def fast_engine() -> FocusEngine:
    """Create FocusEngine with minimal hysteresis for quick tests."""
    config = FocusEngineConfig(
        short_window=5.0,
        long_window=10.0,
        hysteresis_enter=0.1,
        hysteresis_exit=0.1,
    )
    return FocusEngine(config)


def generate_frames(
    count: int,
    start_time: float,
    interval: float = 0.033,  # ~30 fps
    **kwargs
) -> List[FrameFeatures]:
    """Generate a sequence of frames with given parameters."""
    frames = []
    for i in range(count):
        ts = start_time + i * interval
        frames.append(create_frame_features(timestamp=ts, **kwargs))
    return frames


def feed_frames(engine: FocusEngine, frames: List[FrameFeatures]) -> FocusState:
    """Feed frames to engine and return final state."""
    state = None
    for frame in frames:
        state = engine.process_frame(frame)
    return state


# ============================================================================
# Ring Buffer Tests
# ============================================================================

class TestRingBuffer:
    """Tests for RingBuffer utility."""

    def test_basic_push_and_get(self):
        """Test basic push and retrieval."""
        buf = RingBuffer[int](max_size=10)

        base_time = time.time()
        for i in range(5):
            buf.push(i, base_time + i)

        assert len(buf) == 5
        assert buf.get_last_data() == 4

    def test_window_query(self):
        """Test time-window queries."""
        buf = RingBuffer[int](max_size=100)

        base_time = 1000.0
        for i in range(50):
            buf.push(i, base_time + i * 0.1)  # 0.1s apart

        # Get last 2 seconds (should be ~20 items)
        window_data = buf.get_window_data(2.0, base_time + 5.0)
        assert len(window_data) >= 15
        assert len(window_data) <= 25

    def test_ratio_where(self):
        """Test ratio calculation with predicate."""
        buf = RingBuffer[int](max_size=100)

        base_time = 1000.0  # Fixed timestamp
        # Push 10 even and 10 odd numbers
        for i in range(20):
            buf.push(i, base_time + i * 0.1)

        # Use explicit end_time to match our fixed timestamps
        ratio = buf.ratio_where(10.0, lambda x: x % 2 == 0, end_time=base_time + 2.0)
        assert 0.4 <= ratio <= 0.6  # Should be ~50%

    def test_max_size_limit(self):
        """Test that buffer respects max size."""
        buf = RingBuffer[int](max_size=10)

        for i in range(100):
            buf.push(i)

        assert len(buf) == 10
        assert buf.get_last_data() == 99


# ============================================================================
# Focus Engine Basic Tests
# ============================================================================

class TestFocusEngineBasic:
    """Basic FocusEngine functionality tests."""

    def test_initial_state(self, engine: FocusEngine):
        """Test initial state is UNCERTAIN."""
        assert engine.current_state == FocusState.UNCERTAIN
        assert engine.focus_score == 100.0

    def test_single_frame(self, engine: FocusEngine):
        """Test processing a single frame."""
        frame = create_frame_features(
            face_detected=True,
            head_pitch=-5.0,
            head_yaw=0.0
        )

        state = engine.process_frame(frame)
        assert state is not None

    def test_reset(self, engine: FocusEngine):
        """Test engine reset."""
        # Process some frames
        for i in range(10):
            engine.process_frame(create_frame_features(
                timestamp=time.time() + i * 0.033,
                face_detected=True
            ))

        engine.reset()

        assert engine.current_state == FocusState.UNCERTAIN
        assert engine.focus_score == 100.0
        assert engine.state_duration < 1.0

    def test_state_info_contains_reason_and_confidence(self, engine: FocusEngine):
        """State info should expose diagnostic fields for UI/debug."""
        engine.process_frame(create_frame_features(
            timestamp=time.time(),
            face_detected=True,
            head_pitch=-5.0,
            head_yaw=0.0,
        ))

        info = engine.get_state_info()

        assert "reason" in info
        assert "intended_state" in info
        assert "intended_confidence" in info
        assert isinstance(info["reason"], str)


# ============================================================================
# ON_SCREEN_READING Tests
# ============================================================================

class TestOnScreenReading:
    """Tests for ON_SCREEN_READING state detection."""

    def test_looking_at_screen(self, fast_engine: FocusEngine):
        """User looking at screen should be ON_SCREEN_READING."""
        base_time = time.time()

        # Generate 5 seconds of looking at screen
        frames = generate_frames(
            count=150,
            start_time=base_time,
            face_detected=True,
            head_pitch=-5.0,  # Slightly down, but not head_down
            head_yaw=0.0,     # Facing forward
            ear_avg=0.28,     # Eyes open
            is_eye_closed=False
        )

        final_state = feed_frames(fast_engine, frames)

        assert final_state == FocusState.ON_SCREEN_READING

    def test_slight_head_movement(self, fast_engine: FocusEngine):
        """Small head movements should still be ON_SCREEN_READING."""
        base_time = time.time()

        frames = []
        for i in range(150):
            # Slight yaw variation
            yaw = 5.0 * (i % 20 - 10) / 10  # -5 to +5 degrees
            frames.append(create_frame_features(
                timestamp=base_time + i * 0.033,
                face_detected=True,
                head_pitch=-3.0 + (i % 10) * 0.2,
                head_yaw=yaw,
                ear_avg=0.26
            ))

        final_state = feed_frames(fast_engine, frames)

        assert final_state == FocusState.ON_SCREEN_READING

    def test_borderline_yaw_with_neutral_pose(self, fast_engine: FocusEngine):
        """Slightly over yaw threshold should still count as on-screen in neutral posture."""
        base_time = time.time()

        frames = generate_frames(
            count=150,
            start_time=base_time,
            face_detected=True,
            head_pitch=-5.0,
            head_yaw=32.0,
            ear_avg=0.30,
            is_eye_closed=False,
            eye_look_down=0.20,
            phone_present=False,
        )

        final_state = feed_frames(fast_engine, frames)
        info = fast_engine.get_state_info()

        assert final_state == FocusState.ON_SCREEN_READING
        assert info["confidence"] >= 0.80
        assert "deg" in info["reason"]

    def test_large_yaw_still_uncertain(self, fast_engine: FocusEngine):
        """Very large yaw should not be re-labeled as on-screen by grace logic."""
        base_time = time.time()

        frames = generate_frames(
            count=150,
            start_time=base_time,
            face_detected=True,
            head_pitch=-5.0,
            head_yaw=45.0,
            ear_avg=0.30,
            is_eye_closed=False,
            eye_look_down=0.10,
            phone_present=False,
        )

        final_state = feed_frames(fast_engine, frames)

        assert final_state == FocusState.UNCERTAIN


# ============================================================================
# OFFSCREEN_WRITING Tests - CRITICAL
# ============================================================================

class TestOffscreenWriting:
    """
    Tests for OFFSCREEN_WRITING state detection.

    This is the critical case: head down + writing pattern = still focused!
    """

    def test_head_down_with_high_write_score(self, fast_engine: FocusEngine):
        """
        CRITICAL TEST: Head down + high write_score = OFFSCREEN_WRITING.

        This is the main case we need to get right:
        - Student looking down at desk
        - Hands in writing position with writing motion
        - Should NOT trigger distraction warning
        """
        base_time = time.time()

        # Generate frames: head down, hands writing
        frames = generate_frames(
            count=200,  # ~6.6 seconds
            start_time=base_time,
            face_detected=True,
            head_pitch=-25.0,  # Looking down (below -15 threshold)
            head_yaw=0.0,
            ear_avg=0.25,
            is_eye_closed=False,
            hand_present=True,
            hand_write_score=0.7,  # High write score (above 0.4 threshold)
            hand_region="lower",   # Hands at desk level
            phone_present=False
        )

        # Add some glances up (looking at screen/book)
        for i in range(0, len(frames), 30):  # Every ~1 second
            if i < len(frames):
                frames[i] = create_frame_features(
                    timestamp=frames[i].timestamp,
                    face_detected=True,
                    head_pitch=-2.0,  # Glance up
                    head_yaw=0.0,
                    ear_avg=0.25,
                    hand_present=True,
                    hand_write_score=0.6,
                    hand_region="lower",
                    phone_present=False
                )

        final_state = feed_frames(fast_engine, frames)

        assert final_state == FocusState.OFFSCREEN_WRITING, \
            f"Expected OFFSCREEN_WRITING but got {final_state.name}"

    def test_head_down_moderate_write_score(self, fast_engine: FocusEngine):
        """Head down with moderate write score should be OFFSCREEN_WRITING."""
        base_time = time.time()

        frames = generate_frames(
            count=200,
            start_time=base_time,
            face_detected=True,
            head_pitch=-20.0,
            head_yaw=2.0,
            hand_present=True,
            hand_write_score=0.5,  # Moderate write score
            hand_region="lower",
            phone_present=False
        )

        # Add glances
        for i in range(0, len(frames), 40):
            if i < len(frames):
                frames[i] = create_frame_features(
                    timestamp=frames[i].timestamp,
                    face_detected=True,
                    head_pitch=0.0,  # Glance up
                    head_yaw=0.0,
                    hand_present=True,
                    hand_write_score=0.5,
                    hand_region="lower"
                )

        final_state = feed_frames(fast_engine, frames)

        assert final_state == FocusState.OFFSCREEN_WRITING

    def test_writing_with_intermittent_hand_detection(self, fast_engine: FocusEngine):
        """Writing with hands sometimes not detected should still work."""
        base_time = time.time()

        frames = []
        for i in range(200):
            # Hand detected 70% of the time
            hand_present = (i % 10) < 7

            frames.append(create_frame_features(
                timestamp=base_time + i * 0.033,
                face_detected=True,
                head_pitch=-22.0,
                head_yaw=0.0,
                hand_present=hand_present,
                hand_write_score=0.65 if hand_present else 0.0,
                hand_region="lower" if hand_present else "middle",
                phone_present=False
            ))

        # Add glances
        for i in range(0, len(frames), 30):
            frames[i] = create_frame_features(
                timestamp=frames[i].timestamp,
                face_detected=True,
                head_pitch=-5.0,  # Glance up
                head_yaw=0.0,
                hand_present=True,
                hand_write_score=0.6,
                hand_region="lower"
            )

        final_state = feed_frames(fast_engine, frames)

        # Should recognize as writing despite intermittent hand detection
        assert final_state in (FocusState.OFFSCREEN_WRITING, FocusState.UNCERTAIN)

    def test_head_down_no_hands_is_not_writing(self, fast_engine: FocusEngine):
        """Head down without hands should NOT be OFFSCREEN_WRITING."""
        base_time = time.time()

        frames = generate_frames(
            count=200,
            start_time=base_time,
            face_detected=True,
            head_pitch=-25.0,
            head_yaw=0.0,
            hand_present=False,  # No hands detected
            hand_write_score=0.0,
            phone_present=False
        )

        final_state = feed_frames(fast_engine, frames)

        # Should NOT be writing without hands
        assert final_state != FocusState.OFFSCREEN_WRITING


# ============================================================================
# PHONE_DISTRACTION Tests
# ============================================================================

class TestPhoneDistraction:
    """Tests for PHONE_DISTRACTION state detection."""

    def test_phone_detected(self, fast_engine: FocusEngine):
        """Direct phone detection should trigger PHONE_DISTRACTION."""
        base_time = time.time()

        frames = generate_frames(
            count=100,
            start_time=base_time,
            face_detected=True,
            head_pitch=-20.0,
            phone_present=True  # Phone detected by vision
        )

        final_state = feed_frames(fast_engine, frames)

        assert final_state == FocusState.PHONE_DISTRACTION

    def test_head_down_no_writing_pattern(self, fast_engine: FocusEngine):
        """
        Head down + no writing pattern + few glances = likely phone.

        This is the opposite of the writing case:
        - Head down for long time
        - No hand writing motion
        - Very few glances up
        """
        base_time = time.time()

        # Need enough continuous head down to trigger phone detection (>15s)
        # 20 seconds of continuous head down, no writing
        frames = generate_frames(
            count=700,  # ~23 seconds at 30fps
            start_time=base_time,
            face_detected=True,
            head_pitch=-30.0,  # Deep head down
            head_yaw=5.0,
            hand_present=True,
            hand_write_score=0.1,  # Very low write score
            hand_region="upper",   # Hands up (holding phone)
            phone_present=False    # Phone not directly detected
        )

        # NO glances up at all to trigger phone detection

        final_state = feed_frames(fast_engine, frames)

        # Should be PHONE_DISTRACTION or at least not focused
        assert final_state in (FocusState.PHONE_DISTRACTION, FocusState.UNCERTAIN)

    def test_head_down_eye_down_over_45s_is_phone(self):
        """Head-down + sustained eye-down over 45s should classify as PHONE_DISTRACTION."""
        engine = FocusEngine(FocusEngineConfig(
            short_window=10.0,
            long_window=30.0,
            hysteresis_enter=0.1,
            phone_eye_down_min_duration=45.0,
        ))

        base_time = time.time()
        frames = generate_frames(
            count=500,            # 50s at 10 FPS
            start_time=base_time,
            interval=0.1,
            face_detected=True,
            head_pitch=-25.0,
            head_yaw=3.0,
            hand_present=True,
            hand_write_score=0.08,
            hand_region="upper",
            eye_look_down=0.82,
            eye_look_up=0.05,
            phone_present=False,
        )

        final_state = feed_frames(engine, frames)
        assert final_state == FocusState.PHONE_DISTRACTION

    def test_head_down_but_screen_gaze_low_blink_not_phone(self):
        """Head-down with screen gaze and low blink-rate should not be phone distraction."""
        engine = FocusEngine(FocusEngineConfig(
            short_window=10.0,
            long_window=20.0,
            hysteresis_enter=0.1,
            blink_rate_low_screen_max=10.0,
        ))

        base_time = time.time()
        frames = []
        for i in range(360):  # 12s at 30 FPS
            frames.append(create_frame_features(
                timestamp=base_time + i * 0.033,
                face_detected=True,
                head_pitch=-19.0,
                head_yaw=1.0,
                eye_look_down=0.12,
                eye_look_up=0.55,
                blink_detected=(i % 180 == 0),  # ~6-10 blinks/min in 10s window
                hand_present=False,
                hand_write_score=0.0,
                hand_region="none",
                phone_present=False,
            ))

        final_state = feed_frames(engine, frames)
        assert final_state != FocusState.PHONE_DISTRACTION
        assert final_state in (FocusState.ON_SCREEN_READING, FocusState.UNCERTAIN)

    def test_head_down_high_blink_pattern_maps_to_fatigue(self):
        """Head-down + elevated blink-rate should map to fatigue-like state, not phone."""
        engine = FocusEngine(FocusEngineConfig(
            short_window=10.0,
            long_window=20.0,
            hysteresis_enter=0.1,
            fatigue_head_down_min_duration=15.0,
            blink_rate_high_fatigue_min=22.0,
            phone_eye_down_min_duration=45.0,
        ))

        base_time = time.time()
        frames = []
        for i in range(260):  # 26s at 10 FPS
            frames.append(create_frame_features(
                timestamp=base_time + i * 0.1,
                face_detected=True,
                head_pitch=-23.0,
                head_yaw=2.0,
                ear_avg=0.26,
                is_eye_closed=False,
                blink_detected=(i % 20 == 0),  # ~30 blinks/min
                eye_look_down=0.20,
                eye_look_up=0.32,
                hand_present=False,
                hand_write_score=0.0,
                hand_region="none",
                phone_present=False,
            ))

        final_state = feed_frames(engine, frames)
        assert final_state == FocusState.DROWSY_FATIGUE

    def test_head_down_with_desk_writing_not_phone(self, fast_engine: FocusEngine):
        """Head-down desk writing should not be misclassified as phone distraction."""
        base_time = time.time()

        frames = []
        for i in range(450):  # ~15s
            is_glance = (i % 60 == 0)
            frames.append(create_frame_features(
                timestamp=base_time + i * 0.033,
                face_detected=True,
                head_pitch=-4.0 if is_glance else -23.0,
                head_yaw=2.0,
                hand_present=(i % 8) != 0,  # intermittent hand misses
                hand_write_score=0.55 if (i % 8) != 0 else 0.0,
                hand_region="lower" if (i % 8) != 0 else "middle",
                phone_present=False,
            ))

        final_state = feed_frames(fast_engine, frames)

        assert final_state != FocusState.PHONE_DISTRACTION
        base_time = time.time()

        # Scenario 1: Writing (high write score, glances)
        writing_frames = []
        for i in range(200):
            is_glance = (i % 30 == 0)
            writing_frames.append(create_frame_features(
                timestamp=base_time + i * 0.033,
                face_detected=True,
                head_pitch=-3.0 if is_glance else -25.0,
                hand_present=True,
                hand_write_score=0.7,
                hand_region="lower"
            ))

        engine1 = FocusEngine(FocusEngineConfig(hysteresis_enter=0.1))
        writing_state = feed_frames(engine1, writing_frames)

        # Scenario 2: Phone (low write score, no glances)
        phone_frames = generate_frames(
            count=200,
            start_time=base_time,
            face_detected=True,
            head_pitch=-25.0,  # Same head position!
            hand_present=True,
            hand_write_score=0.15,  # Low write score
            hand_region="upper"     # Hands up (phone)
        )

        engine2 = FocusEngine(FocusEngineConfig(hysteresis_enter=0.1))
        phone_state = feed_frames(engine2, phone_frames)

        # They should be different!
        assert writing_state == FocusState.OFFSCREEN_WRITING
        assert phone_state in (FocusState.PHONE_DISTRACTION, FocusState.UNCERTAIN)

    def test_short_phone_like_burst_does_not_trigger_state(self):
        """Short noisy phone-like bursts should not immediately flip to PHONE_DISTRACTION."""
        engine = FocusEngine(FocusEngineConfig(
            short_window=5.0,
            long_window=10.0,
            hysteresis_enter=0.1,
        ))

        base_time = time.time()
        states = []

        # Stable on-screen period
        for i in range(60):
            states.append(engine.process_frame(create_frame_features(
                timestamp=base_time + i * 0.033,
                face_detected=True,
                head_pitch=-5.0,
                head_yaw=0.0,
            )))

        # Short burst (~0.8s) with phone-like pattern but no direct phone signal
        burst_start = base_time + 60 * 0.033
        for i in range(24):
            states.append(engine.process_frame(create_frame_features(
                timestamp=burst_start + i * 0.033,
                face_detected=True,
                head_pitch=-28.0,
                head_yaw=8.0,
                hand_present=True,
                hand_write_score=0.05,
                hand_region="upper",
                phone_present=False,
            )))

        # Back to on-screen
        recover_start = burst_start + 24 * 0.033
        for i in range(60):
            states.append(engine.process_frame(create_frame_features(
                timestamp=recover_start + i * 0.033,
                face_detected=True,
                head_pitch=-4.0,
                head_yaw=0.0,
            )))

        # Sustained evidence gating should avoid PHONE state for this short burst
        assert FocusState.PHONE_DISTRACTION not in states


# ============================================================================
# DROWSY_FATIGUE Tests
# ============================================================================

class TestDrowsyFatigue:
    """Tests for DROWSY_FATIGUE state detection."""

    def test_eyes_closed_prolonged(self, fast_engine: FocusEngine):
        """Prolonged eye closure should trigger DROWSY."""
        base_time = time.time()

        frames = generate_frames(
            count=200,
            start_time=base_time,
            face_detected=True,
            head_pitch=-10.0,
            ear_avg=0.15,  # Very low EAR (eyes mostly closed)
            is_eye_closed=True
        )

        final_state = feed_frames(fast_engine, frames)

        assert final_state == FocusState.DROWSY_FATIGUE

    def test_high_eye_closure_ratio(self, fast_engine: FocusEngine):
        """High eye closure ratio should trigger DROWSY."""
        base_time = time.time()

        frames = []
        for i in range(300):  # More frames
            # Eyes closed 60% of time (above 30% threshold)
            is_closed = (i % 5) < 3

            frames.append(create_frame_features(
                timestamp=base_time + i * 0.033,
                face_detected=True,
                head_pitch=-20.0,  # Head down (below -15 threshold)
                ear_avg=0.10 if is_closed else 0.28,  # Very low EAR when closed
                is_eye_closed=is_closed,
                idle_seconds=20.0  # High idle to help trigger drowsy
            ))

        final_state = feed_frames(fast_engine, frames)

        assert final_state == FocusState.DROWSY_FATIGUE


# ============================================================================
# AWAY Tests
# ============================================================================

class TestAway:
    """Tests for AWAY state detection."""

    def test_no_face_detected(self, fast_engine: FocusEngine):
        """No face for extended period should be AWAY."""
        base_time = time.time()

        # First some frames with face
        frames = generate_frames(
            count=30,
            start_time=base_time,
            face_detected=True,
            head_pitch=0.0
        )

        # Then no face for 15 seconds
        frames.extend(generate_frames(
            count=450,
            start_time=base_time + 1.0,
            face_detected=False
        ))

        final_state = feed_frames(fast_engine, frames)

        assert final_state == FocusState.AWAY

    def test_intermittent_face_not_away(self, fast_engine: FocusEngine):
        """Intermittent face detection should not be AWAY."""
        base_time = time.time()

        frames = []
        for i in range(200):
            # Face detected 70% of time
            face = (i % 10) < 7

            frames.append(create_frame_features(
                timestamp=base_time + i * 0.033,
                face_detected=face,
                head_pitch=0.0 if face else None
            ))

        final_state = feed_frames(fast_engine, frames)

        assert final_state != FocusState.AWAY


# ============================================================================
# Hysteresis Tests
# ============================================================================

class TestHysteresis:
    """Tests for hysteresis behavior."""

    def test_no_rapid_switching(self):
        """State should not switch rapidly."""
        config = FocusEngineConfig(
            hysteresis_enter=1.0,  # 1 second to enter
            hysteresis_exit=1.0
        )
        engine = FocusEngine(config)

        base_time = time.time()

        # Alternate between looking at screen and looking away every 0.3s
        # Should NOT cause state changes due to hysteresis
        frames = []
        for i in range(60):  # 2 seconds
            if (i // 9) % 2 == 0:  # Every 0.3s alternate
                pitch = -5.0  # Looking at screen
            else:
                pitch = -25.0  # Head down

            frames.append(create_frame_features(
                timestamp=base_time + i * 0.033,
                face_detected=True,
                head_pitch=pitch
            ))

        states_seen = set()
        for frame in frames:
            state = engine.process_frame(frame)
            states_seen.add(state)

        # Due to hysteresis, should mostly stay in initial state
        assert len(states_seen) <= 2

    def test_sustained_change_triggers_transition(self):
        """Sustained state should eventually transition."""
        config = FocusEngineConfig(
            hysteresis_enter=0.5,
            short_window=5.0
        )
        engine = FocusEngine(config)

        base_time = time.time()

        # Start with on-screen reading
        frames = generate_frames(
            count=100,
            start_time=base_time,
            face_detected=True,
            head_pitch=-5.0
        )
        feed_frames(engine, frames)

        initial_state = engine.current_state

        # Then sustained phone distraction
        phone_frames = generate_frames(
            count=100,
            start_time=base_time + 3.33,
            face_detected=True,
            head_pitch=-30.0,
            phone_present=True
        )
        final_state = feed_frames(engine, phone_frames)

        # Should have transitioned
        assert final_state == FocusState.PHONE_DISTRACTION


# ============================================================================
# Focus Score Tests
# ============================================================================

class TestFocusScore:
    """Tests for focus score calculation."""

    def test_score_increases_when_focused(self, fast_engine: FocusEngine):
        """Focus score should stay near max when on-screen reading."""
        initial_score = fast_engine.focus_score

        frames = generate_frames(
            count=300,
            start_time=time.time(),
            face_detected=True,
            head_pitch=-5.0,
            ear_avg=0.28
        )
        feed_frames(fast_engine, frames)

        assert fast_engine.focus_score >= initial_score - 1.0
        assert fast_engine.focus_score >= 95.0

    def test_score_decreases_when_distracted(self, fast_engine: FocusEngine):
        """Focus score should decrease during distraction."""
        # First build up some score
        frames = generate_frames(
            count=200,
            start_time=time.time(),
            face_detected=True,
            head_pitch=-5.0
        )
        feed_frames(fast_engine, frames)

        score_before_distraction = fast_engine.focus_score

        # Then distraction
        distraction_frames = generate_frames(
            count=200,
            start_time=time.time() + 10,
            face_detected=True,
            head_pitch=-30.0,
            phone_present=True
        )
        feed_frames(fast_engine, distraction_frames)

        assert fast_engine.focus_score < score_before_distraction

    def test_writing_maintains_good_score(self, fast_engine: FocusEngine):
        """OFFSCREEN_WRITING should maintain decent focus score."""
        base_time = time.time()

        # Writing frames
        frames = []
        for i in range(300):
            is_glance = (i % 30 == 0)
            frames.append(create_frame_features(
                timestamp=base_time + i * 0.033,
                face_detected=True,
                head_pitch=-3.0 if is_glance else -22.0,
                hand_present=True,
                hand_write_score=0.7,
                hand_region="lower"
            ))

        feed_frames(fast_engine, frames)

        # Score should be reasonably high (writing is focused activity)
        assert fast_engine.focus_score >= 45

    def test_score_similar_across_frame_rates(self):
        """Score progression should be largely time-based, not frame-count based."""
        config = FocusEngineConfig(hysteresis_enter=0.1, short_window=5.0, long_window=10.0)

        engine_30fps = FocusEngine(config)
        engine_15fps = FocusEngine(config)

        start = time.time()

        # Same 10 seconds but different sampling rates
        frames_30 = generate_frames(
            count=300,
            start_time=start,
            interval=10.0 / 300,
            face_detected=True,
            head_pitch=-5.0,
            head_yaw=0.0,
            ear_avg=0.28,
        )
        frames_15 = generate_frames(
            count=150,
            start_time=start,
            interval=10.0 / 150,
            face_detected=True,
            head_pitch=-5.0,
            head_yaw=0.0,
            ear_avg=0.28,
        )

        feed_frames(engine_30fps, frames_30)
        feed_frames(engine_15fps, frames_15)

        # Allow some tolerance due to smoothing/hysteresis behavior
        assert abs(engine_30fps.focus_score - engine_15fps.focus_score) <= 5.0

    def test_large_timestamp_gap_does_not_spike_score(self):
        """A large timestamp gap should not cause unrealistic score jumps."""
        config = FocusEngineConfig(hysteresis_enter=0.1)
        engine = FocusEngine(config)

        t0 = time.time()
        engine.process_frame(create_frame_features(
            timestamp=t0,
            face_detected=True,
            head_pitch=-5.0,
            head_yaw=0.0,
        ))
        score_before = engine.focus_score

        # Simulate delayed next frame after a long gap
        engine.process_frame(create_frame_features(
            timestamp=t0 + 30.0,
            face_detected=True,
            head_pitch=-5.0,
            head_yaw=0.0,
        ))
        score_after = engine.focus_score

        assert score_after - score_before < 2.0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests simulating real usage patterns."""

    def test_typical_study_session(self, fast_engine: FocusEngine):
        """Simulate a typical study session with various activities."""
        base_time = time.time()
        t = base_time

        # Phase 1: Reading (2s)
        for i in range(60):
            fast_engine.process_frame(create_frame_features(
                timestamp=t + i * 0.033,
                face_detected=True,
                head_pitch=-5.0,
                ear_avg=0.27
            ))
        t += 2

        assert fast_engine.current_state == FocusState.ON_SCREEN_READING

        # Phase 2: Writing notes (3s)
        for i in range(90):
            is_glance = (i % 25 == 0)
            fast_engine.process_frame(create_frame_features(
                timestamp=t + i * 0.033,
                face_detected=True,
                head_pitch=-3.0 if is_glance else -20.0,
                hand_present=True,
                hand_write_score=0.65,
                hand_region="lower"
            ))
        t += 3

        assert fast_engine.current_state == FocusState.OFFSCREEN_WRITING

        # Phase 3: Back to reading (1s)
        for i in range(30):
            fast_engine.process_frame(create_frame_features(
                timestamp=t + i * 0.033,
                face_detected=True,
                head_pitch=-5.0
            ))
        t += 1

        # Score should still be good
        assert fast_engine.focus_score >= 40

    def test_distraction_recovery(self, fast_engine: FocusEngine):
        """Test that score recovers after distraction ends."""
        base_time = time.time()
        t = base_time

        # Build initial score (more frames)
        for i in range(150):
            fast_engine.process_frame(create_frame_features(
                timestamp=t + i * 0.033,
                face_detected=True,
                head_pitch=-5.0,
                ear_avg=0.28
            ))
        t += 5.0

        score_before = fast_engine.focus_score

        # Short distraction
        for i in range(60):  # Shorter distraction
            fast_engine.process_frame(create_frame_features(
                timestamp=t + i * 0.033,
                face_detected=True,
                head_pitch=-30.0,
                phone_present=True
            ))
        t += 2.0

        score_during = fast_engine.focus_score
        # Score should have decreased
        assert score_during <= score_before

        # Recovery - back to focused (longer recovery)
        for i in range(300):
            fast_engine.process_frame(create_frame_features(
                timestamp=t + i * 0.033,
                face_detected=True,
                head_pitch=-5.0,
                ear_avg=0.28
            ))

        # Score should recover
        assert fast_engine.focus_score >= score_during
