"""
Hand detection and analysis using MediaPipe Hands.
Calculates hand_write_score to detect writing/note-taking behavior.
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Optional, Tuple, List, NamedTuple
from dataclasses import dataclass
from collections import deque
import time

logger = logging.getLogger(__name__)


class HandRegion:
    """Regions of the frame for hand position analysis."""
    UPPER = "upper"       # Upper third - phone holding area
    MIDDLE = "middle"     # Middle third
    LOWER = "lower"       # Lower third - desk/writing area
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"


class HandState(NamedTuple):
    """Hand detection state for current frame."""
    hand_present: bool           # Any hand detected
    num_hands: int               # Number of hands (0, 1, or 2)
    hand_region_v: str           # Vertical region (upper/middle/lower)
    hand_region_h: str           # Horizontal region (left/center/right)
    hand_y_normalized: float     # Average hand Y position (0=top, 1=bottom)
    hand_motion_energy: float    # Motion intensity (0-1)
    hand_write_score: float      # Writing likelihood score (0-1)
    is_writing_pattern: bool     # Likely writing based on patterns
    landmarks_left: Optional[np.ndarray]   # Left hand landmarks
    landmarks_right: Optional[np.ndarray]  # Right hand landmarks


@dataclass
class HandConfig:
    """Configuration for hand detection."""
    max_num_hands: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False

    # Region thresholds (normalized 0-1)
    lower_region_threshold: float = 0.55   # Below this Y = lower region (desk)
    upper_region_threshold: float = 0.35   # Above this Y = upper region

    # Motion analysis
    motion_window_seconds: float = 2.0     # Window for motion analysis
    motion_smoothing: float = 0.3          # EMA smoothing for motion

    # Writing pattern detection
    write_score_threshold: float = 0.5     # Score above this = likely writing
    write_motion_min: float = 0.05         # Minimum motion for writing
    write_motion_max: float = 0.4          # Maximum motion (too much = not writing)
    write_region_weight: float = 0.4       # Weight for being in lower region
    write_motion_weight: float = 0.4       # Weight for motion pattern
    write_stability_weight: float = 0.2    # Weight for position stability


class HandAnalyzer:
    """
    Analyzes hand presence and motion patterns to detect writing behavior.

    Writing pattern characteristics:
    - Hands in lower portion of frame (desk area)
    - Small, repetitive motion (not large gestures)
    - Relatively stable position with micro-movements
    """

    # Hand landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    def __init__(self, config: Optional[HandConfig] = None):
        self.config = config or HandConfig()

        self._hands = None
        self._initialized = False

        # Motion tracking
        self._prev_landmarks: List[Optional[np.ndarray]] = [None, None]
        self._motion_history: deque = deque(maxlen=120)  # ~4 seconds at 30fps
        self._position_history: deque = deque(maxlen=120)

        # Smoothed values
        self._smooth_motion = 0.0
        self._smooth_write_score = 0.0

        # Statistics
        self._detection_count = 0
        self._no_hand_count = 0

    def initialize(self) -> bool:
        """Initialize MediaPipe Hands."""
        try:
            self._hands = mp.solutions.hands.Hands(
                max_num_hands=self.config.max_num_hands,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
                static_image_mode=self.config.static_image_mode
            )
            self._initialized = True
            logger.info("Hand detector initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Hand detector: {e}")
            self._initialized = False
            return False

    def release(self) -> None:
        """Release resources."""
        if self._hands:
            self._hands.close()
            self._hands = None
        self._initialized = False

    def process(self, frame: np.ndarray) -> HandState:
        """
        Process frame and analyze hand state.

        Args:
            frame: BGR image from OpenCV

        Returns:
            HandState with detection results and write score
        """
        if not self._initialized:
            if not self.initialize():
                return self._empty_state()

        current_time = time.time()
        h, w = frame.shape[:2]

        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = self._hands.process(rgb_frame)

            if not results.multi_hand_landmarks:
                self._no_hand_count += 1
                self._motion_history.append((current_time, 0.0))
                self._smooth_motion *= 0.9  # Decay
                return self._empty_state()

            self._detection_count += 1

            # Extract landmarks for each hand
            landmarks_list = []
            handedness_list = []

            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                # Convert to numpy array
                lm_array = np.array([
                    [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
                ], dtype=np.float32)
                landmarks_list.append(lm_array)
                handedness_list.append(handedness.classification[0].label)

            # Calculate average hand position
            all_y = []
            all_x = []
            for lm in landmarks_list:
                all_y.extend(lm[:, 1])
                all_x.extend(lm[:, 0])

            avg_y = np.mean(all_y)
            avg_x = np.mean(all_x)

            # Determine region
            if avg_y > self.config.lower_region_threshold:
                region_v = HandRegion.LOWER
            elif avg_y < self.config.upper_region_threshold:
                region_v = HandRegion.UPPER
            else:
                region_v = HandRegion.MIDDLE

            if avg_x < 0.33:
                region_h = HandRegion.LEFT
            elif avg_x > 0.67:
                region_h = HandRegion.RIGHT
            else:
                region_h = HandRegion.CENTER

            # Calculate motion energy
            motion_energy = self._calculate_motion(landmarks_list, current_time)

            # Store position for stability analysis
            self._position_history.append((current_time, avg_x, avg_y))

            # Calculate write score
            write_score = self._calculate_write_score(
                region_v, motion_energy, current_time
            )

            # Separate left and right hand landmarks
            landmarks_left = None
            landmarks_right = None
            for lm, hand in zip(landmarks_list, handedness_list):
                if hand == "Left":
                    landmarks_left = lm
                else:
                    landmarks_right = lm

            # Determine if writing pattern
            is_writing = write_score >= self.config.write_score_threshold

            return HandState(
                hand_present=True,
                num_hands=len(landmarks_list),
                hand_region_v=region_v,
                hand_region_h=region_h,
                hand_y_normalized=avg_y,
                hand_motion_energy=motion_energy,
                hand_write_score=write_score,
                is_writing_pattern=is_writing,
                landmarks_left=landmarks_left,
                landmarks_right=landmarks_right
            )

        except Exception as e:
            logger.error(f"Hand processing error: {e}")
            return self._empty_state()

    def _calculate_motion(self, landmarks_list: List[np.ndarray],
                          current_time: float) -> float:
        """Calculate motion energy from landmark changes."""
        motion = 0.0

        for i, lm in enumerate(landmarks_list[:2]):  # Max 2 hands
            if i < len(self._prev_landmarks) and self._prev_landmarks[i] is not None:
                # Calculate average displacement
                prev = self._prev_landmarks[i]
                if prev.shape == lm.shape:
                    diff = np.linalg.norm(lm[:, :2] - prev[:, :2], axis=1)
                    motion += np.mean(diff)

            # Store for next frame
            if i < len(self._prev_landmarks):
                self._prev_landmarks[i] = lm.copy()

        # Normalize motion (typical range 0-0.1 for small movements)
        motion = min(1.0, motion * 5.0)

        # Store in history
        self._motion_history.append((current_time, motion))

        # Apply smoothing
        alpha = self.config.motion_smoothing
        self._smooth_motion = alpha * motion + (1 - alpha) * self._smooth_motion

        return self._smooth_motion

    def _calculate_write_score(self, region_v: str, motion_energy: float,
                               current_time: float) -> float:
        """
        Calculate writing likelihood score.

        Writing characteristics:
        1. Hands in lower region (desk)
        2. Moderate, consistent motion (not still, not wild)
        3. Stable general position
        """
        score = 0.0

        # Factor 1: Region (lower = writing area)
        if region_v == HandRegion.LOWER:
            region_score = 1.0
        elif region_v == HandRegion.MIDDLE:
            region_score = 0.4
        else:
            region_score = 0.0

        # Factor 2: Motion pattern
        # Writing has small, consistent motion
        if self.config.write_motion_min <= motion_energy <= self.config.write_motion_max:
            # Ideal motion range for writing
            motion_score = 1.0 - abs(motion_energy - 0.15) / 0.25
            motion_score = max(0.0, min(1.0, motion_score))
        elif motion_energy < self.config.write_motion_min:
            # Too still - might be just resting
            motion_score = motion_energy / self.config.write_motion_min * 0.3
        else:
            # Too much motion - gesturing, not writing
            motion_score = max(0.0, 1.0 - (motion_energy - self.config.write_motion_max))

        # Factor 3: Position stability
        window_start = current_time - self.config.motion_window_seconds
        recent_positions = [(t, x, y) for t, x, y in self._position_history if t > window_start]

        if len(recent_positions) > 5:
            x_vals = [x for _, x, _ in recent_positions]
            y_vals = [y for _, _, y in recent_positions]

            # Low variance = stable position = more likely writing
            position_variance = np.std(x_vals) + np.std(y_vals)
            stability_score = max(0.0, 1.0 - position_variance * 5)
        else:
            stability_score = 0.5  # Not enough data

        # Combine factors
        score = (
            self.config.write_region_weight * region_score +
            self.config.write_motion_weight * motion_score +
            self.config.write_stability_weight * stability_score
        )

        # Apply smoothing
        alpha = 0.3
        self._smooth_write_score = alpha * score + (1 - alpha) * self._smooth_write_score

        return self._smooth_write_score

    def _empty_state(self) -> HandState:
        """Return empty state when no hands detected."""
        return HandState(
            hand_present=False,
            num_hands=0,
            hand_region_v=HandRegion.MIDDLE,
            hand_region_h=HandRegion.CENTER,
            hand_y_normalized=0.5,
            hand_motion_energy=0.0,
            hand_write_score=0.0,
            is_writing_pattern=False,
            landmarks_left=None,
            landmarks_right=None
        )

    def reset(self) -> None:
        """Reset tracking state."""
        self._prev_landmarks = [None, None]
        self._motion_history.clear()
        self._position_history.clear()
        self._smooth_motion = 0.0
        self._smooth_write_score = 0.0

    def get_stats(self) -> Tuple[int, int]:
        """Get (detection_count, no_hand_count)."""
        return (self._detection_count, self._no_hand_count)

    def draw_hands(self, frame: np.ndarray, state: HandState) -> np.ndarray:
        """Draw hand landmarks and analysis on frame."""
        h, w = frame.shape[:2]

        # Draw landmarks
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        if state.landmarks_left is not None:
            self._draw_hand_landmarks(frame, state.landmarks_left, (255, 0, 0))

        if state.landmarks_right is not None:
            self._draw_hand_landmarks(frame, state.landmarks_right, (0, 0, 255))

        # Draw region lines
        lower_y = int(self.config.lower_region_threshold * h)
        upper_y = int(self.config.upper_region_threshold * h)

        cv2.line(frame, (0, lower_y), (w, lower_y), (100, 100, 100), 1)
        cv2.line(frame, (0, upper_y), (w, upper_y), (100, 100, 100), 1)

        # Add labels
        cv2.putText(frame, "Upper (phone)", (5, upper_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        cv2.putText(frame, "Lower (desk/writing)", (5, lower_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Draw stats
        stats_x = w - 200
        stats_y = 30

        color = (0, 255, 0) if state.hand_present else (0, 0, 255)
        cv2.putText(frame, f"Hands: {state.num_hands}", (stats_x, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if state.hand_present:
            stats_y += 20
            cv2.putText(frame, f"Region: {state.hand_region_v}", (stats_x, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            stats_y += 20
            cv2.putText(frame, f"Motion: {state.hand_motion_energy:.3f}", (stats_x, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            stats_y += 20
            write_color = (0, 255, 0) if state.is_writing_pattern else (255, 255, 255)
            cv2.putText(frame, f"Write Score: {state.hand_write_score:.3f}", (stats_x, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, write_color, 1)

            if state.is_writing_pattern:
                stats_y += 25
                cv2.putText(frame, "WRITING DETECTED", (stats_x - 20, stats_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def _draw_hand_landmarks(self, frame: np.ndarray, landmarks: np.ndarray,
                             color: Tuple[int, int, int]) -> None:
        """Draw hand landmarks on frame."""
        h, w = frame.shape[:2]

        # Convert normalized to pixel coordinates
        points = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks]

        # Draw points
        for i, pt in enumerate(points):
            radius = 4 if i in [self.WRIST, self.THUMB_TIP, self.INDEX_TIP,
                               self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP] else 2
            cv2.circle(frame, pt, radius, color, -1)

        # Draw connections
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm
            (5, 9), (9, 13), (13, 17)
        ]

        for start, end in connections:
            cv2.line(frame, points[start], points[end], color, 1)

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).replace("\\", "/").rsplit("/", 3)[0])

    from app.vision.camera import CameraCapture

    logging.basicConfig(level=logging.INFO)

    camera = CameraCapture()
    hand_analyzer = HandAnalyzer()

    if camera.start() and hand_analyzer.initialize():
        print("Hand analysis test running. Press 'q' to quit.")
        print("Try placing your hands in the lower part of the frame and making small writing motions.")

        while camera.is_running:
            frame = camera.get_processed_frame()
            if frame is not None:
                state = hand_analyzer.process(frame)
                frame = hand_analyzer.draw_hands(frame, state)

                cv2.imshow("Hand Analysis Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        hand_analyzer.release()
        camera.stop()
        cv2.destroyAllWindows()
    else:
        print("Failed to initialize")
