"""
Blink detection using Eye Aspect Ratio (EAR) from Face Mesh landmarks.
Detects blinks, prolonged eye closure, and drowsiness indicators.
"""

import numpy as np
import cv2
import logging
from typing import Optional, Tuple, List, NamedTuple
from dataclasses import dataclass, field
from collections import deque
import time

from .face_mesh import FaceLandmarks, FaceMeshIndices

logger = logging.getLogger(__name__)


class BlinkState(NamedTuple):
    """Blink detection state for current frame."""
    ear_left: float           # Left eye aspect ratio
    ear_right: float          # Right eye aspect ratio
    ear_avg: float            # Average EAR
    is_eye_closed: bool       # Eyes currently closed
    blink_detected: bool      # New blink detected this frame
    blink_count: int          # Total blinks in window
    blink_rate: float         # Blinks per minute
    eye_closure_ratio: float  # % time eyes closed in window
    perclos: float            # PERCLOS (% eyes closed > 80% of the time)
    is_drowsy: bool           # Drowsiness indicator


@dataclass
class BlinkConfig:
    """Configuration for blink detection."""
    # EAR threshold for considering eyes closed
    ear_threshold: float = 0.21

    # Minimum consecutive frames for a blink
    min_blink_frames: int = 2

    # Maximum consecutive frames for a blink (longer = prolonged closure)
    max_blink_frames: int = 8

    # Window size for statistics (seconds)
    window_seconds: float = 60.0

    # Smoothing factor for EAR
    smoothing_factor: float = 0.5

    # Drowsiness thresholds
    drowsy_ear_threshold: float = 0.18  # Lower EAR = more closed
    drowsy_closure_ratio: float = 0.3   # 30% eyes closed
    perclos_threshold: float = 0.15     # PERCLOS > 15% indicates drowsiness

    # Blink rate thresholds (normal: 15-20 blinks/min)
    low_blink_rate: float = 8.0         # May indicate staring/fatigue
    high_blink_rate: float = 30.0       # May indicate discomfort


@dataclass
class BlinkEvent:
    """Record of a blink event."""
    timestamp: float
    duration_frames: int
    min_ear: float


class BlinkDetector:
    """
    Detects blinks and eye closure using Eye Aspect Ratio (EAR).

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    where p1-p6 are the 6 eye landmarks in order:
    p1: outer corner, p2: upper-1, p3: upper-2, p4: inner corner, p5: lower-2, p6: lower-1

    EAR approaches 0 when eye is closed and ~0.3 when fully open.
    """

    def __init__(self, config: Optional[BlinkConfig] = None):
        self.config = config or BlinkConfig()

        # Current state
        self._current_ear_left = 0.0
        self._current_ear_right = 0.0
        self._smooth_ear_left = 0.25  # Initial value (open eyes)
        self._smooth_ear_right = 0.25

        # Blink detection state machine
        self._eye_closed_frames = 0
        self._in_blink = False
        self._blink_start_ear = 0.0

        # Historical data
        self._blink_events: deque = deque(maxlen=100)  # Recent blinks
        self._ear_history: deque = deque(maxlen=1800)  # EAR values (60s at 30fps)
        self._closure_history: deque = deque(maxlen=1800)  # Binary: closed or not

        # Statistics
        self._total_blinks = 0
        self._last_blink_time = 0.0
        self._start_time = time.time()

    def calculate_ear(self, landmarks: FaceLandmarks, eye_indices: List[int]) -> float:
        """
        Calculate Eye Aspect Ratio for one eye.

        Args:
            landmarks: Face landmarks
            eye_indices: 6 landmark indices for the eye

        Returns:
            EAR value (0-0.4 typically)
        """
        # Get eye landmark positions
        points = landmarks.get_landmarks_array(eye_indices, pixel=True)

        # For MediaPipe, the order is different:
        # Left eye [362, 385, 387, 263, 373, 380]:
        #   362: inner corner, 385: upper-1, 387: upper-2
        #   263: outer corner, 373: lower-2, 380: lower-1
        # Right eye [33, 160, 158, 133, 153, 144]:
        #   33: outer corner, 160: upper-1, 158: upper-2
        #   133: inner corner, 153: lower-2, 144: lower-1

        # Reorder to standard format: outer, upper-1, upper-2, inner, lower-2, lower-1
        if eye_indices == FaceMeshIndices.LEFT_EYE:
            # Left eye: 362(inner), 385(up1), 387(up2), 263(outer), 373(low2), 380(low1)
            # Reorder to: outer, up1, up2, inner, low2, low1
            p1 = points[3]  # outer (263)
            p2 = points[1]  # upper-1 (385)
            p3 = points[2]  # upper-2 (387)
            p4 = points[0]  # inner (362)
            p5 = points[4]  # lower-2 (373)
            p6 = points[5]  # lower-1 (380)
        else:
            # Right eye: 33(outer), 160(up1), 158(up2), 133(inner), 153(low2), 144(low1)
            p1 = points[0]  # outer (33)
            p2 = points[1]  # upper-1 (160)
            p3 = points[2]  # upper-2 (158)
            p4 = points[3]  # inner (133)
            p5 = points[4]  # lower-2 (153)
            p6 = points[5]  # lower-1 (144)

        # Calculate vertical distances
        vertical_1 = np.linalg.norm(p2[:2] - p6[:2])
        vertical_2 = np.linalg.norm(p3[:2] - p5[:2])

        # Calculate horizontal distance
        horizontal = np.linalg.norm(p1[:2] - p4[:2])

        # Compute EAR
        if horizontal < 1e-6:
            return 0.0

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

        return float(ear)

    def process(self, landmarks: FaceLandmarks) -> BlinkState:
        """
        Process landmarks and detect blink state.

        Args:
            landmarks: Face landmarks from FaceMeshDetector

        Returns:
            BlinkState with current EAR values and blink statistics
        """
        current_time = time.time()

        # Calculate EAR for both eyes
        ear_left = self.calculate_ear(landmarks, FaceMeshIndices.LEFT_EYE)
        ear_right = self.calculate_ear(landmarks, FaceMeshIndices.RIGHT_EYE)

        self._current_ear_left = ear_left
        self._current_ear_right = ear_right

        # Apply smoothing
        alpha = self.config.smoothing_factor
        self._smooth_ear_left = alpha * ear_left + (1 - alpha) * self._smooth_ear_left
        self._smooth_ear_right = alpha * ear_right + (1 - alpha) * self._smooth_ear_right

        ear_avg = (self._smooth_ear_left + self._smooth_ear_right) / 2.0

        # Detect if eyes are closed
        is_eye_closed = ear_avg < self.config.ear_threshold

        # Store in history
        self._ear_history.append((current_time, ear_avg))
        self._closure_history.append((current_time, is_eye_closed))

        # Blink detection state machine
        blink_detected = False

        if is_eye_closed:
            if not self._in_blink:
                # Start of potential blink
                self._in_blink = True
                self._eye_closed_frames = 1
                self._blink_start_ear = ear_avg
            else:
                self._eye_closed_frames += 1
        else:
            if self._in_blink:
                # End of blink - check if it was valid
                if (self.config.min_blink_frames <= self._eye_closed_frames <=
                    self.config.max_blink_frames):
                    # Valid blink detected
                    blink_detected = True
                    self._total_blinks += 1
                    self._last_blink_time = current_time

                    # Record blink event
                    self._blink_events.append(BlinkEvent(
                        timestamp=current_time,
                        duration_frames=self._eye_closed_frames,
                        min_ear=self._blink_start_ear
                    ))

                self._in_blink = False
                self._eye_closed_frames = 0

        # Calculate statistics over window
        window_start = current_time - self.config.window_seconds

        # Blink count and rate
        recent_blinks = [b for b in self._blink_events if b.timestamp > window_start]
        blink_count = len(recent_blinks)
        blink_rate = blink_count * (60.0 / self.config.window_seconds)

        # Eye closure ratio
        recent_closures = [(t, c) for t, c in self._closure_history if t > window_start]
        if recent_closures:
            eye_closure_ratio = sum(1 for _, c in recent_closures if c) / len(recent_closures)
        else:
            eye_closure_ratio = 0.0

        # PERCLOS calculation (% of time eyes are ≥80% closed)
        # We approximate this by checking if EAR is very low
        recent_ears = [(t, e) for t, e in self._ear_history if t > window_start]
        if recent_ears:
            perclos = sum(1 for _, e in recent_ears if e < self.config.drowsy_ear_threshold) / len(recent_ears)
        else:
            perclos = 0.0

        # Drowsiness detection
        is_drowsy = (
            perclos > self.config.perclos_threshold or
            eye_closure_ratio > self.config.drowsy_closure_ratio or
            (ear_avg < self.config.drowsy_ear_threshold and self._eye_closed_frames > self.config.max_blink_frames)
        )

        return BlinkState(
            ear_left=self._smooth_ear_left,
            ear_right=self._smooth_ear_right,
            ear_avg=ear_avg,
            is_eye_closed=is_eye_closed,
            blink_detected=blink_detected,
            blink_count=blink_count,
            blink_rate=blink_rate,
            eye_closure_ratio=eye_closure_ratio,
            perclos=perclos,
            is_drowsy=is_drowsy
        )

    def reset(self) -> None:
        """Reset all state and history."""
        self._smooth_ear_left = 0.25
        self._smooth_ear_right = 0.25
        self._eye_closed_frames = 0
        self._in_blink = False
        self._blink_events.clear()
        self._ear_history.clear()
        self._closure_history.clear()
        self._total_blinks = 0
        self._start_time = time.time()

    def get_total_blinks(self) -> int:
        """Get total blinks since start/reset."""
        return self._total_blinks

    def get_average_ear(self) -> float:
        """Get average EAR over the history window."""
        if not self._ear_history:
            return 0.25
        return np.mean([e for _, e in self._ear_history])

    def calibrate_threshold(self, open_ear: float) -> None:
        """
        Calibrate threshold based on measured open eye EAR.
        Sets threshold to 70% of open eye EAR.
        """
        self.config.ear_threshold = open_ear * 0.7
        self.config.drowsy_ear_threshold = open_ear * 0.55
        logger.info(f"Calibrated EAR thresholds: closed={self.config.ear_threshold:.3f}, "
                   f"drowsy={self.config.drowsy_ear_threshold:.3f}")

    def draw_eyes(self, frame: np.ndarray, landmarks: FaceLandmarks,
                  state: BlinkState) -> np.ndarray:
        """Draw eye landmarks and EAR values on frame."""
        # Draw left eye
        for idx in FaceMeshIndices.LEFT_EYE:
            x, y = landmarks.get_pixel_coords(idx)
            color = (0, 0, 255) if state.is_eye_closed else (0, 255, 0)
            cv2.circle(frame, (x, y), 2, color, -1)

        # Draw right eye
        for idx in FaceMeshIndices.RIGHT_EYE:
            x, y = landmarks.get_pixel_coords(idx)
            color = (0, 0, 255) if state.is_eye_closed else (0, 255, 0)
            cv2.circle(frame, (x, y), 2, color, -1)

        # Draw EAR values
        h = frame.shape[0]
        y_offset = h - 120

        cv2.putText(frame, f"EAR L: {state.ear_left:.3f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"EAR R: {state.ear_right:.3f}", (10, y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Blinks: {state.blink_count} ({state.blink_rate:.1f}/min)",
                   (10, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Closure: {state.eye_closure_ratio*100:.1f}%",
                   (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"PERCLOS: {state.perclos*100:.1f}%",
                   (10, y_offset + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Drowsy indicator
        if state.is_drowsy:
            cv2.putText(frame, "DROWSY!", (frame.shape[1] - 120, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Blink indicator
        if state.blink_detected:
            cv2.putText(frame, "BLINK", (frame.shape[1]//2 - 40, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        return frame


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).replace("\\", "/").rsplit("/", 3)[0])

    from app.vision.camera import CameraCapture
    from app.vision.face_mesh import FaceMeshDetector

    logging.basicConfig(level=logging.INFO)

    camera = CameraCapture()
    face_mesh = FaceMeshDetector()
    blink_detector = BlinkDetector()

    if camera.start() and face_mesh.initialize():
        print("Blink detection test running. Press 'q' to quit.")

        while camera.is_running:
            frame = camera.get_processed_frame()
            if frame is not None:
                landmarks = face_mesh.process(frame)

                if landmarks is not None:
                    state = blink_detector.process(landmarks)
                    frame = blink_detector.draw_eyes(frame, landmarks, state)
                else:
                    cv2.putText(frame, "No face detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.imshow("Blink Detection Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        face_mesh.release()
        camera.stop()
        cv2.destroyAllWindows()
    else:
        print("Failed to initialize")
