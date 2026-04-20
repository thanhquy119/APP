"""
Vision Pipeline - Unified vision processing for FocusGuardian.

Combines FaceLandmarker and HandLandmarker with head pose estimation
and eye closure detection into a single pipeline.
"""

import time
import logging
import threading
from typing import Optional, Callable, NamedTuple
from dataclasses import dataclass, field

import numpy as np
import cv2

from .face_landmarker import (
    FaceLandmarker,
    FaceLandmarkResult,
    calculate_ear,
    get_eye_closure_from_blendshapes,
    get_eye_gaze_vertical_from_blendshapes,
)
from .hand_landmarker import HandLandmarker, HandLandmarkResult

logger = logging.getLogger(__name__)


@dataclass
class HeadPose:
    """Head pose estimation result."""
    pitch: float  # Up/down rotation (negative = looking down)
    yaw: float    # Left/right rotation (positive = looking right)
    roll: float   # Tilt rotation


@dataclass
class EyeMetrics:
    """Eye-related metrics."""
    left_ear: float   # Left eye aspect ratio
    right_ear: float  # Right eye aspect ratio
    avg_ear: float    # Average EAR
    left_closure: float   # From blendshapes (0-1)
    right_closure: float  # From blendshapes (0-1)
    look_down: float  # Vertical gaze down score from blendshapes (0-1)
    look_up: float    # Vertical gaze up score from blendshapes (0-1)
    is_closed: bool   # True if eyes appear closed
    blink_detected: bool


@dataclass
class HandMetrics:
    """Hand-related metrics."""
    detected: bool
    num_hands: int
    region: str  # "upper", "middle", "lower", "none"
    write_score: float  # 0-1, higher = likely writing
    dominant_hand: str  # "Left", "Right", "Unknown"


@dataclass
class VisionResult:
    """
    Unified vision result combining all detections.

    This is the main output of the VisionPipeline.
    """
    timestamp_ms: int

    # Face detection
    face_detected: bool
    face_landmarks: Optional[np.ndarray] = None  # (478, 3) normalized

    # Head pose
    head_pose: Optional[HeadPose] = None

    # Eye metrics
    eye_metrics: Optional[EyeMetrics] = None

    # Hand detection
    hand_metrics: Optional[HandMetrics] = None

    # Raw results for advanced use
    face_result: Optional[FaceLandmarkResult] = None
    hand_result: Optional[HandLandmarkResult] = None


class VisionPipeline:
    """
    Unified vision processing pipeline.

    Combines:
    - Face detection with 478 landmarks
    - Head pose estimation via solvePnP
    - Eye closure detection via EAR and blendshapes
    - Hand detection with write score

    Supports both synchronous (VIDEO) and asynchronous (LIVE_STREAM) modes.
    """

    # 3D model points for head pose estimation (canonical face model)
    # These are approximate positions in a normalized coordinate system
    MODEL_POINTS = np.array([
        [0.0, 0.0, 0.0],          # Nose tip
        [-0.225, 0.170, -0.115],  # Left eye outer corner
        [0.225, 0.170, -0.115],   # Right eye outer corner
        [-0.075, -0.085, -0.115], # Mouth left corner
        [0.075, -0.085, -0.115],  # Mouth right corner
        [0.0, -0.180, -0.086],    # Chin
    ], dtype=np.float64)

    # Corresponding landmark indices
    POSE_LANDMARK_INDICES = [1, 33, 263, 61, 291, 199]

    # Eye landmark indices for EAR calculation
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    def __init__(
        self,
        use_live_stream: bool = True,
        result_callback: Optional[Callable[[VisionResult], None]] = None,
    ):
        """
        Initialize VisionPipeline.

        Args:
            use_live_stream: Use LIVE_STREAM mode for async processing
            result_callback: Callback for results (LIVE_STREAM mode)
        """
        self.use_live_stream = use_live_stream
        self.result_callback = result_callback

        self._face_landmarker: Optional[FaceLandmarker] = None
        self._hand_landmarker: Optional[HandLandmarker] = None

        self._latest_result: Optional[VisionResult] = None
        self._result_lock = threading.Lock()

        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        self._initialized = False

        # Blink detection state
        self._prev_ear = 0.3
        self._blink_threshold = 0.21
        self._blink_state = False

    def set_blink_threshold(self, threshold: float) -> None:
        """Update EAR threshold used by lightweight blink detection in this pipeline."""
        try:
            self._blink_threshold = max(0.12, min(0.35, float(threshold)))
        except (TypeError, ValueError):
            return

    def initialize(self, frame_width: int = 640, frame_height: int = 480) -> bool:
        """
        Initialize the pipeline.

        Args:
            frame_width: Camera frame width
            frame_height: Camera frame height

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        # Set up camera matrix for head pose estimation
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        self._camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        # Initialize face landmarker
        self._face_landmarker = FaceLandmarker(
            num_faces=1,
            use_live_stream=self.use_live_stream,
            result_callback=self._on_face_result if self.use_live_stream else None,
        )
        if not self._face_landmarker.initialize():
            logger.error("Failed to initialize FaceLandmarker")
            return False

        # Initialize hand landmarker
        self._hand_landmarker = HandLandmarker(
            num_hands=2,
            use_live_stream=self.use_live_stream,
            result_callback=self._on_hand_result if self.use_live_stream else None,
        )
        if not self._hand_landmarker.initialize():
            logger.error("Failed to initialize HandLandmarker")
            return False

        self._initialized = True
        self._frame_size = (frame_width, frame_height)
        logger.info(f"VisionPipeline initialized ({frame_width}x{frame_height})")
        return True

    def _on_face_result(self, result: FaceLandmarkResult):
        """Callback for face detection results."""
        # Results are combined in process()
        pass

    def _on_hand_result(self, result: HandLandmarkResult):
        """Callback for hand detection results."""
        # Results are combined in process()
        pass

    def _estimate_head_pose(
        self,
        landmarks: np.ndarray,
        frame_width: int,
        frame_height: int
    ) -> Optional[HeadPose]:
        """
        Estimate head pose using solvePnP.

        Args:
            landmarks: (478, 3) normalized face landmarks
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            HeadPose or None if estimation fails
        """
        try:
            # Get 2D image points from landmarks
            image_points = np.array([
                landmarks[idx][:2] * [frame_width, frame_height]
                for idx in self.POSE_LANDMARK_INDICES
            ], dtype=np.float64)

            # Solve PnP
            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.MODEL_POINTS,
                image_points,
                self._camera_matrix,
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return None

            # Convert rotation vector to Euler angles
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = cv2.hconcat([rotation_mat, translation_vec])
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

            pitch = self._normalize_pitch(float(euler_angles[0, 0]))
            yaw = self._normalize_signed_angle(float(euler_angles[1, 0]))
            roll = self._normalize_pitch(float(euler_angles[2, 0]))

            return HeadPose(pitch=pitch, yaw=yaw, roll=roll)

        except Exception as e:
            logger.debug(f"Head pose estimation error: {e}")
            return None

    @staticmethod
    def _normalize_signed_angle(angle: float) -> float:
        """Normalize angle into [-180, 180)."""
        return ((angle + 180.0) % 360.0) - 180.0

    @classmethod
    def _normalize_pitch(cls, angle: float) -> float:
        """Normalize pitch/roll into [-90, 90] to avoid 180-degree flips."""
        angle = cls._normalize_signed_angle(angle)
        if angle > 90.0:
            angle = 180.0 - angle
        elif angle < -90.0:
            angle = -180.0 - angle
        return angle

    def _calculate_eye_metrics(
        self,
        landmarks: np.ndarray,
        blendshapes: Optional[dict]
    ) -> EyeMetrics:
        """Calculate eye metrics from landmarks and blendshapes."""
        # Calculate EAR from landmarks
        left_ear = calculate_ear(landmarks, self.LEFT_EYE_INDICES)
        right_ear = calculate_ear(landmarks, self.RIGHT_EYE_INDICES)
        avg_ear = (left_ear + right_ear) / 2

        # Get closure from blendshapes if available
        left_closure, right_closure = 0.0, 0.0
        look_down, look_up = 0.0, 0.0
        if blendshapes:
            left_closure, right_closure = get_eye_closure_from_blendshapes(blendshapes)
            look_down, look_up = get_eye_gaze_vertical_from_blendshapes(blendshapes)

        # Determine if eyes are closed
        # Use blendshapes if available (more accurate), otherwise EAR
        if blendshapes and (left_closure > 0 or right_closure > 0):
            is_closed = (left_closure + right_closure) / 2 > 0.5
        else:
            is_closed = avg_ear < self._blink_threshold

        # Detect blink (transition from open to closed)
        blink_detected = False
        if avg_ear < self._blink_threshold and self._prev_ear >= self._blink_threshold:
            blink_detected = True
        self._prev_ear = avg_ear

        return EyeMetrics(
            left_ear=left_ear,
            right_ear=right_ear,
            avg_ear=avg_ear,
            left_closure=left_closure,
            right_closure=right_closure,
            look_down=look_down,
            look_up=look_up,
            is_closed=is_closed,
            blink_detected=blink_detected,
        )

    def _calculate_hand_metrics(
        self,
        hand_result: Optional[HandLandmarkResult]
    ) -> HandMetrics:
        """Calculate hand metrics from detection result."""
        if hand_result is None or not hand_result.hand_detected:
            return HandMetrics(
                detected=False,
                num_hands=0,
                region="none",
                write_score=0.0,
                dominant_hand="Unknown",
            )

        hand = hand_result.get_dominant_hand()
        write_score = self._hand_landmarker.calculate_write_score(hand_result)

        return HandMetrics(
            detected=True,
            num_hands=hand_result.num_hands,
            region=hand.region if hand else "none",
            write_score=write_score,
            dominant_hand=hand.handedness if hand else "Unknown",
        )

    def process(
        self,
        frame: np.ndarray,
        timestamp_ms: Optional[int] = None
    ) -> VisionResult:
        """
        Process a frame through the vision pipeline.

        Args:
            frame: BGR image from OpenCV
            timestamp_ms: Frame timestamp in milliseconds

        Returns:
            VisionResult with all detection results
        """
        if not self._initialized:
            if not self.initialize(frame.shape[1], frame.shape[0]):
                return VisionResult(
                    timestamp_ms=timestamp_ms or int(time.time() * 1000),
                    face_detected=False,
                )

        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        frame_height, frame_width = frame.shape[:2]

        # Process face
        face_result = self._face_landmarker.process(frame, timestamp_ms)

        # Process hands
        hand_result = self._hand_landmarker.process(frame, timestamp_ms)

        # Build vision result
        face_detected = face_result is not None and face_result.face_detected
        face_landmarks = None
        head_pose = None
        eye_metrics = None

        if face_detected and face_result.landmarks:
            face_landmarks = face_result.landmarks[0]

            # Estimate head pose
            head_pose = self._estimate_head_pose(
                face_landmarks, frame_width, frame_height
            )

            # Calculate eye metrics
            blendshapes = face_result.blendshapes[0] if face_result.blendshapes else None
            eye_metrics = self._calculate_eye_metrics(face_landmarks, blendshapes)

        # Calculate hand metrics
        hand_metrics = self._calculate_hand_metrics(hand_result)

        result = VisionResult(
            timestamp_ms=timestamp_ms,
            face_detected=face_detected,
            face_landmarks=face_landmarks,
            head_pose=head_pose,
            eye_metrics=eye_metrics,
            hand_metrics=hand_metrics,
            face_result=face_result,
            hand_result=hand_result,
        )

        with self._result_lock:
            self._latest_result = result

        if self.result_callback:
            self.result_callback(result)

        return result

    def get_latest_result(self) -> Optional[VisionResult]:
        """Get the latest result."""
        with self._result_lock:
            return self._latest_result

    def close(self):
        """Release resources."""
        if self._face_landmarker:
            self._face_landmarker.close()
            self._face_landmarker = None

        if self._hand_landmarker:
            self._hand_landmarker.close()
            self._hand_landmarker = None

        self._initialized = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def draw_vision_overlay(
    frame: np.ndarray,
    result: VisionResult,
    show_landmarks: bool = False
) -> np.ndarray:
    """
    Draw vision results overlay on frame.

    Args:
        frame: BGR image to draw on
        result: VisionResult to visualize
        show_landmarks: Whether to draw face landmarks

    Returns:
        Frame with overlay drawn
    """
    display = frame.copy()
    h, w = display.shape[:2]

    # Colors
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    WHITE = (255, 255, 255)

    y_offset = 30

    # Face status
    if result.face_detected:
        cv2.putText(display, "Face: Detected", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)

        # Draw face landmarks if requested
        if show_landmarks and result.face_landmarks is not None:
            for lm in result.face_landmarks:
                x = int(lm[0] * w)
                y = int(lm[1] * h)
                cv2.circle(display, (x, y), 1, GREEN, -1)
    else:
        cv2.putText(display, "Face: Not detected", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)

    y_offset += 25

    # Head pose
    if result.head_pose:
        hp = result.head_pose
        cv2.putText(display, f"Head: P={hp.pitch:.0f} Y={hp.yaw:.0f} R={hp.roll:.0f}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        y_offset += 20

    # Eye metrics
    if result.eye_metrics:
        em = result.eye_metrics
        status = "Closed" if em.is_closed else "Open"
        color = RED if em.is_closed else GREEN
        cv2.putText(display, f"Eyes: {status} (EAR={em.avg_ear:.2f})",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 20

    # Hand status
    if result.hand_metrics:
        hm = result.hand_metrics
        if hm.detected:
            cv2.putText(display, f"Hand: {hm.region} (write={hm.write_score:.2f})",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1)
        else:
            cv2.putText(display, "Hand: Not detected",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    return display
