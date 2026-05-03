"""
Vision Pipeline - Unified vision processing for FocusGuardian.

Combines FaceLandmarker and HandLandmarker with head pose estimation
and temporal behavior signals into a single pipeline.
"""

import time
import logging
import threading
from collections import deque
from typing import Optional, Callable, Tuple, Dict, Any
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
class VisionCalibration:
    """Per-profile calibration bundle for neutral head pose and eye baselines."""

    head_pitch_offset: float = 0.0
    head_yaw_offset: float = 0.0
    head_roll_offset: float = 0.0
    ear_open_baseline: float = 0.0
    ear_closed_threshold_adaptive: float = 0.21
    eye_closure_baseline: float = 0.0
    calibrated_at_ms: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "head_pitch_offset": float(self.head_pitch_offset),
            "head_yaw_offset": float(self.head_yaw_offset),
            "head_roll_offset": float(self.head_roll_offset),
            "ear_open_baseline": float(self.ear_open_baseline),
            "ear_closed_threshold_adaptive": float(self.ear_closed_threshold_adaptive),
            "eye_closure_baseline": float(self.eye_closure_baseline),
            "calibrated_at_ms": int(self.calibrated_at_ms),
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "VisionCalibration":
        data = payload or {}
        return cls(
            head_pitch_offset=float(data.get("head_pitch_offset", 0.0) or 0.0),
            head_yaw_offset=float(data.get("head_yaw_offset", 0.0) or 0.0),
            head_roll_offset=float(data.get("head_roll_offset", 0.0) or 0.0),
            ear_open_baseline=float(data.get("ear_open_baseline", 0.0) or 0.0),
            ear_closed_threshold_adaptive=float(data.get("ear_closed_threshold_adaptive", 0.21) or 0.21),
            eye_closure_baseline=float(data.get("eye_closure_baseline", 0.0) or 0.0),
            calibrated_at_ms=int(float(data.get("calibrated_at_ms", 0) or 0)),
        )


@dataclass
class CalibrationResult:
    """Result emitted when calibration session finishes."""

    success: bool
    duration_seconds: float
    sample_count: int
    message: str
    calibration: VisionCalibration

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "success": bool(self.success),
            "duration_seconds": float(self.duration_seconds),
            "sample_count": int(self.sample_count),
            "message": str(self.message),
        }
        payload.update(self.calibration.to_dict())
        return payload


@dataclass
class EyeTemporalStats:
    """Windowed eye behavior statistics used for fatigue and blink analytics."""

    window_seconds: float
    sample_count: int
    blink_count_window: int
    blink_rate_per_min: float
    eye_closure_ratio: float
    perclos_ratio: float
    avg_ear_window: float
    avg_eye_closure_level: float


@dataclass
class VisionQuality:
    """Quality and confidence report for one processed frame."""

    frame_brightness: float = 0.0
    lighting_quality: str = "Unknown"
    blur_score: float = 0.0
    face_visibility: float = 0.0
    face_tracking_confidence: float = 0.0
    head_pose_confidence: float = 0.0
    eye_confidence: float = 0.0
    hand_confidence: float = 0.0
    overall_confidence: float = 0.0
    quality_warnings: list[str] = field(default_factory=list)


@dataclass
class HeadPose:
    """Head pose estimation result."""
    pitch: float  # Up/down rotation (negative = looking down)
    yaw: float    # Left/right rotation (positive = looking right)
    roll: float   # Tilt rotation
    confidence: float = 0.0
    reprojection_error: Optional[float] = None
    source: str = "solvepnp"


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
    blink_rate_per_min: float = 0.0
    eye_closure_ratio: float = 0.0
    perclos_ratio: float = 0.0
    avg_ear_window: float = 0.0
    avg_eye_closure_level: float = 0.0
    blink_count_window: int = 0
    eye_confidence: float = 0.0
    temporal: Optional[EyeTemporalStats] = None


@dataclass
class HandMetrics:
    """Hand-related metrics."""
    detected: bool
    num_hands: int
    region: str  # "upper", "middle", "lower", "none"
    write_score: float  # 0-1, higher = likely writing
    writing_confidence: float = 0.0
    motion_energy: float = 0.0
    motion_stability: float = 0.0
    lower_region_ratio: float = 0.0
    dominant_hand: str = "Unknown"  # "Left", "Right", "Unknown"


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

    # Optional phone evidence (can be attached by caller if detector is external)
    phone_present: bool = False
    phone_confidence: float = 0.0
    phone_bbox: Optional[Tuple[int, int, int, int]] = None
    phone_evidence_source: str = "unknown"

    # Data quality and confidence layer
    quality: VisionQuality = field(default_factory=VisionQuality)

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
        eye_window_seconds: float = 60.0,
        hand_window_seconds: float = 3.0,
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
        self._frame_size: Tuple[int, int] = (640, 480)

        self._initialized = False

        # Eye state
        self._prev_ear = 0.3
        self._blink_threshold = 0.21
        self._blink_closed_streak = 0
        self._eye_window_seconds = max(15.0, float(eye_window_seconds))
        self._eye_history: deque[Dict[str, Any]] = deque(maxlen=4800)

        # Hand temporal tracking
        self._hand_window_seconds = max(1.5, float(hand_window_seconds))
        self._hand_history: deque[Dict[str, Any]] = deque(maxlen=2400)
        self._head_pitch_history: deque[Tuple[float, float]] = deque(maxlen=2400)

        # Head pose tracking and smoothing
        self._prev_rvec: Optional[np.ndarray] = None
        self._prev_tvec: Optional[np.ndarray] = None
        self._smooth_pitch: Optional[float] = None
        self._smooth_yaw: Optional[float] = None
        self._smooth_roll: Optional[float] = None
        self._head_pose_alpha = 0.36
        self._latest_raw_pose: Optional[Tuple[float, float, float]] = None

        # Quality tracking
        self._face_presence_history: deque[Tuple[float, bool]] = deque(maxlen=300)

        # Calibration
        self._calibration = VisionCalibration()
        self._calibration_active = False
        self._calibration_duration_seconds = 3.0
        self._calibration_started_at = 0.0
        self._calibration_pose_samples: list[Tuple[float, float, float]] = []
        self._calibration_ear_samples: list[float] = []
        self._calibration_closure_samples: list[float] = []
        self._latest_calibration_result: Optional[CalibrationResult] = None

        # Startup warmup tracking
        self._pipeline_started_at: float = time.time()
        self._warmup_seconds: float = 25.0  # grace period for temporal stats

        # Face-loss tracking for selective smooth-pose reset
        self._face_lost_at: Optional[float] = None
        self._face_reset_threshold_seconds: float = 0.55  # only reset after this long without face

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @property
    def is_calibrating(self) -> bool:
        return self._calibration_active

    @property
    def calibration(self) -> VisionCalibration:
        return self._calibration

    def set_blink_threshold(self, threshold: float) -> None:
        """Update EAR threshold used by lightweight blink detection in this pipeline."""
        try:
            self._blink_threshold = max(0.12, min(0.35, float(threshold)))
            self._calibration.ear_closed_threshold_adaptive = self._blink_threshold
        except (TypeError, ValueError):
            return

    def start_calibration(self, duration_seconds: float = 3.0) -> None:
        """Start a short neutral calibration session."""
        self._calibration_duration_seconds = max(1.0, min(8.0, float(duration_seconds)))
        self._calibration_started_at = time.time()
        self._calibration_pose_samples.clear()
        self._calibration_ear_samples.clear()
        self._calibration_closure_samples.clear()
        self._latest_calibration_result = None
        self._calibration_active = True

    def cancel_calibration(self) -> None:
        self._calibration_active = False
        self._calibration_pose_samples.clear()
        self._calibration_ear_samples.clear()
        self._calibration_closure_samples.clear()

    def get_calibration_progress(self) -> float:
        if not self._calibration_active:
            return 0.0
        elapsed = max(0.0, time.time() - self._calibration_started_at)
        return self._clamp(elapsed / max(0.1, self._calibration_duration_seconds), 0.0, 1.0)

    def consume_latest_calibration_result(self) -> Optional[CalibrationResult]:
        result = self._latest_calibration_result
        self._latest_calibration_result = None
        return result

    def apply_calibration(self, payload: Optional[Dict[str, Any]]) -> None:
        self._calibration = VisionCalibration.from_dict(payload)
        self._blink_threshold = self._clamp(
            self._calibration.ear_closed_threshold_adaptive,
            0.12,
            0.34,
        )

    def export_calibration(self) -> Dict[str, Any]:
        return self._calibration.to_dict()

    def _finalize_calibration(self) -> None:
        self._calibration_active = False

        pose_samples = np.array(self._calibration_pose_samples, dtype=np.float64) if self._calibration_pose_samples else np.empty((0, 3), dtype=np.float64)
        ear_samples = np.array(self._calibration_ear_samples, dtype=np.float64) if self._calibration_ear_samples else np.empty((0,), dtype=np.float64)
        closure_samples = np.array(self._calibration_closure_samples, dtype=np.float64) if self._calibration_closure_samples else np.empty((0,), dtype=np.float64)

        sample_count = int(len(pose_samples))
        if sample_count < 15:
            self._latest_calibration_result = CalibrationResult(
                success=False,
                duration_seconds=float(self._calibration_duration_seconds),
                sample_count=sample_count,
                message="Calibration failed: insufficient stable samples",
                calibration=self._calibration,
            )
            return

        pitch_offset, yaw_offset, roll_offset = np.median(pose_samples, axis=0).tolist()
        ear_open = float(np.median(ear_samples)) if len(ear_samples) > 0 else max(0.22, self._blink_threshold / 0.7)
        closure_open = float(np.median(closure_samples)) if len(closure_samples) > 0 else 0.0

        adaptive_closed = self._clamp(ear_open * 0.7, 0.12, 0.33)

        self._calibration = VisionCalibration(
            head_pitch_offset=float(pitch_offset),
            head_yaw_offset=float(yaw_offset),
            head_roll_offset=float(roll_offset),
            ear_open_baseline=float(ear_open),
            ear_closed_threshold_adaptive=float(adaptive_closed),
            eye_closure_baseline=float(self._clamp(closure_open, 0.0, 1.0)),
            calibrated_at_ms=int(time.time() * 1000),
        )
        self._blink_threshold = adaptive_closed

        self._latest_calibration_result = CalibrationResult(
            success=True,
            duration_seconds=float(self._calibration_duration_seconds),
            sample_count=sample_count,
            message="Calibration complete",
            calibration=self._calibration,
        )

    def _record_calibration_sample(
        self,
        head_pose: Optional[HeadPose],
        eye_metrics: Optional[EyeMetrics],
        quality: VisionQuality,
    ) -> None:
        if not self._calibration_active:
            return

        if quality.overall_confidence < 0.35:
            if self.get_calibration_progress() >= 1.0:
                self._finalize_calibration()
            return

        if self._latest_raw_pose is not None:
            self._calibration_pose_samples.append(self._latest_raw_pose)

        if eye_metrics is not None:
            if eye_metrics.avg_ear > 0.0 and not eye_metrics.is_closed:
                self._calibration_ear_samples.append(float(eye_metrics.avg_ear))
            self._calibration_closure_samples.append(float(eye_metrics.avg_eye_closure_level))

        if self.get_calibration_progress() >= 1.0:
            self._finalize_calibration()

    def get_quality_summary(self, quality: Optional[VisionQuality] = None) -> str:
        """Return compact UI-friendly camera quality summary."""
        candidate = quality
        if candidate is None:
            latest = self.get_latest_result()
            candidate = latest.quality if latest is not None else VisionQuality()

        if candidate.quality_warnings:
            warning_priority = [
                "Anh sang yeu",
                "Anh sang qua gat",
                "Khung hinh mo",
                "Mat bi lech khoi camera",
                "Chua hieu chinh",
                "Tracking khong on dinh",
            ]
            lower = [w.lower() for w in candidate.quality_warnings]
            for item in warning_priority:
                for idx, warning in enumerate(candidate.quality_warnings):
                    if item.lower() in lower[idx]:
                        return warning
            return candidate.quality_warnings[0]

        if candidate.overall_confidence >= 0.7:
            return "Camera on"
        if candidate.overall_confidence >= 0.45:
            return "Tracking tam chap nhan"
        return "Tracking khong on dinh"

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
        self._smooth_pitch = None
        self._smooth_yaw = None
        self._smooth_roll = None
        self._prev_rvec = None
        self._prev_tvec = None
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
        frame_height: int,
        transform_matrix: Optional[np.ndarray] = None,
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
            image_points = np.array(
                [
                    landmarks[idx][:2] * [frame_width, frame_height]
                    for idx in self.POSE_LANDMARK_INDICES
                ],
                dtype=np.float64,
            )

            use_guess = self._prev_rvec is not None and self._prev_tvec is not None
            if use_guess:
                success, rotation_vec, translation_vec = cv2.solvePnP(
                    self.MODEL_POINTS,
                    image_points,
                    self._camera_matrix,
                    self._dist_coeffs,
                    rvec=self._prev_rvec,
                    tvec=self._prev_tvec,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            else:
                success, rotation_vec, translation_vec = cv2.solvePnP(
                    self.MODEL_POINTS,
                    image_points,
                    self._camera_matrix,
                    self._dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )

            if not success:
                fallback = self._estimate_head_pose_from_matrix(transform_matrix)
                if fallback is None:
                    return None
                return fallback

            projected_points, _ = cv2.projectPoints(
                self.MODEL_POINTS,
                rotation_vec,
                translation_vec,
                self._camera_matrix,
                self._dist_coeffs,
            )
            projected_points = projected_points.reshape(-1, 2)
            reprojection_error = float(np.mean(np.linalg.norm(image_points - projected_points, axis=1)))

            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            raw_pitch, raw_yaw, raw_roll = self._rotation_matrix_to_euler_degrees(rotation_mat)
            self._latest_raw_pose = (raw_pitch, raw_yaw, raw_roll)

            calibrated_pitch = raw_pitch - self._calibration.head_pitch_offset
            calibrated_yaw = raw_yaw - self._calibration.head_yaw_offset
            calibrated_roll = raw_roll - self._calibration.head_roll_offset

            if self._smooth_pitch is None:
                self._smooth_pitch = calibrated_pitch
                self._smooth_yaw = calibrated_yaw
                self._smooth_roll = calibrated_roll
            else:
                alpha = self._head_pose_alpha
                self._smooth_pitch = (alpha * calibrated_pitch) + ((1.0 - alpha) * self._smooth_pitch)
                self._smooth_yaw = (alpha * calibrated_yaw) + ((1.0 - alpha) * self._smooth_yaw)
                self._smooth_roll = (alpha * calibrated_roll) + ((1.0 - alpha) * self._smooth_roll)

            self._prev_rvec = rotation_vec
            self._prev_tvec = translation_vec

            conf = self._clamp(1.0 - (reprojection_error / 18.0), 0.0, 1.0)

            return HeadPose(
                pitch=self._normalize_pitch(float(self._smooth_pitch)),
                yaw=self._normalize_signed_angle(float(self._smooth_yaw)),
                roll=self._normalize_pitch(float(self._smooth_roll)),
                confidence=conf,
                reprojection_error=reprojection_error,
                source="solvepnp",
            )

        except Exception as e:
            logger.debug(f"Head pose estimation error: {e}")
            return self._estimate_head_pose_from_matrix(transform_matrix)

    def _estimate_head_pose_from_matrix(self, transform_matrix: Optional[np.ndarray]) -> Optional[HeadPose]:
        """Fallback pose estimation from MediaPipe facial transformation matrix."""
        if transform_matrix is None:
            return None

        try:
            matrix = np.array(transform_matrix, dtype=np.float64)
            if matrix.size == 16:
                matrix = matrix.reshape((4, 4))
            if matrix.shape == (4, 4):
                rotation = matrix[:3, :3]
            elif matrix.shape == (3, 3):
                rotation = matrix
            else:
                return None

            raw_pitch, raw_yaw, raw_roll = self._rotation_matrix_to_euler_degrees(rotation)
            self._latest_raw_pose = (raw_pitch, raw_yaw, raw_roll)

            calibrated_pitch = raw_pitch - self._calibration.head_pitch_offset
            calibrated_yaw = raw_yaw - self._calibration.head_yaw_offset
            calibrated_roll = raw_roll - self._calibration.head_roll_offset

            if self._smooth_pitch is None:
                self._smooth_pitch = calibrated_pitch
                self._smooth_yaw = calibrated_yaw
                self._smooth_roll = calibrated_roll
            else:
                alpha = min(0.5, self._head_pose_alpha)
                self._smooth_pitch = (alpha * calibrated_pitch) + ((1.0 - alpha) * self._smooth_pitch)
                self._smooth_yaw = (alpha * calibrated_yaw) + ((1.0 - alpha) * self._smooth_yaw)
                self._smooth_roll = (alpha * calibrated_roll) + ((1.0 - alpha) * self._smooth_roll)

            return HeadPose(
                pitch=self._normalize_pitch(float(self._smooth_pitch)),
                yaw=self._normalize_signed_angle(float(self._smooth_yaw)),
                roll=self._normalize_pitch(float(self._smooth_roll)),
                confidence=0.46,
                reprojection_error=None,
                source="facial_transform",
            )
        except Exception:
            return None

    @classmethod
    def _rotation_matrix_to_euler_degrees(cls, rotation_mat: np.ndarray) -> Tuple[float, float, float]:
        pose_mat = np.hstack([rotation_mat, np.zeros((3, 1), dtype=np.float64)])
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        pitch = cls._normalize_pitch(float(euler_angles[0, 0]))
        yaw = cls._normalize_signed_angle(float(euler_angles[1, 0]))
        roll = cls._normalize_pitch(float(euler_angles[2, 0]))
        return pitch, yaw, roll

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
        blendshapes: Optional[dict],
        timestamp_s: float,
        quality_hint: float,
    ) -> EyeMetrics:
        """Calculate eye metrics from landmarks and blendshapes."""
        left_ear = calculate_ear(landmarks, self.LEFT_EYE_INDICES)
        right_ear = calculate_ear(landmarks, self.RIGHT_EYE_INDICES)
        avg_ear = (left_ear + right_ear) / 2

        left_closure, right_closure = 0.0, 0.0
        look_down, look_up = 0.0, 0.0
        if blendshapes:
            left_closure, right_closure = get_eye_closure_from_blendshapes(blendshapes)
            look_down, look_up = get_eye_gaze_vertical_from_blendshapes(blendshapes)

        closure_level = (float(left_closure) + float(right_closure)) / 2.0
        using_blendshape_closure = bool(blendshapes) and (left_closure > 0.0 or right_closure > 0.0)

        open_baseline = self._calibration.ear_open_baseline if self._calibration.ear_open_baseline > 0.0 else 0.30
        closed_threshold = self._clamp(
            self._calibration.ear_closed_threshold_adaptive,
            0.12,
            0.34,
        )

        if not using_blendshape_closure:
            denom = max(1e-3, open_baseline - closed_threshold)
            closure_level = self._clamp((open_baseline - avg_ear) / denom, 0.0, 1.0)

        closure_open = self._clamp(self._calibration.eye_closure_baseline, 0.0, 0.6)
        closure_close_threshold = self._clamp(max(0.40, closure_open + 0.22), 0.35, 0.82)
        is_closed = bool((closure_level >= closure_close_threshold) or (avg_ear < closed_threshold))

        blink_detected = False
        if is_closed:
            self._blink_closed_streak += 1
        else:
            if 2 <= self._blink_closed_streak <= 10:
                blink_detected = True
            self._blink_closed_streak = 0

        self._prev_ear = avg_ear

        temporal = self._update_eye_temporal_stats(
            timestamp_s=timestamp_s,
            avg_ear=avg_ear,
            closure_level=closure_level,
            is_closed=is_closed,
            blink_detected=blink_detected,
        )

        blendshape_score = 1.0 if using_blendshape_closure else 0.55
        elapsed_since_start = max(0.0, timestamp_s - self._pipeline_started_at)
        if elapsed_since_start < self._warmup_seconds:
            # During warmup, floor temporal_score so early frames aren't penalised
            warmup_ratio = elapsed_since_start / max(1.0, self._warmup_seconds)
            warmup_floor = self._clamp(0.62 - 0.20 * warmup_ratio, 0.42, 0.62)
            temporal_score = max(warmup_floor, self._clamp(temporal.sample_count / 45.0, 0.2, 1.0))
        else:
            temporal_score = self._clamp(temporal.sample_count / 45.0, 0.2, 1.0)
        eye_confidence = self._clamp(
            (0.40 * blendshape_score)
            + (0.30 * temporal_score)
            + (0.30 * self._clamp(quality_hint, 0.0, 1.0)),
            0.0,
            1.0,
        )

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
            blink_rate_per_min=temporal.blink_rate_per_min,
            eye_closure_ratio=temporal.eye_closure_ratio,
            perclos_ratio=temporal.perclos_ratio,
            avg_ear_window=temporal.avg_ear_window,
            avg_eye_closure_level=temporal.avg_eye_closure_level,
            blink_count_window=temporal.blink_count_window,
            eye_confidence=eye_confidence,
            temporal=temporal,
        )

    def _update_eye_temporal_stats(
        self,
        timestamp_s: float,
        avg_ear: float,
        closure_level: float,
        is_closed: bool,
        blink_detected: bool,
    ) -> EyeTemporalStats:
        self._eye_history.append(
            {
                "ts": float(timestamp_s),
                "ear": float(avg_ear),
                "closure": float(self._clamp(closure_level, 0.0, 1.0)),
                "closed": bool(is_closed),
                "blink": bool(blink_detected),
            }
        )

        window_start = float(timestamp_s) - self._eye_window_seconds
        while self._eye_history and self._eye_history[0]["ts"] < window_start:
            self._eye_history.popleft()

        samples = list(self._eye_history)
        count = len(samples)
        if count <= 0:
            return EyeTemporalStats(
                window_seconds=float(self._eye_window_seconds),
                sample_count=0,
                blink_count_window=0,
                blink_rate_per_min=0.0,
                eye_closure_ratio=0.0,
                perclos_ratio=0.0,
                avg_ear_window=0.0,
                avg_eye_closure_level=0.0,
            )

        blink_count = int(sum(1 for item in samples if item["blink"]))
        blink_rate = float(blink_count) * (60.0 / max(1e-6, self._eye_window_seconds))
        closure_ratio = float(sum(1 for item in samples if item["closed"])) / float(count)
        perclos_ratio = float(sum(1 for item in samples if item["closure"] >= 0.8)) / float(count)
        avg_ear_window = float(sum(item["ear"] for item in samples) / float(count))
        avg_closure_level = float(sum(item["closure"] for item in samples) / float(count))

        return EyeTemporalStats(
            window_seconds=float(self._eye_window_seconds),
            sample_count=count,
            blink_count_window=blink_count,
            blink_rate_per_min=blink_rate,
            eye_closure_ratio=closure_ratio,
            perclos_ratio=perclos_ratio,
            avg_ear_window=avg_ear_window,
            avg_eye_closure_level=avg_closure_level,
        )

    def _calculate_hand_metrics(
        self,
        hand_result: Optional[HandLandmarkResult],
        timestamp_s: float,
        head_pose: Optional[HeadPose],
    ) -> HandMetrics:
        """Calculate hand metrics from detection result."""
        if hand_result is None or not hand_result.hand_detected:
            self._hand_history.append(
                {
                    "ts": float(timestamp_s),
                    "detected": False,
                    "region": "none",
                    "write": 0.0,
                    "cx": None,
                    "cy": None,
                    "dominant": "Unknown",
                }
            )
            self._trim_hand_history(timestamp_s)
            lower_ratio = self._compute_hand_lower_ratio()
            return HandMetrics(
                detected=False,
                num_hands=0,
                region="none",
                write_score=0.0,
                writing_confidence=0.0,
                motion_energy=0.0,
                motion_stability=0.0,
                lower_region_ratio=lower_ratio,
                dominant_hand="Unknown",
            )

        hand = hand_result.get_dominant_hand()
        instant_write_score = self._hand_landmarker.calculate_write_score(hand_result)
        center_x = float(hand.center_x) if hand is not None else None
        center_y = float(hand.center_y) if hand is not None else None
        region = hand.region if hand else "none"
        dominant_hand = hand.handedness if hand else "Unknown"

        self._hand_history.append(
            {
                "ts": float(timestamp_s),
                "detected": True,
                "region": region,
                "write": float(instant_write_score),
                "cx": center_x,
                "cy": center_y,
                "dominant": dominant_hand,
            }
        )
        self._trim_hand_history(timestamp_s)

        motion_energy, motion_stability = self._compute_hand_motion_profile()
        lower_ratio = self._compute_hand_lower_ratio()
        glance_support = self._compute_recent_glance_support(timestamp_s, head_pose)
        stability_support = self._clamp(1.0 - (motion_energy / 0.08), 0.0, 1.0)

        temporal_write = self._clamp(
            (0.52 * lower_ratio)
            + (0.26 * motion_stability)
            + (0.22 * glance_support),
            0.0,
            1.0,
        )
        # Bias toward temporal when we have enough hand history (stable detection);
        # trust instant more only if history is short (< 5 frames).
        detected_ratio = self._compute_hand_detected_ratio()
        _history_len = len(self._hand_history)
        if _history_len >= 5:
            # Stable history: temporal outweighs single-frame score to avoid one-frame spurts
            write_score = self._clamp((0.42 * instant_write_score) + (0.58 * temporal_write), 0.0, 1.0)
        else:
            write_score = self._clamp((0.58 * instant_write_score) + (0.42 * temporal_write), 0.0, 1.0)
        writing_confidence = self._clamp(
            (0.34 * detected_ratio)
            + (0.24 * lower_ratio)
            + (0.22 * stability_support)
            + (0.20 * glance_support),
            0.0,
            1.0,
        )

        return HandMetrics(
            detected=True,
            num_hands=hand_result.num_hands,
            region=region,
            write_score=write_score,
            writing_confidence=writing_confidence,
            motion_energy=motion_energy,
            motion_stability=motion_stability,
            lower_region_ratio=lower_ratio,
            dominant_hand=dominant_hand,
        )

    def _trim_hand_history(self, timestamp_s: float) -> None:
        cutoff = float(timestamp_s) - self._hand_window_seconds
        while self._hand_history and self._hand_history[0]["ts"] < cutoff:
            self._hand_history.popleft()

    def _compute_hand_detected_ratio(self) -> float:
        if not self._hand_history:
            return 0.0
        detected = sum(1 for item in self._hand_history if item["detected"])
        return float(detected) / float(len(self._hand_history))

    def _compute_hand_lower_ratio(self) -> float:
        detected_items = [item for item in self._hand_history if item["detected"]]
        if not detected_items:
            return 0.0
        lower = sum(1 for item in detected_items if item["region"] == "lower")
        return float(lower) / float(len(detected_items))

    def _compute_hand_motion_profile(self) -> Tuple[float, float]:
        detected_items = [item for item in self._hand_history if item["detected"] and item["cx"] is not None and item["cy"] is not None]
        if len(detected_items) < 3:
            return 0.0, 0.0

        speeds: list[float] = []
        for prev, curr in zip(detected_items[:-1], detected_items[1:]):
            dt = max(1e-3, float(curr["ts"] - prev["ts"]))
            dx = float(curr["cx"] - prev["cx"])
            dy = float(curr["cy"] - prev["cy"])
            speed = float(np.hypot(dx, dy)) / dt
            speeds.append(speed)

        if not speeds:
            return 0.0, 0.0

        motion_energy = float(np.mean(speeds))
        motion_std = float(np.std(speeds))
        motion_stability = self._clamp(1.0 - (motion_std / 0.10), 0.0, 1.0)
        return motion_energy, motion_stability

    def _compute_recent_glance_support(self, timestamp_s: float, head_pose: Optional[HeadPose]) -> float:
        if head_pose is not None:
            self._head_pitch_history.append((float(timestamp_s), float(head_pose.pitch)))

        cutoff = float(timestamp_s) - 5.0
        while self._head_pitch_history and self._head_pitch_history[0][0] < cutoff:
            self._head_pitch_history.popleft()

        if len(self._head_pitch_history) < 3:
            return 0.0

        glances = 0
        was_down = False
        for _, pitch in self._head_pitch_history:
            if pitch <= -12.0:
                was_down = True
            elif was_down and pitch >= -5.0:
                glances += 1
                was_down = False

        return self._clamp(glances / 3.0, 0.0, 1.0)

    def _assess_quality(
        self,
        frame: np.ndarray,
        timestamp_s: float,
        face_detected: bool,
        face_landmarks: Optional[np.ndarray],
        head_pose: Optional[HeadPose],
        eye_metrics: Optional[EyeMetrics],
        hand_metrics: Optional[HandMetrics],
    ) -> VisionQuality:
        warnings: list[str] = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if brightness < 55.0:
            lighting_quality = "Low"
            warnings.append("Ánh sáng yếu")
        elif brightness > 205.0:
            lighting_quality = "Strong"
            warnings.append("Ánh sáng quá gắt")
        else:
            lighting_quality = "Tốt"

        if blur_score < 55.0:
            warnings.append("Khung hình mờ")

        face_visibility = 0.0
        if face_landmarks is not None and len(face_landmarks) > 0:
            xs = np.clip(face_landmarks[:, 0], 0.0, 1.0)
            ys = np.clip(face_landmarks[:, 1], 0.0, 1.0)
            width = max(0.0, float(np.max(xs) - np.min(xs)))
            height = max(0.0, float(np.max(ys) - np.min(ys)))
            area = width * height
            cx = float((np.max(xs) + np.min(xs)) * 0.5)
            cy = float((np.max(ys) + np.min(ys)) * 0.5)
            center_penalty = min(1.0, np.hypot(cx - 0.5, cy - 0.5) / 0.65)
            face_visibility = self._clamp((area / 0.22) * (1.0 - 0.35 * center_penalty), 0.0, 1.0)

        self._face_presence_history.append((float(timestamp_s), bool(face_detected)))
        face_cutoff = float(timestamp_s) - 5.0
        while self._face_presence_history and self._face_presence_history[0][0] < face_cutoff:
            self._face_presence_history.popleft()
        if self._face_presence_history:
            face_ratio = sum(1 for _, present in self._face_presence_history if present) / float(len(self._face_presence_history))
        else:
            face_ratio = 0.0

        face_tracking_confidence = self._clamp((0.65 * face_ratio) + (0.35 * face_visibility), 0.0, 1.0)

        if face_detected and face_visibility < 0.22:
            warnings.append("Mặt bị lệch khỏi camera")
        if not face_detected:
            warnings.append("Không thấy khuôn mặt ổn định")

        if head_pose is None:
            head_pose_confidence = 0.0
            if face_detected:
                warnings.append("Head pose không ổn định")
        else:
            head_pose_confidence = self._clamp(head_pose.confidence, 0.0, 1.0)
            if head_pose.reprojection_error is not None and head_pose.reprojection_error > 9.5:
                warnings.append("Head pose confidence thấp")

        eye_confidence = eye_metrics.eye_confidence if eye_metrics is not None else 0.0
        hand_confidence = hand_metrics.writing_confidence if hand_metrics is not None else 0.0

        brightness_conf = self._clamp(1.0 - abs(brightness - 128.0) / 128.0, 0.0, 1.0)
        blur_conf = self._clamp((blur_score - 25.0) / 110.0, 0.0, 1.0)
        imaging_conf = self._clamp((0.55 * brightness_conf) + (0.45 * blur_conf), 0.0, 1.0)

        overall_confidence = self._clamp(
            (0.30 * face_tracking_confidence)
            + (0.24 * head_pose_confidence)
            + (0.22 * eye_confidence)
            + (0.12 * imaging_conf)
            + (0.12 * hand_confidence),
            0.0,
            1.0,
        )

        if self._calibration.calibrated_at_ms <= 0:
            warnings.append("Chưa hiệu chỉnh")
            # Don't let missing calibration crater confidence when face+eye signal is strong.
            # Apply a soft floor so the app remains usable before first calibration.
            calibration_penalty_floor = self._clamp(
                0.35 + 0.25 * face_tracking_confidence + 0.15 * eye_confidence,
                0.35,
                0.62,
            )
            overall_confidence = max(overall_confidence, calibration_penalty_floor)
        if overall_confidence < 0.40:
            warnings.append("Tracking không ổn định")

        return VisionQuality(
            frame_brightness=brightness,
            lighting_quality=lighting_quality,
            blur_score=blur_score,
            face_visibility=face_visibility,
            face_tracking_confidence=face_tracking_confidence,
            head_pose_confidence=head_pose_confidence,
            eye_confidence=eye_confidence,
            hand_confidence=hand_confidence,
            overall_confidence=overall_confidence,
            quality_warnings=warnings,
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
                    quality=VisionQuality(
                        lighting_quality="Unknown",
                        quality_warnings=["Vision pipeline unavailable"],
                    ),
                )

        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        frame_height, frame_width = frame.shape[:2]
        timestamp_s = float(timestamp_ms) / 1000.0

        # Pre-compute lightweight imaging quality so it can inform eye metrics.
        try:
            _gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _brightness = float(np.mean(_gray))
            _blur = float(cv2.Laplacian(_gray, cv2.CV_64F).var())
            _brightness_conf = self._clamp(1.0 - abs(_brightness - 128.0) / 128.0, 0.0, 1.0)
            _blur_conf = self._clamp((_blur - 25.0) / 110.0, 0.0, 1.0)
            imaging_conf = self._clamp(0.55 * _brightness_conf + 0.45 * _blur_conf, 0.0, 1.0)
        except Exception:
            imaging_conf = 0.6

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
            # Face is present — clear the face-loss timer
            self._face_lost_at = None
            face_landmarks = face_result.landmarks[0]

            transform_matrix = (
                face_result.transformation_matrices[0]
                if face_result.transformation_matrices
                else None
            )

            # Estimate head pose
            head_pose = self._estimate_head_pose(
                face_landmarks,
                frame_width,
                frame_height,
                transform_matrix=transform_matrix,
            )

            # Calculate eye metrics — pass real imaging_conf instead of 1.0
            blendshapes = face_result.blendshapes[0] if face_result.blendshapes else None
            eye_metrics = self._calculate_eye_metrics(
                face_landmarks,
                blendshapes,
                timestamp_s=timestamp_s,
                quality_hint=imaging_conf,
            )
        else:
            # Face not detected — only reset pose smoothing after a sustained absence
            now_s = float(timestamp_ms) / 1000.0
            if self._face_lost_at is None:
                self._face_lost_at = now_s
            face_absent_seconds = now_s - self._face_lost_at
            if face_absent_seconds >= self._face_reset_threshold_seconds:
                # Truly absent: clear tracking state
                self._latest_raw_pose = None
                self._prev_rvec = None
                self._prev_tvec = None
                self._smooth_pitch = None
                self._smooth_yaw = None
                self._smooth_roll = None

        # Calculate hand metrics
        hand_metrics = self._calculate_hand_metrics(
            hand_result,
            timestamp_s=timestamp_s,
            head_pose=head_pose,
        )

        quality = self._assess_quality(
            frame=frame,
            timestamp_s=timestamp_s,
            face_detected=face_detected,
            face_landmarks=face_landmarks,
            head_pose=head_pose,
            eye_metrics=eye_metrics,
            hand_metrics=hand_metrics,
        )

        if eye_metrics is not None:
            eye_metrics.eye_confidence = self._clamp(
                (0.65 * eye_metrics.eye_confidence) + (0.35 * quality.overall_confidence),
                0.0,
                1.0,
            )

        self._record_calibration_sample(
            head_pose=head_pose,
            eye_metrics=eye_metrics,
            quality=quality,
        )

        result = VisionResult(
            timestamp_ms=timestamp_ms,
            face_detected=face_detected,
            face_landmarks=face_landmarks,
            head_pose=head_pose,
            eye_metrics=eye_metrics,
            hand_metrics=hand_metrics,
            quality=quality,
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
        cv2.putText(display, f"Head: P={hp.pitch:.0f} Y={hp.yaw:.0f} R={hp.roll:.0f} C={hp.confidence:.2f}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        y_offset += 20

    # Eye metrics
    if result.eye_metrics:
        em = result.eye_metrics
        status = "Closed" if em.is_closed else "Open"
        color = RED if em.is_closed else GREEN
        cv2.putText(display, f"Eyes: {status} EAR={em.avg_ear:.2f} BR={em.blink_rate_per_min:.1f}/m",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 20
        cv2.putText(display, f"PERCLOS={em.perclos_ratio:.2f} Cl={em.eye_closure_ratio:.2f}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        y_offset += 20

    # Hand status
    if result.hand_metrics:
        hm = result.hand_metrics
        if hm.detected:
            cv2.putText(display, f"Hand: {hm.region} write={hm.write_score:.2f} conf={hm.writing_confidence:.2f}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1)
        else:
            cv2.putText(display, "Hand: Not detected",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        y_offset += 20

    quality = result.quality
    q_color = GREEN if quality.overall_confidence >= 0.65 else RED if quality.overall_confidence < 0.4 else WHITE
    cv2.putText(
        display,
        f"Quality: {quality.overall_confidence:.2f} ({quality.lighting_quality})",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        q_color,
        1,
    )
    y_offset += 20
    if quality.quality_warnings:
        warning = quality.quality_warnings[0][:58]
        cv2.putText(display, f"Warn: {warning}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)

    return display
