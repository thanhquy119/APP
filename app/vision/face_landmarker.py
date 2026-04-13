"""
Face Landmarker - MediaPipe Tasks API implementation.

Uses FaceLandmarker for face detection and 478 3D landmarks.
Supports LIVE_STREAM mode with async callbacks.
"""

import time
import logging
import threading
from pathlib import Path
from typing import Optional, Callable, List, NamedTuple, Any
from dataclasses import dataclass, field

import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from .model_manager import get_model_path, download_model

logger = logging.getLogger(__name__)


@dataclass
class FaceLandmarkResult:
    """Result from face landmark detection."""
    timestamp_ms: int
    face_detected: bool
    landmarks: Optional[List[np.ndarray]] = None  # List of (478, 3) arrays
    blendshapes: Optional[List[dict]] = None  # List of blendshape dicts
    transformation_matrices: Optional[List[np.ndarray]] = None

    @property
    def num_faces(self) -> int:
        return len(self.landmarks) if self.landmarks else 0


class FaceLandmarker:
    """
    Face Landmarker using MediaPipe Tasks API.

    Detects faces and extracts 478 3D landmarks per face.
    Supports VIDEO mode (synchronous) and LIVE_STREAM mode (async).
    """

    # Key landmark indices for pose estimation
    NOSE_TIP = 1
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    CHIN = 199

    # Eye landmarks for EAR calculation
    LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

    def __init__(
        self,
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        use_live_stream: bool = False,
        result_callback: Optional[Callable[[FaceLandmarkResult], None]] = None,
    ):
        """
        Initialize FaceLandmarker.

        Args:
            num_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            use_live_stream: Use LIVE_STREAM mode (async) instead of VIDEO mode
            result_callback: Callback for LIVE_STREAM mode results
        """
        self.num_faces = num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.use_live_stream = use_live_stream
        self.result_callback = result_callback

        self._landmarker: Optional[Any] = None  # mp_vision.FaceLandmarker
        self._latest_result: Optional[FaceLandmarkResult] = None
        self._result_lock = threading.Lock()
        self._initialized = False

    def _ensure_model(self) -> Optional[Path]:
        """Ensure model file exists, download if needed."""
        model_path = get_model_path("face_landmarker")

        if not model_path.exists():
            logger.info("Face landmarker model not found, downloading...")
            model_path = download_model("face_landmarker")

        if model_path is None or not model_path.exists():
            logger.error("Failed to get face landmarker model")
            return None

        return model_path

    def initialize(self) -> bool:
        """Initialize the landmarker. Returns True on success."""
        if self._initialized:
            return True

        model_path = self._ensure_model()
        if model_path is None:
            return False

        try:
            base_options = mp_tasks.BaseOptions(
                model_asset_path=str(model_path)
            )

            if self.use_live_stream:
                running_mode = mp_vision.RunningMode.LIVE_STREAM
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=running_mode,
                    num_faces=self.num_faces,
                    min_face_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                    result_callback=self._on_result,
                )
            else:
                running_mode = mp_vision.RunningMode.VIDEO
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=running_mode,
                    num_faces=self.num_faces,
                    min_face_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                )

            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self._initialized = True
            logger.info(f"FaceLandmarker initialized (mode: {running_mode.name})")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize FaceLandmarker: {e}")
            return False

    def _on_result(
        self,
        result: Any,  # mp_vision.FaceLandmarkerResult
        output_image: Any,  # mp.Image
        timestamp_ms: int
    ):
        """Callback for LIVE_STREAM mode results."""
        face_result = self._convert_result(result, timestamp_ms)

        with self._result_lock:
            self._latest_result = face_result

        if self.result_callback:
            self.result_callback(face_result)

    def _convert_result(
        self,
        result: Any,  # mp_vision.FaceLandmarkerResult
        timestamp_ms: int
    ) -> FaceLandmarkResult:
        """Convert MediaPipe result to our format."""
        if not result.face_landmarks:
            return FaceLandmarkResult(
                timestamp_ms=timestamp_ms,
                face_detected=False
            )

        landmarks_list = []
        blendshapes_list = []
        matrices_list = []

        for face_landmarks in result.face_landmarks:
            # Convert landmarks to numpy array (478, 3)
            lm_array = np.array([
                [lm.x, lm.y, lm.z] for lm in face_landmarks
            ], dtype=np.float32)
            landmarks_list.append(lm_array)

        if result.face_blendshapes:
            for blendshapes in result.face_blendshapes:
                bs_dict = {bs.category_name: bs.score for bs in blendshapes}
                blendshapes_list.append(bs_dict)

        if result.facial_transformation_matrixes:
            for matrix in result.facial_transformation_matrixes:
                matrices_list.append(np.array(matrix))

        return FaceLandmarkResult(
            timestamp_ms=timestamp_ms,
            face_detected=True,
            landmarks=landmarks_list,
            blendshapes=blendshapes_list if blendshapes_list else None,
            transformation_matrices=matrices_list if matrices_list else None,
        )

    def process(
        self,
        frame: np.ndarray,
        timestamp_ms: Optional[int] = None
    ) -> Optional[FaceLandmarkResult]:
        """
        Process a frame for face landmarks.

        Args:
            frame: BGR image from OpenCV
            timestamp_ms: Frame timestamp in milliseconds (required for VIDEO/LIVE_STREAM)

        Returns:
            FaceLandmarkResult or None if not initialized
        """
        if not self._initialized and not self.initialize():
            return None

        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            if self.use_live_stream:
                # Async mode - result comes via callback
                self._landmarker.detect_async(mp_image, timestamp_ms)
                # Return latest cached result
                with self._result_lock:
                    return self._latest_result
            else:
                # Sync mode
                result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
                return self._convert_result(result, timestamp_ms)

        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return None

    def get_latest_result(self) -> Optional[FaceLandmarkResult]:
        """Get the latest result (for LIVE_STREAM mode)."""
        with self._result_lock:
            return self._latest_result

    def close(self):
        """Release resources."""
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
            self._initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def calculate_ear(landmarks: np.ndarray, eye_indices: List[int]) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) from landmarks.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    Args:
        landmarks: (478, 3) array of normalized landmarks
        eye_indices: 6 indices for eye landmarks [p1, p2, p3, p4, p5, p6]

    Returns:
        EAR value (typically 0.2-0.4 for open eyes, <0.2 for closed)
    """
    p1, p2, p3, p4, p5, p6 = [landmarks[i][:2] for i in eye_indices]

    # Vertical distances
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)

    # Horizontal distance
    h = np.linalg.norm(p1 - p4)

    if h < 1e-6:
        return 0.0

    return (v1 + v2) / (2.0 * h)


def get_eye_closure_from_blendshapes(blendshapes: dict) -> tuple[float, float]:
    """
    Get eye closure values from blendshapes.

    Returns:
        (left_closure, right_closure) - 0.0 = open, 1.0 = closed
    """
    left = blendshapes.get("eyeBlinkLeft", 0.0)
    right = blendshapes.get("eyeBlinkRight", 0.0)
    return left, right


def get_eye_gaze_vertical_from_blendshapes(blendshapes: dict) -> tuple[float, float]:
    """
    Estimate vertical eye gaze from blendshapes.

    Returns:
        (look_down, look_up) where each value is in [0, 1].
    """
    down_left = float(blendshapes.get("eyeLookDownLeft", 0.0))
    down_right = float(blendshapes.get("eyeLookDownRight", 0.0))
    up_left = float(blendshapes.get("eyeLookUpLeft", 0.0))
    up_right = float(blendshapes.get("eyeLookUpRight", 0.0))

    look_down = max(0.0, min(1.0, (down_left + down_right) / 2.0))
    look_up = max(0.0, min(1.0, (up_left + up_right) / 2.0))
    return look_down, look_up
