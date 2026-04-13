"""
Hand Landmarker - MediaPipe Tasks API implementation.

Uses HandLandmarker for hand detection and 21 3D landmarks per hand.
Supports LIVE_STREAM mode with async callbacks.
"""

import time
import logging
import threading
from pathlib import Path
from typing import Optional, Callable, List, NamedTuple, Any
from dataclasses import dataclass, field
from collections import deque

import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from .model_manager import get_model_path, download_model

logger = logging.getLogger(__name__)


@dataclass
class HandInfo:
    """Information about a detected hand."""
    landmarks: np.ndarray  # (21, 3) normalized landmarks
    world_landmarks: Optional[np.ndarray]  # (21, 3) world coordinates in meters
    handedness: str  # "Left" or "Right"
    handedness_score: float
    region: str  # "upper", "middle", "lower" based on Y position

    @property
    def center_y(self) -> float:
        """Get Y center of hand (normalized 0-1, 0=top)."""
        return np.mean(self.landmarks[:, 1])

    @property
    def center_x(self) -> float:
        """Get X center of hand (normalized 0-1)."""
        return np.mean(self.landmarks[:, 0])


@dataclass
class HandLandmarkResult:
    """Result from hand landmark detection."""
    timestamp_ms: int
    hand_detected: bool
    hands: List[HandInfo] = field(default_factory=list)

    @property
    def num_hands(self) -> int:
        return len(self.hands)

    def get_dominant_hand(self) -> Optional[HandInfo]:
        """Get the hand with highest confidence."""
        if not self.hands:
            return None
        return max(self.hands, key=lambda h: h.handedness_score)


class HandLandmarker:
    """
    Hand Landmarker using MediaPipe Tasks API.

    Detects hands and extracts 21 3D landmarks per hand.
    Supports VIDEO mode (synchronous) and LIVE_STREAM mode (async).
    """

    # Landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    INDEX_MCP = 5  # Base of index finger

    def __init__(
        self,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        use_live_stream: bool = False,
        result_callback: Optional[Callable[["HandLandmarkResult"], None]] = None,
    ):
        """
        Initialize HandLandmarker.

        Args:
            num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            use_live_stream: Use LIVE_STREAM mode (async) instead of VIDEO mode
            result_callback: Callback for LIVE_STREAM mode results
        """
        self.num_hands = num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.use_live_stream = use_live_stream
        self.result_callback = result_callback

        self._landmarker: Optional[Any] = None  # mp_vision.HandLandmarker
        self._latest_result: Optional[HandLandmarkResult] = None
        self._result_lock = threading.Lock()
        self._initialized = False

        # Motion tracking for write detection
        self._position_history: deque = deque(maxlen=30)  # ~1 second at 30fps
        self._last_timestamp = 0

    def _ensure_model(self) -> Optional[Path]:
        """Ensure model file exists, download if needed."""
        model_path = get_model_path("hand_landmarker")

        if not model_path.exists():
            logger.info("Hand landmarker model not found, downloading...")
            model_path = download_model("hand_landmarker")

        if model_path is None or not model_path.exists():
            logger.error("Failed to get hand landmarker model")
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
                options = mp_vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=running_mode,
                    num_hands=self.num_hands,
                    min_hand_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                    result_callback=self._on_result,
                )
            else:
                running_mode = mp_vision.RunningMode.VIDEO
                options = mp_vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=running_mode,
                    num_hands=self.num_hands,
                    min_hand_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )

            self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
            self._initialized = True
            logger.info(f"HandLandmarker initialized (mode: {running_mode.name})")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize HandLandmarker: {e}")
            return False

    def _on_result(
        self,
        result: Any,  # mp_vision.HandLandmarkerResult
        output_image: Any,  # mp.Image
        timestamp_ms: int
    ):
        """Callback for LIVE_STREAM mode results."""
        hand_result = self._convert_result(result, timestamp_ms)

        with self._result_lock:
            self._latest_result = hand_result

        if self.result_callback:
            self.result_callback(hand_result)

    def _get_hand_region(self, center_y: float) -> str:
        """Determine hand region based on Y position."""
        if center_y < 0.33:
            return "upper"
        elif center_y < 0.66:
            return "middle"
        else:
            return "lower"

    def _convert_result(
        self,
        result: Any,  # mp_vision.HandLandmarkerResult
        timestamp_ms: int
    ) -> HandLandmarkResult:
        """Convert MediaPipe result to our format."""
        if not result.hand_landmarks:
            return HandLandmarkResult(
                timestamp_ms=timestamp_ms,
                hand_detected=False
            )

        hands = []
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            # Convert landmarks to numpy array (21, 3)
            lm_array = np.array([
                [lm.x, lm.y, lm.z] for lm in hand_landmarks
            ], dtype=np.float32)

            # World landmarks if available
            world_lm = None
            if result.hand_world_landmarks and i < len(result.hand_world_landmarks):
                world_lm = np.array([
                    [lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[i]
                ], dtype=np.float32)

            # Handedness
            handedness = "Unknown"
            handedness_score = 0.0
            if result.handedness and i < len(result.handedness):
                handedness = result.handedness[i][0].category_name
                handedness_score = result.handedness[i][0].score

            # Determine region
            center_y = np.mean(lm_array[:, 1])
            region = self._get_hand_region(center_y)

            hands.append(HandInfo(
                landmarks=lm_array,
                world_landmarks=world_lm,
                handedness=handedness,
                handedness_score=handedness_score,
                region=region,
            ))

        return HandLandmarkResult(
            timestamp_ms=timestamp_ms,
            hand_detected=True,
            hands=hands,
        )

    def process(
        self,
        frame: np.ndarray,
        timestamp_ms: Optional[int] = None
    ) -> Optional[HandLandmarkResult]:
        """
        Process a frame for hand landmarks.

        Args:
            frame: BGR image from OpenCV
            timestamp_ms: Frame timestamp in milliseconds

        Returns:
            HandLandmarkResult or None if not initialized
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
            logger.error(f"Hand detection error: {e}")
            return None

    def calculate_write_score(self, result: HandLandmarkResult) -> float:
        """
        Calculate a writing score based on hand position and motion.

        Heuristics:
        - Hand in lower portion of frame (desk area)
        - Small, repetitive horizontal movements
        - Fingers in writing posture

        Returns:
            Score from 0.0 to 1.0 (higher = more likely writing)
        """
        if not result.hand_detected or not result.hands:
            self._position_history.clear()
            return 0.0

        hand = result.get_dominant_hand()
        if hand is None:
            return 0.0

        score = 0.0

        # Factor 1: Hand in lower region (desk area) - 0.4 weight
        if hand.region == "lower":
            score += 0.4
        elif hand.region == "middle":
            score += 0.2

        # Factor 2: Writing posture (index finger extended) - 0.2 weight
        # Check if index finger is more extended than others
        index_tip = hand.landmarks[self.INDEX_TIP]
        index_mcp = hand.landmarks[self.INDEX_MCP]
        wrist = hand.landmarks[self.WRIST]

        # Distance from wrist to index tip vs wrist to index base
        tip_dist = np.linalg.norm(index_tip[:2] - wrist[:2])
        mcp_dist = np.linalg.norm(index_mcp[:2] - wrist[:2])

        if mcp_dist > 0.01 and tip_dist / mcp_dist > 1.3:
            score += 0.2

        # Factor 3: Small repetitive motion - 0.4 weight
        current_pos = hand.center_x
        self._position_history.append((result.timestamp_ms, current_pos))

        if len(self._position_history) >= 10:
            positions = [p[1] for p in self._position_history]
            motion_std = np.std(positions)

            # Small but non-zero movement indicates writing
            # Too much = large gestures, too little = stationary
            if 0.005 < motion_std < 0.05:
                score += 0.4
            elif 0.002 < motion_std < 0.08:
                score += 0.2

        return min(1.0, score)

    def get_latest_result(self) -> Optional[HandLandmarkResult]:
        """Get the latest result (for LIVE_STREAM mode)."""
        with self._result_lock:
            return self._latest_result

    def close(self):
        """Release resources."""
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
            self._initialized = False
        self._position_history.clear()

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
