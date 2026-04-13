"""
Vision module for FocusGuardian.

Uses MediaPipe Tasks API for modern Python (3.12+) compatibility.

Main components:
- VisionPipeline: Unified pipeline combining face and hand detection
- FaceLandmarker: Face detection with 478 3D landmarks
- HandLandmarker: Hand detection with 21 3D landmarks per hand
- CameraCapture: Camera capture utility
"""

# Core pipeline (recommended entry point)
from .vision_pipeline import VisionPipeline, VisionResult, HeadPose, EyeMetrics, HandMetrics, draw_vision_overlay

# Individual components
from .face_landmarker import (
    FaceLandmarker,
    FaceLandmarkResult,
    calculate_ear,
    get_eye_gaze_vertical_from_blendshapes,
)
from .hand_landmarker import HandLandmarker, HandLandmarkResult, HandInfo
from .camera import CameraCapture, CameraConfig
from .model_manager import ensure_models, download_model, get_model_path

# Legacy compatibility (deprecated, use VisionPipeline instead)
try:
    from .face_mesh import FaceMeshDetector
    from .head_pose import HeadPoseEstimator
    from .blink import BlinkDetector
    from .hands import HandAnalyzer
except ImportError:
    # Legacy modules may not work on Python 3.12+
    FaceMeshDetector = None
    HeadPoseEstimator = None
    BlinkDetector = None
    HandAnalyzer = None

__all__ = [
    # Primary API
    'VisionPipeline',
    'VisionResult',
    'HeadPose',
    'EyeMetrics',
    'HandMetrics',
    'draw_vision_overlay',

    # Components
    'FaceLandmarker',
    'FaceLandmarkResult',
    'HandLandmarker',
    'HandLandmarkResult',
    'HandInfo',
    'CameraCapture',

    # Model management
    'ensure_models',
    'download_model',
    'get_model_path',
    'calculate_ear',
    'get_eye_gaze_vertical_from_blendshapes',

    # Legacy (deprecated)
    'FaceMeshDetector',
    'HeadPoseEstimator',
    'BlinkDetector',
    'HandAnalyzer',
]
