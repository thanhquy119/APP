"""Reliability-focused tests for the unified vision stack."""

import numpy as np
import pytest

from app.vision.face_landmarker import FaceLandmarkResult
from app.vision.hand_landmarker import HandLandmarkResult
from app.vision.phone_detector import PhoneDetector, PhoneDetectorConfig, PhoneState
from app.vision.vision_pipeline import VisionPipeline


class _NoFaceLandmarker:
    def process(self, frame, timestamp_ms=None):
        return FaceLandmarkResult(
            timestamp_ms=int(timestamp_ms or 0),
            face_detected=False,
            landmarks=None,
            blendshapes=None,
            transformation_matrices=None,
        )

    def close(self):
        return None


class _NoHandLandmarker:
    def process(self, frame, timestamp_ms=None):
        return HandLandmarkResult(
            timestamp_ms=int(timestamp_ms or 0),
            hand_detected=False,
            hands=[],
        )

    def close(self):
        return None


def _build_offline_pipeline() -> VisionPipeline:
    pipeline = VisionPipeline(use_live_stream=False)
    pipeline._initialized = True
    pipeline._face_landmarker = _NoFaceLandmarker()
    pipeline._hand_landmarker = _NoHandLandmarker()
    return pipeline


def test_no_face_frame_does_not_crash_pipeline():
    pipeline = _build_offline_pipeline()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    result = pipeline.process(frame, timestamp_ms=1000)

    assert result is not None
    assert result.face_detected is False
    assert result.head_pose is None
    assert result.eye_metrics is None
    assert result.hand_metrics is not None
    assert result.hand_metrics.detected is False
    pipeline.close()


def test_dark_and_blurry_frame_emits_quality_warnings():
    pipeline = _build_offline_pipeline()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    result = pipeline.process(frame, timestamp_ms=1040)
    warnings_text = " | ".join(result.quality.quality_warnings).lower()

    assert "anh sang yeu" in warnings_text
    assert "khung hinh mo" in warnings_text
    pipeline.close()


def test_calibration_export_apply_roundtrip_sanity():
    source = VisionPipeline(use_live_stream=False)
    payload = {
        "head_pitch_offset": -3.2,
        "head_yaw_offset": 1.8,
        "head_roll_offset": -0.9,
        "ear_open_baseline": 0.28,
        "ear_closed_threshold_adaptive": 0.20,
        "eye_closure_baseline": 0.11,
        "calibrated_at_ms": 1700000000000,
    }
    source.apply_calibration(payload)

    exported = source.export_calibration()

    target = VisionPipeline(use_live_stream=False)
    target.apply_calibration(exported)

    assert target.calibration.head_pitch_offset == pytest.approx(payload["head_pitch_offset"])
    assert target.calibration.head_yaw_offset == pytest.approx(payload["head_yaw_offset"])
    assert target.calibration.head_roll_offset == pytest.approx(payload["head_roll_offset"])
    assert target.calibration.ear_open_baseline == pytest.approx(payload["ear_open_baseline"])
    assert target.calibration.ear_closed_threshold_adaptive == pytest.approx(
        payload["ear_closed_threshold_adaptive"]
    )
    assert target.calibration.eye_closure_baseline == pytest.approx(payload["eye_closure_baseline"])


@pytest.mark.parametrize(
    "ear_samples,expected_threshold",
    [
        ([0.05] * 20, 0.12),
        ([1.00] * 20, 0.33),
    ],
)
def test_adaptive_ear_threshold_is_clamped_to_safe_range(ear_samples, expected_threshold):
    pipeline = VisionPipeline(use_live_stream=False)
    pipeline._calibration_pose_samples = [(0.0, 0.0, 0.0)] * 20
    pipeline._calibration_ear_samples = list(ear_samples)
    pipeline._calibration_closure_samples = [0.1] * 20
    pipeline._calibration_duration_seconds = 3.0

    pipeline._finalize_calibration()

    assert pipeline.calibration.ear_closed_threshold_adaptive == pytest.approx(expected_threshold)
    assert 0.12 <= pipeline.calibration.ear_closed_threshold_adaptive <= 0.33


def test_phone_heuristic_confidence_cap_is_enforced(monkeypatch):
    detector = PhoneDetector(
        PhoneDetectorConfig(
            enabled=True,
            model_type="heuristic",
            confidence_threshold=0.2,
            run_interval_frames=1,
            confirmation_window_seconds=1.0,
            confirmation_min_hits=1,
            heuristic_confidence_cap=0.62,
        )
    )
    assert detector.initialize() is True

    def _fake_backend(frame):
        return PhoneState(
            phone_present=True,
            confidence=0.98,
            bbox=(10, 10, 22, 36),
            evidence_source="heuristic",
            strong_present=True,
            detector_available=True,
            frame_evaluated=True,
        )

    monkeypatch.setattr(detector, "_run_backend", _fake_backend)

    state = detector.process(np.zeros((140, 140, 3), dtype=np.uint8), timestamp_ms=1000)

    assert state.phone_present is True
    assert state.evidence_source == "heuristic"
    assert state.phone_confidence <= 0.620001
    detector.release()
