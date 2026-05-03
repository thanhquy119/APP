"""
Phone detection stub module.
Optional feature that can be enabled with YOLOv8n or MediaPipe Object Detection.
Currently provides a stub implementation that always returns no phone detected.
"""

import numpy as np
import cv2
import logging
import importlib
import time
from collections import deque
from typing import Optional, Tuple, NamedTuple, Deque, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PhoneState(NamedTuple):
    """Phone detection state."""
    phone_present: bool
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]]  # x, y, w, h
    evidence_source: str = "disabled"
    strong_present: bool = False
    detector_available: bool = False
    frame_evaluated: bool = False

    @property
    def phone_confidence(self) -> float:
        return float(self.confidence)

    @property
    def phone_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        return self.bbox

    @property
    def phone_evidence_source(self) -> str:
        return str(self.evidence_source)


@dataclass
class PhoneDetectorConfig:
    """Configuration for phone detection."""
    enabled: bool = False
    model_type: str = "heuristic"  # "heuristic", "stub", "yolov8", "mediapipe"
    confidence_threshold: float = 0.55
    model_path: Optional[str] = None
    run_interval_frames: int = 3
    confirmation_window_seconds: float = 2.5
    confirmation_min_hits: int = 5          # raised from 3 → 5 to reduce false positives
    heuristic_confidence_cap: float = 0.57  # lowered from 0.62 to limit heuristic overconfidence


class PhoneDetector:
    """
    Phone detection module.

    This is a stub implementation. To enable actual phone detection:

    Option 1: YOLOv8n (recommended for accuracy)
    - pip install ultralytics
    - Use YOLOv8n pretrained on COCO (class 67 = cell phone)

    Option 2: MediaPipe Object Detection
    - Use MediaPipe's object detection with EfficientDet

    Note: Phone detection adds ~50-100ms latency per frame with YOLOv8n.
    Consider running it every N frames or in a separate thread.
    """

    def __init__(self, config: Optional[PhoneDetectorConfig] = None):
        self.config = config or PhoneDetectorConfig()
        self._model = None
        self._initialized = False
        self._detection_count = 0
        self._detector_available = False
        self._frame_index = 0
        self._last_raw_state = PhoneState(False, 0.0, None, evidence_source="disabled", detector_available=False)
        self._last_output_state = PhoneState(False, 0.0, None, evidence_source="disabled", detector_available=False)
        self._evidence: Deque[dict[str, Any]] = deque(maxlen=240)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    def _empty_state(self, source: str, available: bool, evaluated: bool = False) -> PhoneState:
        return PhoneState(
            phone_present=False,
            confidence=0.0,
            bbox=None,
            evidence_source=source,
            strong_present=False,
            detector_available=available,
            frame_evaluated=evaluated,
        )

    def initialize(self) -> bool:
        """Initialize phone detector."""
        if not self.config.enabled:
            logger.info("Phone detection disabled (stub mode)")
            self._initialized = True
            self._detector_available = False
            return True

        if self.config.model_type == "heuristic":
            self._initialized = True
            self._detector_available = True
            logger.info("Phone detection: heuristic mode enabled")
            return True
        if self.config.model_type == "yolov8":
            return self._init_yolov8()
        elif self.config.model_type == "mediapipe":
            return self._init_mediapipe()
        else:
            logger.info("Phone detection: using stub (no actual detection)")
            self._initialized = True
            self._detector_available = False
            return True

    def _init_yolov8(self) -> bool:
        """Initialize YOLOv8n model."""
        try:
            ultralytics = importlib.import_module("ultralytics")
            YOLO = getattr(ultralytics, "YOLO")

            model_path = self.config.model_path or "yolov8n.pt"
            self._model = YOLO(model_path)
            self._initialized = True
            self._detector_available = True
            logger.info(f"YOLOv8 phone detector initialized: {model_path}")
            return True

        except ImportError:
            logger.warning("ultralytics not installed. Run: pip install ultralytics")
            self._detector_available = False
            return False
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8: {e}")
            self._detector_available = False
            return False

    def _init_mediapipe(self) -> bool:
        """Initialize MediaPipe Object Detection."""
        try:
            # MediaPipe object detection setup
            # This is a placeholder - actual implementation would use:
            # mediapipe.tasks.vision.ObjectDetector
            logger.warning("MediaPipe object detection not fully implemented")
            self._initialized = True
            self._detector_available = False
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            self._detector_available = False
            return False

    def release(self) -> None:
        """Release resources."""
        self._model = None
        self._initialized = False
        self._detector_available = False
        self._evidence.clear()
        self._frame_index = 0

    def process(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> PhoneState:
        """
        Detect phone in frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            PhoneState with detection results
        """
        if not self._initialized:
            return self._empty_state("unavailable", available=False, evaluated=False)

        if not self.config.enabled:
            return self._empty_state("disabled", available=False, evaluated=False)

        self._detection_count += 1
        self._frame_index += 1

        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        now_s = float(timestamp_ms) / 1000.0

        run_interval = max(1, int(self.config.run_interval_frames))
        should_run_backend = (self._frame_index % run_interval == 0) or (self._frame_index <= 2)

        if not self._detector_available and self.config.model_type == "yolov8":
            return self._empty_state("unavailable", available=False, evaluated=False)

        if should_run_backend:
            raw_state = self._run_backend(frame)
            self._last_raw_state = raw_state
            self._record_evidence(now_s, raw_state)
        else:
            raw_state = self._last_raw_state

        output_state = self._build_temporal_state(now_s, raw_state, evaluated=should_run_backend)
        self._last_output_state = output_state
        return output_state

    def _run_backend(self, frame: np.ndarray) -> PhoneState:
        if self.config.model_type == "heuristic":
            return self._process_heuristic(frame)

        if self.config.model_type == "yolov8" and self._model is not None:
            return self._process_yolov8(frame)

        if self.config.model_type == "mediapipe":
            return self._process_mediapipe(frame)

        return self._empty_state("stub", available=False, evaluated=True)

    def _record_evidence(self, timestamp_s: float, raw_state: PhoneState) -> None:
        self._evidence.append(
            {
                "ts": float(timestamp_s),
                "present": bool(raw_state.phone_present),
                "conf": float(raw_state.confidence),
                "bbox": raw_state.bbox,
                "source": str(raw_state.evidence_source),
            }
        )
        cutoff = float(timestamp_s) - max(0.6, float(self.config.confirmation_window_seconds))
        while self._evidence and self._evidence[0]["ts"] < cutoff:
            self._evidence.popleft()

    def _build_temporal_state(self, timestamp_s: float, raw_state: PhoneState, evaluated: bool) -> PhoneState:
        cutoff = float(timestamp_s) - max(0.6, float(self.config.confirmation_window_seconds))
        samples = [item for item in self._evidence if item["ts"] >= cutoff and item["source"] == raw_state.evidence_source]

        threshold = self._clamp(self.config.confidence_threshold, 0.05, 0.95)
        hits = [item for item in samples if item["present"] and item["conf"] >= threshold * 0.9]
        min_hits = max(1, int(self.config.confirmation_min_hits))
        strong_present = len(hits) >= min_hits

        if strong_present and hits:
            confidence = max(float(item["conf"]) for item in hits)
            bbox = hits[-1]["bbox"]
        else:
            confidence = float(raw_state.confidence)
            bbox = raw_state.bbox

        if raw_state.evidence_source == "heuristic":
            confidence = min(confidence, self._clamp(self.config.heuristic_confidence_cap, 0.45, 0.70))

        present = bool(strong_present)

        return PhoneState(
            phone_present=present,
            confidence=self._clamp(confidence, 0.0, 1.0),
            bbox=bbox if present else None,
            evidence_source=raw_state.evidence_source,
            strong_present=strong_present,
            detector_available=bool(raw_state.detector_available),
            frame_evaluated=bool(evaluated),
        )

    def _process_heuristic(self, frame: np.ndarray) -> PhoneState:
        """
        Detect phone-like rectangular objects with OpenCV heuristics.

        This fallback is intentionally lightweight and works without extra models.
        Stricter shape/size constraints reduce false positives from books, cups, monitors.
        """
        try:
            h, w = frame.shape[:2]
            frame_area = float(h * w)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            edges = cv2.Canny(gray, 70, 180)
            edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_conf = 0.0
            best_bbox: Optional[Tuple[int, int, int, int]] = None

            # Phones are typically 3–15 % of frame area; wider range misses context
            min_area = frame_area * 0.020
            max_area = frame_area * 0.40   # was 0.70 — monitors/books excluded

            for cnt in contours:
                contour_area = float(cv2.contourArea(cnt))
                if contour_area < min_area or contour_area > max_area:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter < 50:
                    continue

                approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
                if len(approx) < 4 or len(approx) > 8:  # was 10 — stricter polygon
                    continue

                x, y, bw, bh = cv2.boundingRect(approx)
                if bw < 45 or bh < 65:
                    continue

                rect_area = float(bw * bh)
                if rect_area <= 0:
                    continue

                fill_ratio = contour_area / rect_area
                if fill_ratio < 0.60:   # was 0.52 — tighter fill required
                    continue

                short_side = min(bw, bh)
                long_side = max(bw, bh)
                short_over_long = short_side / long_side

                # Phone aspect ratio: 0.40–0.60 (portrait) or 0.55–0.70 (landscape)
                # Books tend to be 0.65–0.80; cups are nearly square (0.85+)
                is_portrait_ratio = 0.38 <= short_over_long <= 0.62
                is_landscape_ratio = 0.55 <= short_over_long <= 0.72
                if not (is_portrait_ratio or is_landscape_ratio):
                    continue

                # Tight aspect score: reward ratios closest to 0.50 (portrait) or 0.625 (landscape)
                target = 0.50 if is_portrait_ratio else 0.625
                aspect_score = max(0.0, 1.0 - abs(short_over_long - target) / 0.18)

                area_ratio = rect_area / frame_area
                area_score = max(0.0, min(1.0, (area_ratio - 0.020) / 0.20))

                fill_score = max(0.0, min(1.0, (fill_ratio - 0.60) / 0.30))

                roi_edges = edges[y:y + bh, x:x + bw]
                edge_density = float(np.count_nonzero(roi_edges)) / max(1.0, float(roi_edges.size))
                edge_score = max(0.0, min(1.0, (edge_density - 0.04) / 0.18))

                # Phones are typically held in front, not pressed to frame edge
                cx = x + bw / 2.0
                cy = y + bh / 2.0
                nx = (cx - (w / 2.0)) / max(1.0, w / 2.0)
                ny = (cy - (h / 2.0)) / max(1.0, h / 2.0)
                center_dist = (nx * nx + ny * ny) ** 0.5
                center_score = max(0.0, 1.0 - min(1.0, center_dist * 1.3))

                confidence = (
                    0.38 * aspect_score
                    + 0.22 * fill_score
                    + 0.18 * area_score
                    + 0.14 * edge_score
                    + 0.08 * center_score
                )

                if confidence > best_conf:
                    best_conf = confidence
                    best_bbox = (x, y, bw, bh)

            best_conf = min(best_conf, self._clamp(self.config.heuristic_confidence_cap, 0.45, 0.70))

            if best_conf >= self.config.confidence_threshold and best_bbox is not None:
                return PhoneState(
                    phone_present=True,
                    confidence=best_conf,
                    bbox=best_bbox,
                    evidence_source="heuristic",
                    detector_available=True,
                    frame_evaluated=True,
                )

            return PhoneState(
                phone_present=False,
                confidence=best_conf,
                bbox=None,
                evidence_source="heuristic",
                detector_available=True,
                frame_evaluated=True,
            )

        except Exception as e:
            logger.debug(f"Heuristic phone detection error: {e}")
            return PhoneState(
                phone_present=False,
                confidence=0.0,
                bbox=None,
                evidence_source="heuristic",
                detector_available=True,
                frame_evaluated=True,
            )

    def _process_yolov8(self, frame: np.ndarray) -> PhoneState:
        """Process with YOLOv8."""
        try:
            # COCO class 67 = cell phone
            results = self._model(frame, classes=[67], verbose=False)

            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes

                # Get highest confidence detection
                best_idx = boxes.conf.argmax()
                conf = float(boxes.conf[best_idx])

                if conf >= self.config.confidence_threshold:
                    box = boxes.xyxy[best_idx].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)

                    return PhoneState(
                        phone_present=True,
                        confidence=conf,
                        bbox=(x1, y1, x2 - x1, y2 - y1),
                        evidence_source="yolov8",
                        detector_available=True,
                        frame_evaluated=True,
                    )

            return PhoneState(
                phone_present=False,
                confidence=0.0,
                bbox=None,
                evidence_source="yolov8",
                detector_available=True,
                frame_evaluated=True,
            )

        except Exception as e:
            logger.error(f"YOLOv8 processing error: {e}")
            return PhoneState(
                phone_present=False,
                confidence=0.0,
                bbox=None,
                evidence_source="yolov8",
                detector_available=False,
                frame_evaluated=True,
            )

    def _process_mediapipe(self, frame: np.ndarray) -> PhoneState:
        """Process with MediaPipe (placeholder)."""
        _ = frame
        return PhoneState(
            phone_present=False,
            confidence=0.0,
            bbox=None,
            evidence_source="mediapipe",
            detector_available=False,
            frame_evaluated=True,
        )

    def draw_detection(self, frame: np.ndarray, state: PhoneState) -> np.ndarray:
        """Draw phone detection on frame."""
        if state.phone_present and state.bbox is not None:
            x, y, w, h = state.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f"PHONE {state.confidence:.2f} ({state.evidence_source})", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show status
        status = "Phone Detection: "
        if not self.config.enabled:
            status += "DISABLED"
            color = (128, 128, 128)
        elif state.phone_present:
            status += "DETECTED"
            color = (0, 0, 255)
        elif not state.detector_available:
            status += "Unavailable"
            color = (0, 165, 255)
        else:
            status += "Not detected"
            color = (0, 255, 0)

        h = frame.shape[0]
        cv2.putText(frame, status, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

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

    print("Phone Detector Test")
    print("=" * 40)
    print("This module is a STUB by default.")
    print("To enable actual detection, install ultralytics:")
    print("  pip install ultralytics")
    print("Then set config.enabled=True and model_type='yolov8'")
    print("=" * 40)

    camera = CameraCapture()

    # Test with stub mode
    detector = PhoneDetector(PhoneDetectorConfig(enabled=False))

    if camera.start() and detector.initialize():
        print("Running with stub mode. Press 'q' to quit.")

        while camera.is_running:
            frame = camera.get_processed_frame()
            if frame is not None:
                state = detector.process(frame)
                frame = detector.draw_detection(frame, state)

                cv2.imshow("Phone Detection Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        detector.release()
        camera.stop()
        cv2.destroyAllWindows()
