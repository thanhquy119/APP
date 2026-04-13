"""
Camera capture module with threading support for real-time webcam access.
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CameraState(Enum):
    """Camera states."""
    STOPPED = "stopped"
    RUNNING = "running"
    ERROR = "error"
    NO_CAMERA = "no_camera"


@dataclass
class CameraConfig:
    """Camera configuration."""
    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    process_width: int = 480  # Resize for processing to reduce CPU load
    process_height: int = 360
    buffer_size: int = 1  # Minimize latency


class CameraCapture:
    """
    Thread-safe camera capture with frame buffering.
    Handles camera initialization, frame grabbing, and graceful error recovery.
    """

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._processed_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._state = CameraState.STOPPED
        self._fps_actual = 0.0
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._on_frame_callback: Optional[Callable[[np.ndarray], None]] = None

    @property
    def state(self) -> CameraState:
        return self._state

    @property
    def fps(self) -> float:
        return self._fps_actual

    @property
    def is_running(self) -> bool:
        return self._running and self._state == CameraState.RUNNING

    def set_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set callback function to be called on each new frame."""
        self._on_frame_callback = callback

    @staticmethod
    def list_cameras(max_cameras: int = 5) -> list[int]:
        """List available camera indices."""
        available = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(i)
                cap.release()
        return available

    def start(self) -> bool:
        """Start camera capture in a separate thread."""
        if self._running:
            logger.warning("Camera already running")
            return True

        # Try to open camera
        try:
            self._cap = cv2.VideoCapture(self.config.camera_index, cv2.CAP_DSHOW)

            if not self._cap.isOpened():
                logger.error(f"Cannot open camera {self.config.camera_index}")
                self._state = CameraState.NO_CAMERA
                return False

            # Configure camera
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

            # Verify settings
            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

            logger.info(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps}fps")

            self._running = True
            self._state = CameraState.RUNNING
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

            return True

        except Exception as e:
            logger.error(f"Camera start error: {e}")
            self._state = CameraState.ERROR
            return False

    def stop(self) -> None:
        """Stop camera capture."""
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self._cap:
            self._cap.release()
            self._cap = None

        self._state = CameraState.STOPPED
        logger.info("Camera stopped")

    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        consecutive_failures = 0
        max_failures = 30  # Allow some frame drops

        while self._running:
            try:
                if self._cap is None or not self._cap.isOpened():
                    self._state = CameraState.ERROR
                    break

                ret, frame = self._cap.read()

                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > max_failures:
                        logger.error("Too many consecutive frame failures")
                        self._state = CameraState.ERROR
                        break
                    time.sleep(0.01)
                    continue

                consecutive_failures = 0

                # Flip horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Create processed frame (smaller for vision algorithms)
                processed = cv2.resize(
                    frame,
                    (self.config.process_width, self.config.process_height),
                    interpolation=cv2.INTER_LINEAR
                )

                with self._frame_lock:
                    self._frame = frame
                    self._processed_frame = processed

                # Update FPS counter
                self._frame_count += 1
                now = time.time()
                elapsed = now - self._last_fps_time
                if elapsed >= 1.0:
                    self._fps_actual = self._frame_count / elapsed
                    self._frame_count = 0
                    self._last_fps_time = now

                # Call callback if set
                if self._on_frame_callback:
                    try:
                        self._on_frame_callback(processed.copy())
                    except Exception as e:
                        logger.error(f"Frame callback error: {e}")

            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                time.sleep(0.1)

        self._running = False

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest full-resolution frame (thread-safe)."""
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def get_processed_frame(self) -> Optional[np.ndarray]:
        """Get the latest processed (resized) frame for vision algorithms."""
        with self._frame_lock:
            return self._processed_frame.copy() if self._processed_frame is not None else None

    def get_frame_size(self) -> Tuple[int, int]:
        """Get configured frame size (width, height)."""
        return (self.config.width, self.config.height)

    def get_process_size(self) -> Tuple[int, int]:
        """Get processing frame size (width, height)."""
        return (self.config.process_width, self.config.process_height)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Available cameras:", CameraCapture.list_cameras())

    camera = CameraCapture()
    if camera.start():
        print("Camera started. Press 'q' to quit.")

        while camera.is_running:
            frame = camera.get_frame()
            if frame is not None:
                cv2.putText(frame, f"FPS: {camera.fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.stop()
        cv2.destroyAllWindows()
    else:
        print("Failed to start camera")
