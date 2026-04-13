"""
MediaPipe Face Mesh wrapper for facial landmark detection.
Provides 468 3D facial landmarks for head pose estimation and eye tracking.
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Optional, Tuple, List, NamedTuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FaceMeshConfig:
    """Configuration for Face Mesh detector."""
    max_num_faces: int = 1
    refine_landmarks: bool = True  # Enable iris landmarks for better eye tracking
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False  # False = tracking mode for video


class FaceLandmarks(NamedTuple):
    """Face landmarks data structure."""
    landmarks: np.ndarray  # Shape: (478, 3) if refined, (468, 3) otherwise
    image_width: int
    image_height: int

    def get_pixel_coords(self, index: int) -> Tuple[int, int]:
        """Get landmark position in pixel coordinates."""
        lm = self.landmarks[index]
        return (int(lm[0] * self.image_width), int(lm[1] * self.image_height))

    def get_normalized_coords(self, index: int) -> Tuple[float, float, float]:
        """Get landmark in normalized coordinates (0-1)."""
        return tuple(self.landmarks[index])

    def get_landmarks_array(self, indices: List[int], pixel: bool = True) -> np.ndarray:
        """Get array of landmarks for given indices."""
        if pixel:
            result = []
            for idx in indices:
                lm = self.landmarks[idx]
                result.append([
                    lm[0] * self.image_width,
                    lm[1] * self.image_height,
                    lm[2]  # Keep depth as-is
                ])
            return np.array(result, dtype=np.float32)
        else:
            return self.landmarks[indices]


# Key landmark indices for various features
class FaceMeshIndices:
    """Important Face Mesh landmark indices."""

    # Nose tip and bridge
    NOSE_TIP = 1
    NOSE_BRIDGE = 6

    # Face outline for head pose estimation (6 key points)
    POSE_LANDMARKS = [1, 33, 263, 61, 291, 199]
    # 1: nose tip
    # 33: left eye outer corner
    # 263: right eye outer corner
    # 61: left mouth corner
    # 291: right mouth corner
    # 199: chin

    # More landmarks for better pose estimation
    POSE_LANDMARKS_EXTENDED = [1, 33, 263, 61, 291, 199, 10, 152, 234, 454]
    # + 10: top of head (forehead)
    # + 152: chin bottom
    # + 234: left cheek
    # + 454: right cheek

    # Left eye landmarks (6 points for EAR calculation)
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    # Upper: 385, 387
    # Lower: 373, 380
    # Corners: 362 (inner), 263 (outer)

    # Right eye landmarks (6 points for EAR calculation)
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    # Upper: 160, 158
    # Lower: 153, 144
    # Corners: 33 (outer), 133 (inner)

    # Iris landmarks (if refine_landmarks=True)
    LEFT_IRIS = [468, 469, 470, 471, 472]  # center, top, right, bottom, left
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    # Mouth landmarks
    MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

    # Face oval for boundary detection
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


class FaceMeshDetector:
    """
    MediaPipe Face Mesh detector wrapper.
    Detects 468 (or 478 with iris refinement) 3D facial landmarks.
    """

    def __init__(self, config: Optional[FaceMeshConfig] = None):
        self.config = config or FaceMeshConfig()
        self._face_mesh = None
        self._initialized = False
        self._last_landmarks: Optional[FaceLandmarks] = None
        self._detection_count = 0
        self._no_detection_count = 0

    def initialize(self) -> bool:
        """Initialize MediaPipe Face Mesh."""
        try:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=self.config.max_num_faces,
                refine_landmarks=self.config.refine_landmarks,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
                static_image_mode=self.config.static_image_mode
            )
            self._initialized = True
            logger.info("FaceMesh initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize FaceMesh: {e}")
            self._initialized = False
            return False

    def release(self) -> None:
        """Release resources."""
        if self._face_mesh:
            self._face_mesh.close()
            self._face_mesh = None
        self._initialized = False

    def process(self, frame: np.ndarray) -> Optional[FaceLandmarks]:
        """
        Process a frame and detect face landmarks.

        Args:
            frame: BGR image from OpenCV

        Returns:
            FaceLandmarks if face detected, None otherwise
        """
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            results = self._face_mesh.process(rgb_frame)

            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                # Get first face
                face_landmarks = results.multi_face_landmarks[0]

                # Convert to numpy array
                h, w = frame.shape[:2]
                landmarks = np.array([
                    [lm.x, lm.y, lm.z] for lm in face_landmarks.landmark
                ], dtype=np.float32)

                self._last_landmarks = FaceLandmarks(
                    landmarks=landmarks,
                    image_width=w,
                    image_height=h
                )
                self._detection_count += 1
                self._no_detection_count = 0

                return self._last_landmarks
            else:
                self._no_detection_count += 1
                return None

        except Exception as e:
            logger.error(f"Face mesh processing error: {e}")
            return None

    def get_last_landmarks(self) -> Optional[FaceLandmarks]:
        """Get the last detected landmarks."""
        return self._last_landmarks

    def get_detection_stats(self) -> Tuple[int, int]:
        """Get (detection_count, no_detection_count)."""
        return (self._detection_count, self._no_detection_count)

    def draw_landmarks(self, frame: np.ndarray, landmarks: FaceLandmarks,
                       draw_tesselation: bool = False,
                       draw_contours: bool = True,
                       draw_irises: bool = True) -> np.ndarray:
        """
        Draw face mesh landmarks on frame.

        Args:
            frame: Image to draw on
            landmarks: FaceLandmarks to draw
            draw_tesselation: Draw full mesh connections
            draw_contours: Draw face contours
            draw_irises: Draw iris circles

        Returns:
            Frame with drawn landmarks
        """
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh

        # Convert landmarks back to MediaPipe format for drawing
        class FakeLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        class FakeLandmarkList:
            def __init__(self, landmarks_array):
                self.landmark = [FakeLandmark(lm[0], lm[1], lm[2]) for lm in landmarks_array]

        fake_landmarks = FakeLandmarkList(landmarks.landmarks)

        if draw_tesselation:
            mp_drawing.draw_landmarks(
                frame,
                fake_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

        if draw_contours:
            mp_drawing.draw_landmarks(
                frame,
                fake_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

        if draw_irises and self.config.refine_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                fake_landmarks,
                mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )

        return frame

    def draw_key_points(self, frame: np.ndarray, landmarks: FaceLandmarks,
                        color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw key landmark points for debugging."""
        # Draw pose landmarks
        for idx in FaceMeshIndices.POSE_LANDMARKS:
            x, y = landmarks.get_pixel_coords(idx)
            cv2.circle(frame, (x, y), 3, color, -1)
            cv2.putText(frame, str(idx), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # Draw eye landmarks
        for idx in FaceMeshIndices.LEFT_EYE + FaceMeshIndices.RIGHT_EYE:
            x, y = landmarks.get_pixel_coords(idx)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

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

    camera = CameraCapture()
    detector = FaceMeshDetector()

    if camera.start() and detector.initialize():
        print("Face Mesh test running. Press 'q' to quit.")

        while camera.is_running:
            frame = camera.get_processed_frame()
            if frame is not None:
                landmarks = detector.process(frame)

                if landmarks is not None:
                    # Draw landmarks
                    frame = detector.draw_key_points(frame, landmarks)

                    # Show nose position
                    nose_x, nose_y = landmarks.get_pixel_coords(FaceMeshIndices.NOSE_TIP)
                    cv2.putText(frame, f"Nose: ({nose_x}, {nose_y})", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No face detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.imshow("Face Mesh Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        detector.release()
        camera.stop()
        cv2.destroyAllWindows()
    else:
        print("Failed to initialize")
