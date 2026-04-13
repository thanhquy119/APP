"""
Head pose estimation using solvePnP from MediaPipe Face Mesh landmarks.
Estimates pitch (up/down), yaw (left/right), and roll (tilt) angles.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, NamedTuple
from dataclasses import dataclass

from .face_mesh import FaceLandmarks, FaceMeshIndices

logger = logging.getLogger(__name__)


class HeadPose(NamedTuple):
    """Head pose estimation result."""
    pitch: float  # Positive = looking up, Negative = looking down
    yaw: float    # Positive = looking right, Negative = looking left
    roll: float   # Positive = tilting right, Negative = tilting left

    # Translation vector (distance from camera)
    translation: np.ndarray

    # Rotation vector (for 3D rendering)
    rotation_vec: np.ndarray

    # Confidence based on reprojection error
    confidence: float


@dataclass
class HeadPoseConfig:
    """Configuration for head pose estimation."""
    # Smoothing factor for exponential moving average (0-1, higher = more smoothing)
    smoothing_factor: float = 0.3

    # Threshold for considering head as "down" (negative pitch in degrees)
    head_down_threshold: float = -15.0

    # Threshold for considering head as looking away (yaw in degrees)
    look_away_threshold: float = 30.0

    # Calibration offsets (set during calibration)
    pitch_offset: float = 0.0
    yaw_offset: float = 0.0
    roll_offset: float = 0.0


class HeadPoseEstimator:
    """
    Estimates head pose (pitch, yaw, roll) from Face Mesh landmarks using solvePnP.
    Uses a 3D model of facial landmarks and projects them to solve for camera pose.
    """

    # 3D model points for a generic face (in mm, centered at nose)
    # These are approximate positions based on average human face proportions
    MODEL_POINTS_6 = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (-30.0, -35.0, -30.0),    # Left eye outer corner
        (30.0, -35.0, -30.0),     # Right eye outer corner
        (-25.0, 30.0, -20.0),     # Left mouth corner
        (25.0, 30.0, -20.0),      # Right mouth corner
        (0.0, 55.0, -10.0),       # Chin
    ], dtype=np.float64)

    # Extended model with more points for better accuracy
    MODEL_POINTS_10 = np.array([
        (0.0, 0.0, 0.0),          # 1: Nose tip
        (-30.0, -35.0, -30.0),    # 33: Left eye outer corner
        (30.0, -35.0, -30.0),     # 263: Right eye outer corner
        (-25.0, 30.0, -20.0),     # 61: Left mouth corner
        (25.0, 30.0, -20.0),      # 291: Right mouth corner
        (0.0, 55.0, -10.0),       # 199: Chin
        (0.0, -60.0, -20.0),      # 10: Forehead
        (0.0, 70.0, 0.0),         # 152: Chin bottom
        (-50.0, 0.0, -40.0),      # 234: Left cheek
        (50.0, 0.0, -40.0),       # 454: Right cheek
    ], dtype=np.float64)

    def __init__(self, config: Optional[HeadPoseConfig] = None):
        self.config = config or HeadPoseConfig()

        # Camera matrix placeholder (will be computed based on frame size)
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Assume no lens distortion

        # Smoothed values
        self._smooth_pitch = 0.0
        self._smooth_yaw = 0.0
        self._smooth_roll = 0.0

        # Previous rotation vector for tracking
        self._prev_rvec: Optional[np.ndarray] = None
        self._prev_tvec: Optional[np.ndarray] = None

        # Use extended model for better accuracy
        self._model_points = self.MODEL_POINTS_10
        self._landmark_indices = FaceMeshIndices.POSE_LANDMARKS_EXTENDED

        # Statistics
        self._frame_count = 0
        self._success_count = 0

    def _get_camera_matrix(self, image_width: int, image_height: int) -> np.ndarray:
        """
        Compute camera intrinsic matrix.
        Assumes a typical webcam with ~60-70 degree FOV.
        """
        # Approximate focal length based on image size
        # This is a rough estimate; for better accuracy, camera should be calibrated
        focal_length = image_width * 1.0  # Approximation for ~60 degree horizontal FOV

        center = (image_width / 2, image_height / 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        return camera_matrix

    def estimate(self, landmarks: FaceLandmarks) -> Optional[HeadPose]:
        """
        Estimate head pose from face landmarks.

        Args:
            landmarks: FaceLandmarks from FaceMeshDetector

        Returns:
            HeadPose with pitch, yaw, roll angles, or None if estimation fails
        """
        self._frame_count += 1

        try:
            # Get camera matrix for this image size
            if (self._camera_matrix is None or
                self._camera_matrix[0, 2] != landmarks.image_width / 2):
                self._camera_matrix = self._get_camera_matrix(
                    landmarks.image_width, landmarks.image_height
                )

            # Extract 2D image points for key landmarks
            image_points = landmarks.get_landmarks_array(
                self._landmark_indices,
                pixel=True
            )[:, :2]  # Only x, y

            # Use previous solution as initial guess if available
            use_extrinsic_guess = self._prev_rvec is not None

            if use_extrinsic_guess:
                rvec = self._prev_rvec.copy()
                tvec = self._prev_tvec.copy()
            else:
                rvec = None
                tvec = None

            # Solve PnP to get rotation and translation
            if use_extrinsic_guess:
                success, rvec, tvec = cv2.solvePnP(
                    self._model_points,
                    image_points,
                    self._camera_matrix,
                    self._dist_coeffs,
                    rvec=rvec,
                    tvec=tvec,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            else:
                success, rvec, tvec = cv2.solvePnP(
                    self._model_points,
                    image_points,
                    self._camera_matrix,
                    self._dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

            if not success:
                logger.debug("solvePnP failed")
                return None

            # Calculate reprojection error for confidence
            projected_points, _ = cv2.projectPoints(
                self._model_points,
                rvec,
                tvec,
                self._camera_matrix,
                self._dist_coeffs
            )
            projected_points = projected_points.reshape(-1, 2)

            reprojection_error = np.mean(np.linalg.norm(image_points - projected_points, axis=1))
            confidence = max(0.0, min(1.0, 1.0 - reprojection_error / 50.0))

            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # Extract Euler angles from rotation matrix
            # Using the convention: pitch (x), yaw (y), roll (z)
            pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_matrix)

            # Convert to degrees
            pitch_deg = np.degrees(pitch)
            yaw_deg = np.degrees(yaw)
            roll_deg = np.degrees(roll)

            # Apply calibration offsets
            pitch_deg -= self.config.pitch_offset
            yaw_deg -= self.config.yaw_offset
            roll_deg -= self.config.roll_offset

            # Apply smoothing
            alpha = self.config.smoothing_factor
            self._smooth_pitch = alpha * pitch_deg + (1 - alpha) * self._smooth_pitch
            self._smooth_yaw = alpha * yaw_deg + (1 - alpha) * self._smooth_yaw
            self._smooth_roll = alpha * roll_deg + (1 - alpha) * self._smooth_roll

            # Save for next iteration
            self._prev_rvec = rvec
            self._prev_tvec = tvec
            self._success_count += 1

            return HeadPose(
                pitch=self._smooth_pitch,
                yaw=self._smooth_yaw,
                roll=self._smooth_roll,
                translation=tvec.flatten(),
                rotation_vec=rvec.flatten(),
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Head pose estimation error: {e}")
            return None

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (pitch, yaw, roll).
        Uses the ZYX convention (yaw-pitch-roll).
        """
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(-R[2, 0], sy)  # Rotation around X
            yaw = np.arctan2(R[1, 0], R[0, 0])  # Rotation around Y
            roll = np.arctan2(R[2, 1], R[2, 2])  # Rotation around Z
        else:
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(-R[1, 2], R[1, 1])
            roll = 0

        return pitch, yaw, roll

    def is_head_down(self, pose: HeadPose) -> bool:
        """Check if head is tilted down (looking at desk/phone)."""
        return pose.pitch < self.config.head_down_threshold

    def is_looking_away(self, pose: HeadPose) -> bool:
        """Check if head is turned away from screen."""
        return abs(pose.yaw) > self.config.look_away_threshold

    def calibrate_neutral(self, pose: HeadPose) -> None:
        """Set current pose as neutral (straight ahead)."""
        self.config.pitch_offset = pose.pitch
        self.config.yaw_offset = pose.yaw
        self.config.roll_offset = pose.roll
        logger.info(f"Calibrated neutral pose: pitch={pose.pitch:.1f}, yaw={pose.yaw:.1f}")

    def reset_calibration(self) -> None:
        """Reset calibration to defaults."""
        self.config.pitch_offset = 0.0
        self.config.yaw_offset = 0.0
        self.config.roll_offset = 0.0

    def reset_smoothing(self) -> None:
        """Reset smoothing state (call when face is lost and reappears)."""
        self._smooth_pitch = 0.0
        self._smooth_yaw = 0.0
        self._smooth_roll = 0.0
        self._prev_rvec = None
        self._prev_tvec = None

    def get_stats(self) -> Tuple[int, int, float]:
        """Get (frame_count, success_count, success_rate)."""
        rate = self._success_count / self._frame_count if self._frame_count > 0 else 0.0
        return (self._frame_count, self._success_count, rate)

    def draw_pose(self, frame: np.ndarray, landmarks: FaceLandmarks,
                  pose: HeadPose, axis_length: float = 50.0) -> np.ndarray:
        """
        Draw head pose axes on frame.
        Red = X (right), Green = Y (up), Blue = Z (forward)
        """
        # Get nose position as origin
        nose_x, nose_y = landmarks.get_pixel_coords(FaceMeshIndices.NOSE_TIP)

        # Define 3D axis points
        axis_points = np.float32([
            [0, 0, 0],
            [axis_length, 0, 0],    # X axis (red)
            [0, axis_length, 0],    # Y axis (green)
            [0, 0, axis_length]     # Z axis (blue, pointing out)
        ])

        # Project axis points
        if self._camera_matrix is not None:
            projected, _ = cv2.projectPoints(
                axis_points,
                pose.rotation_vec,
                pose.translation,
                self._camera_matrix,
                self._dist_coeffs
            )
            projected = projected.reshape(-1, 2).astype(np.int32)

            # Draw axes
            origin = tuple(projected[0])
            cv2.arrowedLine(frame, origin, tuple(projected[1]), (0, 0, 255), 2)  # X: Red
            cv2.arrowedLine(frame, origin, tuple(projected[2]), (0, 255, 0), 2)  # Y: Green
            cv2.arrowedLine(frame, origin, tuple(projected[3]), (255, 0, 0), 2)  # Z: Blue

        # Draw text info
        text_y = 30
        cv2.putText(frame, f"Pitch: {pose.pitch:+.1f}°", (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        text_y += 25
        cv2.putText(frame, f"Yaw: {pose.yaw:+.1f}°", (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        text_y += 25
        cv2.putText(frame, f"Roll: {pose.roll:+.1f}°", (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        text_y += 25
        cv2.putText(frame, f"Conf: {pose.confidence:.2f}", (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Indicate head down
        if self.is_head_down(pose):
            cv2.putText(frame, "HEAD DOWN", (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
    head_pose = HeadPoseEstimator()

    if camera.start() and face_mesh.initialize():
        print("Head Pose test running. Press 'c' to calibrate, 'q' to quit.")

        while camera.is_running:
            frame = camera.get_processed_frame()
            if frame is not None:
                landmarks = face_mesh.process(frame)

                if landmarks is not None:
                    pose = head_pose.estimate(landmarks)

                    if pose is not None:
                        frame = head_pose.draw_pose(frame, landmarks, pose)
                else:
                    cv2.putText(frame, "No face detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    head_pose.reset_smoothing()

                cv2.imshow("Head Pose Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                if landmarks is not None:
                    pose = head_pose.estimate(landmarks)
                    if pose:
                        head_pose.calibrate_neutral(pose)
                        print("Calibrated!")

        face_mesh.release()
        camera.stop()
        cv2.destroyAllWindows()
    else:
        print("Failed to initialize")
