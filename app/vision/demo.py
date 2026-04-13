"""
FocusGuardian Vision Demo
=========================
Demo script to test all vision modules together.
Shows head pose (pitch/yaw), blink detection (EAR), and hand write score.

Usage:
    python -m app.vision.demo

Controls:
    q - Quit
    c - Calibrate head pose (look straight at screen)
    r - Reset all tracking
    h - Toggle hand detection
    p - Toggle preview mode (show/hide debug overlays)
"""

import cv2
import numpy as np
import logging
import time
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.vision.camera import CameraCapture, CameraConfig
from app.vision.face_mesh import FaceMeshDetector, FaceMeshConfig
from app.vision.head_pose import HeadPoseEstimator, HeadPoseConfig
from app.vision.blink import BlinkDetector, BlinkConfig
from app.vision.hands import HandAnalyzer, HandConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VisionDemo:
    """
    Integrated demo for all vision modules.
    """

    def __init__(self):
        # Initialize components
        self.camera = CameraCapture(CameraConfig(
            camera_index=0,
            width=640,
            height=480,
            process_width=480,
            process_height=360
        ))

        self.face_mesh = FaceMeshDetector(FaceMeshConfig(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ))

        self.head_pose = HeadPoseEstimator(HeadPoseConfig(
            smoothing_factor=0.4,
            head_down_threshold=-15.0,
            look_away_threshold=30.0
        ))

        self.blink_detector = BlinkDetector(BlinkConfig(
            ear_threshold=0.21,
            window_seconds=60.0
        ))

        self.hand_analyzer = HandAnalyzer(HandConfig(
            max_num_hands=2,
            min_detection_confidence=0.5
        ))

        # State
        self.show_hands = True
        self.show_debug = True
        self.running = False

        # Performance tracking
        self.frame_times = []
        self.fps = 0.0

    def start(self) -> bool:
        """Start all components."""
        logger.info("Starting vision demo...")

        if not self.camera.start():
            logger.error("Failed to start camera")
            return False

        if not self.face_mesh.initialize():
            logger.error("Failed to initialize face mesh")
            self.camera.stop()
            return False

        if self.show_hands and not self.hand_analyzer.initialize():
            logger.warning("Failed to initialize hand analyzer")
            self.show_hands = False

        self.running = True
        logger.info("Vision demo started successfully")
        return True

    def stop(self):
        """Stop all components."""
        self.running = False
        self.hand_analyzer.release()
        self.face_mesh.release()
        self.camera.stop()
        cv2.destroyAllWindows()
        logger.info("Vision demo stopped")

    def run(self):
        """Main loop."""
        if not self.start():
            return

        print("\n" + "="*60)
        print("FOCUSGUARDIAN VISION DEMO")
        print("="*60)
        print("Controls:")
        print("  q - Quit")
        print("  c - Calibrate (look straight at screen)")
        print("  r - Reset tracking")
        print("  h - Toggle hand detection")
        print("  d - Toggle debug overlays")
        print("="*60 + "\n")

        last_print_time = time.time()

        try:
            while self.running and self.camera.is_running:
                frame_start = time.time()

                # Get frame
                frame = self.camera.get_processed_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Create display frame
                display = frame.copy()

                # Process face mesh
                landmarks = self.face_mesh.process(frame)

                pose = None
                blink_state = None
                hand_state = None

                if landmarks is not None:
                    # Estimate head pose
                    pose = self.head_pose.estimate(landmarks)

                    # Detect blinks
                    blink_state = self.blink_detector.process(landmarks)

                    if self.show_debug:
                        # Draw face landmarks (subtle)
                        display = self.face_mesh.draw_key_points(display, landmarks, (0, 200, 0))

                        # Draw head pose
                        if pose is not None:
                            display = self.head_pose.draw_pose(display, landmarks, pose)

                        # Draw blink info
                        display = self.blink_detector.draw_eyes(display, landmarks, blink_state)
                else:
                    # No face detected
                    self.head_pose.reset_smoothing()
                    cv2.putText(display, "NO FACE DETECTED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Process hands
                if self.show_hands:
                    hand_state = self.hand_analyzer.process(frame)
                    if self.show_debug:
                        display = self.hand_analyzer.draw_hands(display, hand_state)

                # Draw FPS
                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time)
                if len(self.frame_times) > 30:
                    self.frame_times.pop(0)
                self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))

                cv2.putText(display, f"FPS: {self.fps:.1f}", (display.shape[1] - 100, display.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Draw summary panel
                self._draw_summary(display, pose, blink_state, hand_state)

                # Print console output periodically
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    self._print_status(pose, blink_state, hand_state)
                    last_print_time = current_time

                # Show frame
                cv2.imshow("FocusGuardian Vision Demo", display)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self._calibrate(landmarks, pose)
                elif key == ord('r'):
                    self._reset()
                elif key == ord('h'):
                    self._toggle_hands()
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    print(f"Debug overlays: {'ON' if self.show_debug else 'OFF'}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def _draw_summary(self, frame: np.ndarray, pose, blink_state, hand_state):
        """Draw summary panel on frame."""
        h, w = frame.shape[:2]

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 220, 120), (w - 5, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Summary text
        x = w - 210
        y = 140
        line_height = 20

        cv2.putText(frame, "=== SUMMARY ===", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += line_height

        if pose:
            pitch_color = (0, 255, 0) if pose.pitch >= -15 else (0, 165, 255)
            cv2.putText(frame, f"Head Pitch: {pose.pitch:+.1f}", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, pitch_color, 1)
            y += line_height

            cv2.putText(frame, f"Head Yaw: {pose.yaw:+.1f}", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y += line_height
        else:
            cv2.putText(frame, "Head: N/A", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
            y += line_height * 2

        if blink_state:
            cv2.putText(frame, f"EAR: {blink_state.ear_avg:.3f}", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y += line_height

            drowsy_color = (0, 0, 255) if blink_state.is_drowsy else (0, 255, 0)
            drowsy_text = "DROWSY" if blink_state.is_drowsy else "Alert"
            cv2.putText(frame, f"State: {drowsy_text}", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, drowsy_color, 1)
            y += line_height

        if hand_state and hand_state.hand_present:
            write_color = (0, 255, 0) if hand_state.is_writing_pattern else (200, 200, 200)
            cv2.putText(frame, f"Write: {hand_state.hand_write_score:.2f}", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, write_color, 1)
        else:
            cv2.putText(frame, "Hands: None", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

    def _print_status(self, pose, blink_state, hand_state):
        """Print status to console."""
        parts = []

        if pose:
            parts.append(f"Pitch:{pose.pitch:+6.1f}° Yaw:{pose.yaw:+6.1f}°")
        else:
            parts.append("No face")

        if blink_state:
            parts.append(f"EAR:{blink_state.ear_avg:.3f} Blinks:{blink_state.blink_count}")

        if hand_state and hand_state.hand_present:
            parts.append(f"WriteScore:{hand_state.hand_write_score:.2f}")

        parts.append(f"FPS:{self.fps:.1f}")

        print(" | ".join(parts))

    def _calibrate(self, landmarks, pose):
        """Calibrate neutral head position."""
        if pose:
            self.head_pose.calibrate_neutral(pose)
            print("✓ Head pose calibrated to current position")
        else:
            print("✗ Cannot calibrate - no face detected")

    def _reset(self):
        """Reset all tracking."""
        self.head_pose.reset_smoothing()
        self.head_pose.reset_calibration()
        self.blink_detector.reset()
        self.hand_analyzer.reset()
        print("✓ All tracking reset")

    def _toggle_hands(self):
        """Toggle hand detection."""
        if self.show_hands:
            self.show_hands = False
            print("Hand detection: OFF")
        else:
            if self.hand_analyzer.initialize():
                self.show_hands = True
                print("Hand detection: ON")
            else:
                print("Failed to enable hand detection")


def main():
    """Main entry point."""
    demo = VisionDemo()
    demo.run()


if __name__ == "__main__":
    main()
