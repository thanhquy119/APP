"""
Quick test script for vision modules.
Runs webcam and shows head pose, blink, and hand analysis.

Usage: python test_vision.py
"""

import cv2
import sys
import time

# Import vision modules
from app.vision.camera import CameraCapture, CameraConfig
from app.vision.face_mesh import FaceMeshDetector, FaceMeshIndices
from app.vision.head_pose import HeadPoseEstimator
from app.vision.blink import BlinkDetector
from app.vision.hands import HandAnalyzer


def main():
    print("="*60)
    print("FOCUSGUARDIAN - VISION MODULE TEST")
    print("="*60)

    # Initialize camera
    print("[1/5] Initializing camera...")
    camera = CameraCapture(CameraConfig(
        camera_index=0,
        width=640,
        height=480,
        process_width=480,
        process_height=360
    ))

    if not camera.start():
        print("ERROR: Cannot open camera!")
        print("Please check if webcam is connected.")
        return

    print("      Camera OK!")

    # Initialize face mesh
    print("[2/5] Initializing Face Mesh...")
    face_mesh = FaceMeshDetector()
    if not face_mesh.initialize():
        print("ERROR: Cannot initialize Face Mesh!")
        camera.stop()
        return
    print("      Face Mesh OK!")

    # Initialize head pose
    print("[3/5] Initializing Head Pose Estimator...")
    head_pose = HeadPoseEstimator()
    print("      Head Pose OK!")

    # Initialize blink detector
    print("[4/5] Initializing Blink Detector...")
    blink_detector = BlinkDetector()
    print("      Blink Detector OK!")

    # Initialize hand analyzer
    print("[5/5] Initializing Hand Analyzer...")
    hand_analyzer = HandAnalyzer()
    if not hand_analyzer.initialize():
        print("WARNING: Hand detection failed, running without it.")
        use_hands = False
    else:
        print("      Hand Analyzer OK!")
        use_hands = True

    print("="*60)
    print("All systems ready! Press 'q' to quit, 'c' to calibrate.")
    print("="*60)

    # Main loop
    fps_counter = []
    last_print = time.time()

    try:
        while camera.is_running:
            frame_start = time.time()

            # Get frame
            frame = camera.get_processed_frame()
            if frame is None:
                continue

            display = frame.copy()
            h, w = display.shape[:2]

            # Process face
            landmarks = face_mesh.process(frame)

            pose = None
            blink_state = None

            if landmarks is not None:
                # Head pose
                pose = head_pose.estimate(landmarks)

                # Blink
                blink_state = blink_detector.process(landmarks)

                # Draw on frame
                if pose:
                    # Draw pose axes
                    display = head_pose.draw_pose(display, landmarks, pose)

                if blink_state:
                    # Draw eye info
                    display = blink_detector.draw_eyes(display, landmarks, blink_state)
            else:
                cv2.putText(display, "NO FACE", (w//2 - 50, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Process hands
            hand_state = None
            if use_hands:
                hand_state = hand_analyzer.process(frame)
                display = hand_analyzer.draw_hands(display, hand_state)

            # FPS
            frame_time = time.time() - frame_start
            fps_counter.append(frame_time)
            if len(fps_counter) > 30:
                fps_counter.pop(0)
            fps = 1.0 / (sum(fps_counter) / len(fps_counter))

            cv2.putText(display, f"FPS: {fps:.1f}", (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Console output every second
            now = time.time()
            if now - last_print >= 1.0:
                output = []
                if pose:
                    output.append(f"Pitch:{pose.pitch:+6.1f}°")
                    output.append(f"Yaw:{pose.yaw:+6.1f}°")
                if blink_state:
                    output.append(f"EAR:{blink_state.ear_avg:.3f}")
                if hand_state and hand_state.hand_present:
                    output.append(f"WriteScore:{hand_state.hand_write_score:.2f}")
                output.append(f"FPS:{fps:.1f}")

                print(" | ".join(output))
                last_print = now

            # Show
            cv2.imshow("FocusGuardian Vision Test", display)

            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and pose:
                head_pose.calibrate_neutral(pose)
                print(">>> Calibrated head pose!")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        print("Cleaning up...")
        hand_analyzer.release()
        face_mesh.release()
        camera.stop()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()
