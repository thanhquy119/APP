"""Vision demo based on the unified VisionPipeline runtime path."""

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.vision import CameraCapture, CameraConfig, VisionPipeline, draw_vision_overlay
from app.vision.phone_detector import PhoneDetector, PhoneDetectorConfig, PhoneState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class VisionDemo:
    """Interactive demo for vision quality, behavior signals, and calibration."""

    def __init__(self):
        self.camera = CameraCapture(
            CameraConfig(
                camera_index=0,
                width=640,
                height=480,
                fps=30,
                process_width=480,
                process_height=360,
            )
        )
        self.pipeline = VisionPipeline(use_live_stream=False)
        self.phone_detector = PhoneDetector(
            PhoneDetectorConfig(
                enabled=True,
                model_type="heuristic",
                confidence_threshold=0.55,
                run_interval_frames=3,
                confirmation_window_seconds=2.5,
                confirmation_min_hits=3,
            )
        )

        self.running = False
        self.show_debug = True
        self.fps = 0.0
        self._frame_times: list[float] = []
        self._last_console_print = 0.0

    def start(self) -> bool:
        """Start camera and vision components."""
        logger.info("Starting unified vision demo")

        if not self.camera.start():
            logger.error("Failed to start camera")
            return False

        process_w, process_h = self.camera.get_process_size()
        if not self.pipeline.initialize(process_w, process_h):
            logger.error("Failed to initialize VisionPipeline")
            self.camera.stop()
            return False

        if not self.phone_detector.initialize():
            logger.warning("Phone detector unavailable, continuing without strong phone signal")

        self.running = True
        self._last_console_print = time.time()
        return True

    def stop(self) -> None:
        """Stop and release all resources."""
        self.running = False
        self.phone_detector.release()
        self.pipeline.close()
        self.camera.stop()
        cv2.destroyAllWindows()
        logger.info("Demo stopped")

    def run(self) -> None:
        """Run real-time loop."""
        if not self.start():
            return

        print("\n" + "=" * 62)
        print("FOCUSGUARDIAN - UNIFIED VISION DEMO")
        print("=" * 62)
        print("Controls:")
        print("  q - Quit")
        print("  c - Start 3-second neutral calibration")
        print("  d - Toggle debug panel")
        print("=" * 62 + "\n")

        try:
            while self.running and self.camera.is_running:
                frame_start = time.time()
                frame = self.camera.get_processed_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                timestamp_ms = int(time.time() * 1000)
                result = self.pipeline.process(frame, timestamp_ms=timestamp_ms)
                phone_state = self.phone_detector.process(frame, timestamp_ms=timestamp_ms)

                result.phone_present = bool(phone_state.phone_present)
                result.phone_confidence = float(phone_state.phone_confidence)
                result.phone_bbox = phone_state.phone_bbox
                result.phone_evidence_source = str(phone_state.phone_evidence_source)

                calibration_result = self.pipeline.consume_latest_calibration_result()
                if calibration_result is not None:
                    status = "OK" if calibration_result.success else "FAILED"
                    print(
                        f"Calibration {status}: {calibration_result.message} "
                        f"(samples={calibration_result.sample_count}, "
                        f"ear_threshold={calibration_result.calibration.ear_closed_threshold_adaptive:.3f})"
                    )

                if self.show_debug:
                    display = draw_vision_overlay(frame, result, show_landmarks=False)
                    self._draw_debug_panel(
                        display,
                        result=result,
                        phone_state=phone_state,
                        calibration_progress=self.pipeline.get_calibration_progress(),
                    )
                else:
                    display = frame.copy()

                self._update_fps(frame_start)
                cv2.putText(
                    display,
                    f"FPS: {self.fps:.1f}",
                    (8, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 0),
                    1,
                )

                cv2.imshow("FocusGuardian Vision Demo", display)

                now = time.time()
                if now - self._last_console_print >= 1.0:
                    self._print_status(result, phone_state)
                    self._last_console_print = now

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("d"):
                    self.show_debug = not self.show_debug
                    print(f"Debug panel: {'ON' if self.show_debug else 'OFF'}")
                if key == ord("c"):
                    if self.pipeline.is_calibrating:
                        print("Calibration already in progress")
                    else:
                        self.pipeline.start_calibration(duration_seconds=3.0)
                        print("Calibration started: keep neutral posture for 3 seconds")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def _update_fps(self, frame_start: float) -> None:
        frame_time = max(1e-4, time.time() - frame_start)
        self._frame_times.append(frame_time)
        if len(self._frame_times) > 40:
            self._frame_times.pop(0)
        self.fps = 1.0 / (sum(self._frame_times) / max(1, len(self._frame_times)))

    def _draw_debug_panel(
        self,
        frame: np.ndarray,
        result,
        phone_state: PhoneState,
        calibration_progress: float,
    ) -> None:
        h, w = frame.shape[:2]
        panel_w = min(390, max(290, w // 2))
        x0 = max(6, w - panel_w - 6)
        y0 = 6
        y1 = min(h - 6, y0 + 310)  # taller panel for more signals

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

        lines: list[tuple[str, tuple[int, int, int]]] = []
        lines.append(("── Vision diagnostics ──", (200, 200, 200)))

        # Head pose + confidence
        if result.head_pose is not None:
            hp = result.head_pose
            lines.append((f"Head P/Y/R: {hp.pitch:+.1f}/{hp.yaw:+.1f}/{hp.roll:+.1f}", (220, 220, 220)))
            conf_color = (80, 220, 80) if hp.confidence >= 0.65 else (70, 170, 255) if hp.confidence >= 0.40 else (60, 60, 255)
            lines.append((f"Head confidence: {hp.confidence:.2f} [{hp.source}]", conf_color))
        else:
            lines.append(("Head: n/a", (130, 130, 130)))

        # Eye metrics
        if result.eye_metrics is not None:
            em = result.eye_metrics
            eye_conf_color = (80, 220, 80) if em.eye_confidence >= 0.60 else (70, 170, 255) if em.eye_confidence >= 0.40 else (60, 60, 255)
            lines.append((f"EAR: {em.avg_ear:.3f}  eye_conf: {em.eye_confidence:.2f}", eye_conf_color))
            lines.append((f"Blink rate: {em.blink_rate_per_min:.1f}/min  count: {em.blink_count_window}", (220, 220, 220)))
            lines.append((f"PERCLOS: {em.perclos_ratio:.2f}  closure_ratio: {em.eye_closure_ratio:.2f}", (220, 220, 220)))
            lines.append((f"look_down: {em.look_down:.2f}  look_up: {em.look_up:.2f}", (220, 220, 220)))
        else:
            lines.append(("Eyes: n/a", (130, 130, 130)))

        # Hand metrics
        if result.hand_metrics is not None:
            hm = result.hand_metrics
            wc = (80, 220, 80) if hm.writing_confidence >= 0.60 else (220, 220, 220)
            lines.append((f"Write score: {hm.write_score:.2f}  conf: {hm.writing_confidence:.2f}", wc))
            lines.append((f"Hand region: {hm.region}  energy: {hm.motion_energy:.3f}", (220, 220, 220)))
        else:
            lines.append(("Hands: n/a", (130, 130, 130)))

        # Phone
        ph_color = (60, 60, 255) if phone_state.phone_present else (110, 210, 255)
        phone_str = (
            f"Phone: {'YES' if phone_state.phone_present else 'no'} "
            f"conf={phone_state.phone_confidence:.2f} src={phone_state.phone_evidence_source} "
            f"strong={phone_state.strong_present}"
        )
        lines.append((phone_str, ph_color))

        # Quality
        quality = result.quality
        q_color = (80, 220, 80) if quality.overall_confidence >= 0.65 else (70, 170, 255) if quality.overall_confidence >= 0.40 else (60, 60, 255)
        lines.append((f"Overall conf: {quality.overall_confidence:.2f}", q_color))
        lines.append((f"Face track: {quality.face_tracking_confidence:.2f}  head: {quality.head_pose_confidence:.2f}", (220, 220, 220)))
        lines.append((f"Eye conf: {quality.eye_confidence:.2f}  blur: {quality.blur_score:.0f}  bright: {quality.frame_brightness:.0f}", (220, 220, 220)))

        # Calibration
        if calibration_progress > 0.0:
            lines.append((f"Calibration: {int(calibration_progress * 100)}%", (100, 230, 255)))

        # Warnings (up to 3)
        for warning in quality.quality_warnings[:3]:
            lines.append((f"! {warning}", (90, 90, 255)))

        line_y = y0 + 16
        for text, color in lines:
            if line_y > y1 - 10:
                break
            cv2.putText(
                frame,
                text,
                (x0 + 8, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
            )
            line_y += 15

    def _print_status(self, result, phone_state: PhoneState) -> None:
        parts: list[str] = []

        if result.head_pose is not None:
            hp = result.head_pose
            parts.append(f"Head {hp.pitch:+.1f}/{hp.yaw:+.1f}/{hp.roll:+.1f} c={hp.confidence:.2f}")
        else:
            parts.append("Head n/a")

        if result.eye_metrics is not None:
            em = result.eye_metrics
            parts.append(f"EAR {em.avg_ear:.3f}")
            parts.append(f"BlinkRate {em.blink_rate_per_min:.1f}/m")
            parts.append(f"PERCLOS {em.perclos_ratio:.2f}")

        if result.hand_metrics is not None:
            hm = result.hand_metrics
            parts.append(f"Write {hm.write_score:.2f}")

        parts.append(f"Phone {phone_state.phone_confidence:.2f}/{phone_state.phone_evidence_source}")
        parts.append(f"Q {result.quality.overall_confidence:.2f}")
        if result.quality.quality_warnings:
            parts.append(f"Warn {result.quality.quality_warnings[0]}")

        parts.append(f"FPS {self.fps:.1f}")
        print(" | ".join(parts))


def main() -> None:
    demo = VisionDemo()
    demo.run()


if __name__ == "__main__":
    main()
