"""
Windows idle time detection using GetLastInputInfo.
Detects how long since the user last interacted with mouse/keyboard.
"""

import ctypes
import time
import logging
from ctypes import Structure, c_uint, sizeof, byref
from typing import Optional

logger = logging.getLogger(__name__)


class LASTINPUTINFO(Structure):
    """Windows LASTINPUTINFO structure."""
    _fields_ = [
        ('cbSize', c_uint),
        ('dwTime', c_uint)
    ]


class WindowsIdleDetector:
    """
    Detects user idle time on Windows using GetLastInputInfo.

    Measures time since last mouse movement, click, or keyboard input.
    Works system-wide, not just for the current application.
    """

    def __init__(self):
        self._available = False
        self._last_input_info = LASTINPUTINFO()
        self._last_input_info.cbSize = sizeof(LASTINPUTINFO)

        # Try to load Windows API
        try:
            self._user32 = ctypes.windll.user32
            self._kernel32 = ctypes.windll.kernel32
            self._available = True
            logger.info("Windows idle detection initialized")
        except Exception as e:
            logger.warning(f"Windows idle detection not available: {e}")
            self._user32 = None
            self._kernel32 = None

    @property
    def is_available(self) -> bool:
        """Check if idle detection is available on this system."""
        return self._available

    def get_idle_seconds(self) -> float:
        """
        Get idle time in seconds since last user input.

        Returns:
            Seconds since last mouse/keyboard input.
            Returns 0.0 if detection is not available.
        """
        if not self._available:
            return 0.0

        try:
            # Get last input time
            self._user32.GetLastInputInfo(byref(self._last_input_info))

            # Get current tick count
            current_tick = self._kernel32.GetTickCount()

            # Calculate idle time in milliseconds
            # Handle tick count overflow (happens every ~49 days)
            last_input_tick = self._last_input_info.dwTime

            if current_tick >= last_input_tick:
                idle_ms = current_tick - last_input_tick
            else:
                # Overflow occurred
                idle_ms = (0xFFFFFFFF - last_input_tick) + current_tick

            return idle_ms / 1000.0

        except Exception as e:
            logger.error(f"Error getting idle time: {e}")
            return 0.0

    def get_idle_minutes(self) -> float:
        """Get idle time in minutes."""
        return self.get_idle_seconds() / 60.0

    def is_idle(self, threshold_seconds: float = 5.0) -> bool:
        """
        Check if user is considered idle.

        Args:
            threshold_seconds: Minimum seconds of inactivity to be considered idle

        Returns:
            True if idle time exceeds threshold
        """
        return self.get_idle_seconds() >= threshold_seconds

    def is_active(self, threshold_seconds: float = 2.0) -> bool:
        """
        Check if user is actively using computer.

        Args:
            threshold_seconds: Maximum seconds since last input to be considered active

        Returns:
            True if user has had recent input
        """
        return self.get_idle_seconds() < threshold_seconds


class IdleTracker:
    """
    Tracks idle patterns over time.

    Provides statistics about user activity patterns,
    useful for detecting fatigue or disengagement.
    """

    def __init__(self, window_seconds: float = 60.0):
        """
        Initialize idle tracker.

        Args:
            window_seconds: Time window for statistics
        """
        self._detector = WindowsIdleDetector()
        self._window_seconds = window_seconds

        # History of (timestamp, idle_seconds) samples
        from collections import deque
        self._history: deque = deque(maxlen=1800)  # ~60 seconds at 30fps

        self._last_sample_time = 0.0
        self._sample_interval = 0.1  # Sample at most every 100ms

    @property
    def is_available(self) -> bool:
        return self._detector.is_available

    def update(self) -> float:
        """
        Sample current idle time and update history.

        Returns:
            Current idle time in seconds
        """
        now = time.time()

        # Rate limit sampling
        if now - self._last_sample_time < self._sample_interval:
            return self._detector.get_idle_seconds()

        idle = self._detector.get_idle_seconds()
        self._history.append((now, idle))
        self._last_sample_time = now

        return idle

    def get_current_idle(self) -> float:
        """Get current idle time without updating history."""
        return self._detector.get_idle_seconds()

    def get_idle_ratio(self, window_seconds: Optional[float] = None,
                       idle_threshold: float = 5.0) -> float:
        """
        Get ratio of time user was idle in window.

        Args:
            window_seconds: Time window (defaults to configured window)
            idle_threshold: Seconds of inactivity to count as idle

        Returns:
            Ratio 0.0-1.0 of time spent idle
        """
        window = window_seconds or self._window_seconds
        cutoff = time.time() - window

        samples = [(t, idle) for t, idle in self._history if t >= cutoff]
        if not samples:
            return 0.0

        idle_count = sum(1 for _, idle in samples if idle >= idle_threshold)
        return idle_count / len(samples)

    def get_avg_idle(self, window_seconds: Optional[float] = None) -> float:
        """
        Get average idle time in window.

        Returns:
            Average idle seconds
        """
        window = window_seconds or self._window_seconds
        cutoff = time.time() - window

        samples = [idle for t, idle in self._history if t >= cutoff]
        if not samples:
            return 0.0

        return sum(samples) / len(samples)

    def get_max_idle(self, window_seconds: Optional[float] = None) -> float:
        """
        Get maximum idle time in window.

        Returns:
            Maximum idle seconds observed
        """
        window = window_seconds or self._window_seconds
        cutoff = time.time() - window

        samples = [idle for t, idle in self._history if t >= cutoff]
        if not samples:
            return 0.0

        return max(samples)

    def get_activity_bursts(self, window_seconds: Optional[float] = None,
                            idle_threshold: float = 3.0) -> int:
        """
        Count activity bursts (transitions from idle to active) in window.

        High number of bursts may indicate distracted checking behavior.
        Low number may indicate sustained focus or disengagement.
        """
        window = window_seconds or self._window_seconds
        cutoff = time.time() - window

        samples = [(t, idle >= idle_threshold) for t, idle in self._history if t >= cutoff]
        if len(samples) < 2:
            return 0

        # Count idle -> active transitions
        bursts = 0
        for i in range(1, len(samples)):
            if samples[i-1][1] and not samples[i][1]:  # was idle, now active
                bursts += 1

        return bursts

    def clear(self) -> None:
        """Clear history."""
        self._history.clear()


# Singleton instance
_idle_detector: Optional[WindowsIdleDetector] = None


def get_idle_detector() -> WindowsIdleDetector:
    """Get shared idle detector instance."""
    global _idle_detector
    if _idle_detector is None:
        _idle_detector = WindowsIdleDetector()
    return _idle_detector


def get_idle_seconds() -> float:
    """Convenience function to get current idle time."""
    return get_idle_detector().get_idle_seconds()


def is_user_idle(threshold: float = 5.0) -> bool:
    """Convenience function to check if user is idle."""
    return get_idle_detector().is_idle(threshold)


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Windows Idle Time Test")
    print("=" * 40)

    detector = WindowsIdleDetector()

    if not detector.is_available:
        print("Idle detection not available on this system.")
    else:
        print("Move mouse/press keys to see idle time change.")
        print("Press Ctrl+C to exit.\n")

        tracker = IdleTracker(window_seconds=30.0)

        try:
            while True:
                idle = tracker.update()

                status = "IDLE" if idle >= 3.0 else "ACTIVE"
                bar_len = min(40, int(idle * 2))
                bar = "█" * bar_len + "░" * (40 - bar_len)

                print(f"\rIdle: {idle:5.1f}s [{bar}] {status}   ", end="", flush=True)

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nFinal statistics:")
            print(f"  Idle ratio (30s): {tracker.get_idle_ratio():.1%}")
            print(f"  Avg idle: {tracker.get_avg_idle():.1f}s")
            print(f"  Max idle: {tracker.get_max_idle():.1f}s")
            print(f"  Activity bursts: {tracker.get_activity_bursts()}")
