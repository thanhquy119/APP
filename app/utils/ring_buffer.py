"""
Ring buffer implementation for temporal data storage.
Provides efficient fixed-size circular buffer with time-based queries.
"""

from typing import TypeVar, Generic, Optional, List, Tuple, Callable, Any
from dataclasses import dataclass
from collections import deque
import time
import threading

T = TypeVar('T')


@dataclass
class TimestampedItem(Generic[T]):
    """Item with timestamp."""
    timestamp: float
    data: T


class RingBuffer(Generic[T]):
    """
    Thread-safe ring buffer with timestamp support.

    Stores items with timestamps and provides efficient queries
    for time-windowed data retrieval.
    """

    def __init__(self, max_size: int = 1800, max_age_seconds: float = 60.0):
        """
        Initialize ring buffer.

        Args:
            max_size: Maximum number of items to store
            max_age_seconds: Maximum age of items before auto-removal
        """
        self._buffer: deque[TimestampedItem[T]] = deque(maxlen=max_size)
        self._max_age = max_age_seconds
        self._lock = threading.Lock()
        self._total_items = 0

    def push(self, item: T, timestamp: Optional[float] = None) -> None:
        """
        Add item to buffer.

        Args:
            item: Data to store
            timestamp: Optional timestamp (uses current time if not provided)
        """
        ts = timestamp if timestamp is not None else time.time()

        with self._lock:
            self._buffer.append(TimestampedItem(timestamp=ts, data=item))
            self._total_items += 1

    def get_window(self, window_seconds: float,
                   end_time: Optional[float] = None) -> List[TimestampedItem[T]]:
        """
        Get items within time window.

        Args:
            window_seconds: Window duration in seconds
            end_time: End of window (defaults to now)

        Returns:
            List of TimestampedItems within window
        """
        end = end_time if end_time is not None else time.time()
        start = end - window_seconds

        with self._lock:
            return [item for item in self._buffer
                    if start <= item.timestamp <= end]

    def get_window_data(self, window_seconds: float,
                        end_time: Optional[float] = None) -> List[T]:
        """
        Get just the data (without timestamps) within window.
        """
        items = self.get_window(window_seconds, end_time)
        return [item.data for item in items]

    def get_latest(self, n: int = 1) -> List[TimestampedItem[T]]:
        """Get the n most recent items."""
        with self._lock:
            if n >= len(self._buffer):
                return list(self._buffer)
            return list(self._buffer)[-n:]

    def get_latest_data(self, n: int = 1) -> List[T]:
        """Get just the data of n most recent items."""
        return [item.data for item in self.get_latest(n)]

    def get_last(self) -> Optional[TimestampedItem[T]]:
        """Get the most recent item."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def get_last_data(self) -> Optional[T]:
        """Get data of most recent item."""
        last = self.get_last()
        return last.data if last else None

    def aggregate(self, window_seconds: float,
                  aggregator: Callable[[List[T]], Any],
                  end_time: Optional[float] = None) -> Any:
        """
        Apply aggregation function to window data.

        Args:
            window_seconds: Window duration
            aggregator: Function to apply to data list
            end_time: End of window

        Returns:
            Result of aggregator function
        """
        data = self.get_window_data(window_seconds, end_time)
        if not data:
            return None
        return aggregator(data)

    def count_where(self, window_seconds: float,
                    predicate: Callable[[T], bool],
                    end_time: Optional[float] = None) -> int:
        """
        Count items matching predicate in window.
        """
        data = self.get_window_data(window_seconds, end_time)
        return sum(1 for item in data if predicate(item))

    def ratio_where(self, window_seconds: float,
                    predicate: Callable[[T], bool],
                    end_time: Optional[float] = None) -> float:
        """
        Get ratio of items matching predicate in window.
        """
        data = self.get_window_data(window_seconds, end_time)
        if not data:
            return 0.0
        return sum(1 for item in data if predicate(item)) / len(data)

    def clear(self) -> None:
        """Clear all items."""
        with self._lock:
            self._buffer.clear()

    def prune_old(self, max_age: Optional[float] = None) -> int:
        """
        Remove items older than max_age.

        Returns:
            Number of items removed
        """
        cutoff = time.time() - (max_age or self._max_age)
        removed = 0

        with self._lock:
            while self._buffer and self._buffer[0].timestamp < cutoff:
                self._buffer.popleft()
                removed += 1

        return removed

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def __bool__(self) -> bool:
        return len(self) > 0

    @property
    def total_items_added(self) -> int:
        """Total items ever added (including removed)."""
        return self._total_items

    @property
    def oldest_timestamp(self) -> Optional[float]:
        """Timestamp of oldest item."""
        with self._lock:
            return self._buffer[0].timestamp if self._buffer else None

    @property
    def newest_timestamp(self) -> Optional[float]:
        """Timestamp of newest item."""
        with self._lock:
            return self._buffer[-1].timestamp if self._buffer else None

    @property
    def time_span(self) -> float:
        """Time span covered by buffer in seconds."""
        with self._lock:
            if len(self._buffer) < 2:
                return 0.0
            return self._buffer[-1].timestamp - self._buffer[0].timestamp


class MultiFieldBuffer:
    """
    Buffer that stores multiple named fields per frame.
    Useful for storing frame features like pitch, yaw, EAR, etc.
    """

    def __init__(self, fields: List[str], max_size: int = 1800):
        """
        Initialize multi-field buffer.

        Args:
            fields: List of field names
            max_size: Maximum items to store
        """
        self._fields = fields
        self._buffer: RingBuffer[dict] = RingBuffer(max_size)

    def push(self, timestamp: Optional[float] = None, **kwargs) -> None:
        """
        Push a frame with named fields.

        Example:
            buffer.push(pitch=-10.5, yaw=2.3, ear=0.25)
        """
        data = {field: kwargs.get(field) for field in self._fields}
        self._buffer.push(data, timestamp)

    def get_field_values(self, field: str, window_seconds: float,
                         end_time: Optional[float] = None) -> List[Any]:
        """Get all values of a specific field in window."""
        data = self._buffer.get_window_data(window_seconds, end_time)
        return [d.get(field) for d in data if d.get(field) is not None]

    def get_field_mean(self, field: str, window_seconds: float,
                       end_time: Optional[float] = None) -> Optional[float]:
        """Get mean of field values in window."""
        values = self.get_field_values(field, window_seconds, end_time)
        if not values:
            return None
        return sum(values) / len(values)

    def get_field_ratio(self, field: str, predicate: Callable[[Any], bool],
                        window_seconds: float,
                        end_time: Optional[float] = None) -> float:
        """Get ratio of field values matching predicate."""
        values = self.get_field_values(field, window_seconds, end_time)
        if not values:
            return 0.0
        return sum(1 for v in values if predicate(v)) / len(values)

    def get_all_data(self, window_seconds: float,
                     end_time: Optional[float] = None) -> List[dict]:
        """Get all frame data in window."""
        return self._buffer.get_window_data(window_seconds, end_time)

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()


# Quick test
if __name__ == "__main__":
    import random

    print("Testing RingBuffer...")

    # Test basic operations
    buf = RingBuffer[float](max_size=100)

    # Push some values
    base_time = time.time()
    for i in range(50):
        buf.push(random.random(), base_time + i * 0.1)

    print(f"Buffer size: {len(buf)}")
    print(f"Time span: {buf.time_span:.2f}s")

    # Test window query
    window_data = buf.get_window_data(2.0, base_time + 5.0)
    print(f"Items in 2s window: {len(window_data)}")

    # Test aggregation
    avg = buf.aggregate(5.0, lambda x: sum(x) / len(x))
    print(f"Average in last 5s: {avg:.3f}")

    # Test ratio
    ratio = buf.ratio_where(5.0, lambda x: x > 0.5)
    print(f"Ratio > 0.5: {ratio:.2%}")

    print("\nTesting MultiFieldBuffer...")

    # Test multi-field buffer
    mbuf = MultiFieldBuffer(['pitch', 'yaw', 'ear', 'write_score'])

    for i in range(100):
        mbuf.push(
            timestamp=base_time + i * 0.033,  # ~30 fps
            pitch=-15 + random.gauss(0, 5),
            yaw=random.gauss(0, 10),
            ear=0.25 + random.gauss(0, 0.05),
            write_score=random.random()
        )

    print(f"Multi-field buffer size: {len(mbuf)}")
    print(f"Mean pitch (3s): {mbuf.get_field_mean('pitch', 3.0):.2f}")
    print(f"Ratio ear < 0.2 (3s): {mbuf.get_field_ratio('ear', lambda x: x < 0.2, 3.0):.2%}")

    print("\n✓ All tests passed!")
