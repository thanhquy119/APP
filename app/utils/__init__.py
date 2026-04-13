"""
Utils module for FocusGuardian.
"""

from .ring_buffer import RingBuffer, MultiFieldBuffer, TimestampedItem
from .win_idle import WindowsIdleDetector, IdleTracker, get_idle_seconds, is_user_idle

__all__ = [
    'RingBuffer',
    'MultiFieldBuffer',
    'TimestampedItem',
    'WindowsIdleDetector',
    'IdleTracker',
    'get_idle_seconds',
    'is_user_idle'
]
