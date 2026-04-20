"""
Logic module for FocusGuardian.
"""

from .focus_engine import (
    FocusEngine,
    FocusEngineConfig,
    FocusState,
    FrameFeatures,
    WindowStats,
    StateTransition,
    create_frame_features
)
from .session_analytics import SessionAnalyticsStore
from .personalization import (
    UserBaseline,
    PersonalizedThresholds,
    UserBaselineStore,
    PersonalizationManager,
)
from .zalo_bot import ZaloBotConfig, ZaloBotClient
from .zalo_alerts import ZaloAlertManager, ZaloAlertEvent
from .auth import UserAccount, CurrentUserSession
from .auth_manager import AuthManager, AuthResult
from .user_store import GoogleSheetsUserStore

__all__ = [
    'FocusEngine',
    'FocusEngineConfig',
    'FocusState',
    'FrameFeatures',
    'WindowStats',
    'StateTransition',
    'create_frame_features',
    'SessionAnalyticsStore',
    'UserBaseline',
    'PersonalizedThresholds',
    'UserBaselineStore',
    'PersonalizationManager',
    'ZaloBotConfig',
    'ZaloBotClient',
    'ZaloAlertManager',
    'ZaloAlertEvent',
    'UserAccount',
    'CurrentUserSession',
    'AuthManager',
    'AuthResult',
    'GoogleSheetsUserStore',
]
