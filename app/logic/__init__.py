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
from .task_context import (
    TaskContextConfig,
    TaskContextSample,
    TaskContextStats,
    TaskContextMonitor,
    TaskContextClassifier,
)
from .auth import UserAccount, CurrentUserSession
from .auth_manager import AuthManager, AuthResult
from .supabase_user_store import SupabaseUserStore

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
    'TaskContextConfig',
    'TaskContextSample',
    'TaskContextStats',
    'TaskContextMonitor',
    'TaskContextClassifier',
    'UserAccount',
    'CurrentUserSession',
    'AuthManager',
    'AuthResult',
    'SupabaseUserStore',
]
