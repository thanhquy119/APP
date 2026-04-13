# -*- mode: python ; coding: utf-8 -*-
"""
FocusGuardian PyInstaller Spec File

Build Windows executable with:
    pyinstaller app.spec

Output will be in dist/FocusGuardian/
"""

import sys
from pathlib import Path

# Analysis settings
block_cipher = None

# Get the root directory
ROOT_DIR = Path(SPECPATH)

# Collect data files
datas = []

# Include assets folder (nature sounds, icons, etc.)
assets_dir = ROOT_DIR / 'assets'
if assets_dir.exists():
    datas.append((str(assets_dir), 'assets'))

# Include model files (CRITICAL for MediaPipe Tasks API)
models_dir = ROOT_DIR / 'assets' / 'models'
if models_dir.exists():
    for model_file in models_dir.glob('*.task'):
        datas.append((str(model_file), 'assets/models'))

# Hidden imports that PyInstaller might miss
hiddenimports = [
    # PyQt6
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.sip',

    # MediaPipe Tasks API
    'mediapipe',
    'mediapipe.tasks',
    'mediapipe.tasks.python',
    'mediapipe.tasks.python.vision',
    'mediapipe.tasks.python.vision.face_landmarker',
    'mediapipe.tasks.python.vision.hand_landmarker',
    'mediapipe.tasks.python.core',
    'mediapipe.tasks.python.components',

    # OpenCV
    'cv2',

    # NumPy
    'numpy',

    # Requests (for model download)
    'requests',

    # App modules
    'app',
    'app.vision',
    'app.vision.vision_pipeline',
    'app.vision.face_landmarker',
    'app.vision.hand_landmarker',
    'app.vision.model_manager',
    'app.vision.camera',
    'app.logic',
    'app.logic.focus_engine',
    'app.logic.session_analytics',
    'app.utils',
    'app.utils.ring_buffer',
    'app.utils.win_idle',
    'app.ui',
    'app.ui.main_window',
    'app.ui.settings_dialog',
    'app.ui.mini_games',
    'app.ui.tray',
    'app.focus_reset_game',
    'app.focus_reset_game.config',
    'app.focus_reset_game.models',
    'app.focus_reset_game.game_logic',
    'app.focus_reset_game.game_gonogo',
    'app.focus_reset_game.game_sequence',
    'app.focus_reset_game.game_visual_search',
    'app.focus_reset_game.metrics',
    'app.focus_reset_game.storage',
    'app.focus_reset_game.ui',
    'app.focus_reset_game.ui_v2',
]

# Collect MediaPipe data files (model files, etc.)
try:
    from PyInstaller.utils.hooks import collect_data_files
    mediapipe_datas = collect_data_files('mediapipe')
    datas.extend(mediapipe_datas)
except Exception:
    pass
        'doctest',
        'test',
        'tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out unnecessary files
def filter_binaries(binaries):
    """Filter out unnecessary binary files."""
    exclude_patterns = [
        'api-ms-win',
        'ucrtbase',
        'VCRUNTIME',
    ]
    result = []
    for name, path, type_ in binaries:
        exclude = False
        for pattern in exclude_patterns:
            if pattern.lower() in name.lower():
                exclude = True
                break
        if not exclude:
            result.append((name, path, type_))
    return result

# Don't filter on Windows - these are needed
# a.binaries = filter_binaries(a.binaries)

# PYZ archive
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

# EXE settings
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FocusGuardian',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT_DIR / 'assets' / 'icon.ico') if (ROOT_DIR / 'assets' / 'icon.ico').exists() else None,
    version='version_info.txt' if Path('version_info.txt').exists() else None,
)

# Collect all files
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FocusGuardian'
)
