# -*- mode: python ; coding: utf-8 -*-
"""
FocusGuardian PyInstaller one-file spec.

Build Windows standalone executable with:
    pyinstaller --distpath dist-onefile app_onefile.spec
"""

from pathlib import Path

block_cipher = None
ROOT_DIR = Path(SPECPATH)

datas = []

assets_dir = ROOT_DIR / 'assets'
if assets_dir.exists():
    datas.append((str(assets_dir), 'assets'))

models_dir = ROOT_DIR / 'assets' / 'models'
if models_dir.exists():
    for model_file in models_dir.glob('*.task'):
        datas.append((str(model_file), 'assets/models'))

config_file = ROOT_DIR / 'config.json'
if config_file.exists():
    datas.append((str(config_file), '.'))

hiddenimports = [
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtNetwork',
    'PyQt6.QtWidgets',
    'PyQt6.sip',
    'mediapipe',
    'mediapipe.tasks',
    'mediapipe.tasks.python',
    'mediapipe.tasks.python.vision',
    'mediapipe.tasks.python.vision.face_landmarker',
    'mediapipe.tasks.python.vision.hand_landmarker',
    'mediapipe.tasks.python.core',
    'mediapipe.tasks.python.components',
    'cv2',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends.backend_agg',
    'numpy',
    'requests',
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
    'app.logic.supabase_sync',
    'app.logic.supabase_user_store',
    'app.logic.cloud_payloads',
    'app.logic.personalization',
    'app.utils',
    'app.utils.ring_buffer',
    'app.utils.win_idle',
    'app.ui',
    'app.ui.main_window',
    'app.ui.settings_dialog',
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

try:
    from PyInstaller.utils.hooks import collect_data_files

    datas.extend(collect_data_files('mediapipe'))
    datas.extend(collect_data_files('matplotlib'))
except Exception:
    pass

a = Analysis(
    ['main.py'],
    pathex=[str(ROOT_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'IPython',
        'jupyter',
        'notebook',
        'doctest',
        'test',
        'tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FocusGuardian',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT_DIR / 'assets' / 'icon.ico') if (ROOT_DIR / 'assets' / 'icon.ico').exists() else None,
    version='version_info.txt' if Path('version_info.txt').exists() else None,
)
