import sys
import re

with open('app/ui/main_window_backup.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Make sure we add theme stylesheet parsing
if 'from .theme import get_stylesheet' not in text:
    text = text.replace('from .mini_games import MiniGamesWidget', "from .mini_games import MiniGamesWidget\nfrom .theme import get_stylesheet")

# Ensure is_dark_mode = True in init_ui
if 'self.is_dark_mode = True' not in text:
    text = text.replace('def _init_ui(self):\n        """Initialize UI components."""', 'def _init_ui(self):\n        """Initialize UI components."""\n        self.is_dark_mode = True')

# Replace init for FocusScoreWidget
old_init = '''def __init__(self, parent=None):
        super().__init__(parent)
        self.score = 50.0'''
new_init = '''def __init__(self, parent=None):
        super().__init__(parent)
        self.is_dark = True
        self.score = 50.0'''
text = text.replace(old_init, new_init)

# Remove old explicit styles from _init_ui
text = re.sub(r'self\.btn_start\.setStyleSheet\(\"\"\"[\s\S]*?\"\"\"\)', '', text)
text = re.sub(r'self\.btn_break\.setStyleSheet\(\"\"\"[\s\S]*?\"\"\"\)', '', text)

# Set the style using get_stylesheet(True) at the end of _init_ui
if '_apply_theme' not in text:
    text = text.replace('def _init_timers(self):', '''    self._apply_theme()
    
    def _apply_theme(self):
        self.setStyleSheet(get_stylesheet(True))
        
    def _init_timers(self):''')

# If the backup had btn_theme, remove it
text = re.sub(r'\s*self\.btn_theme = QPushButton\(.*?\)\n\s*self\.btn_theme\.setMinimumHeight\(.*?\)\n\s*self\.btn_theme\.clicked\.connect\(.*?\)\n\s*controls\.addWidget\(self\.btn_theme\)', '', text)
text = re.sub(r'\s*def _toggle_theme\(self\):[\s\S]*?def _init_timers\(self\):', '\n\n    def _init_timers(self):', text)

with open('app/ui/main_window.py', 'w', encoding='utf-8') as f:
    f.write(text)
