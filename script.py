with open('app/ui/main_window.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if 'self.btn_theme = QPushButton' in line:
        skip = True
    elif 'def _toggle_theme(self):' in line:
        skip = True
        
    if not skip:
        new_lines.append(line)
        
    if skip and 'controls.addWidget(self.btn_theme)' in line:
        skip = False
    elif skip and 'self._apply_theme()' in line and not 'def _apply_theme' in line:
        # Toggle method ends with self._apply_theme(). Wait for it and a blank line?
        pass

with open('app/ui/main_window.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
