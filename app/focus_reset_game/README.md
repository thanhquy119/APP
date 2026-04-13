# Focus Reset Game

Focused short-session attention recovery module for FocusGuardian.

## Why this implementation

This module uses **PyQt6** (same stack as the existing app) for:
- zero extra GUI runtime dependencies in this repository,
- consistent visual style and integration with existing dialogs,
- easier maintenance vs mixing multiple Qt bindings.

PyQt6 APIs used here are close to PySide6 equivalents.

## Features

- Main menu: Start Session, Instructions, History, Exit
- Baseline test (20-30s)
- 3 Go/No-Go rounds with target/no-go probabilities
- Micro-break with breathing animation between rounds
- Final metrics dashboard:
  - average reaction time
  - reaction time variability
  - accuracy
  - commission errors
  - omission errors
  - focus stability score
  - baseline comparison (Better/Similar/Worse)
- JSON history persistence and CSV export

## Run standalone

From project root:

```bash
python -m app.focus_reset_game.main
```

## Integration in FocusGuardian

This module is wired into the break dialog (`app/ui/mini_games.py`) and replaces the old visual reset slot with **Focus Reset Game**.

## Config knobs

Edit `app/focus_reset_game/config.py`:

- `round_count`
- `baseline_duration_s`
- `round_duration_s`
- `break_duration_s`
- `stimulus_duration_ms`
- `inter_stimulus_ms`
- `target_probability`
- breathing cadence (`inhale_seconds`, `exhale_seconds`)

## Data storage

Session history is stored at:

- `analytics/focus_reset_history.json`

CSV export can be triggered from the History page.
