"""Generate FocusGuardian scientific validation and model-accuracy reports.

Usage:
    .venv\\Scripts\\python.exe tools\\validation_report.py
    .venv\\Scripts\\python.exe tools\\validation_report.py --observer-labels analytics\\validation\\observer_labels.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.logic.scientific_validation import ValidationDataStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FocusGuardian validation reports.")
    parser.add_argument("--base-dir", default="analytics/validation", help="Validation data directory.")
    parser.add_argument("--predictions", default="", help="Optional state_predictions.csv path.")
    parser.add_argument("--observer-labels", default="", help="Optional observer_labels.csv path.")
    parser.add_argument("--tolerance", type=float, default=1.5, help="Timestamp alignment tolerance in seconds.")
    parser.add_argument("--print-json", action="store_true", help="Print full reports as JSON.")
    args = parser.parse_args()

    store = ValidationDataStore(Path(args.base_dir))
    label_template = store.ensure_observer_label_template()

    science_report = store.build_scientific_report()
    model_report = store.build_model_accuracy_report(
        prediction_path=Path(args.predictions) if args.predictions else None,
        observer_label_path=Path(args.observer_labels) if args.observer_labels else None,
        tolerance_seconds=args.tolerance,
    )

    if args.print_json:
        print(json.dumps({"scientific": science_report, "model_accuracy": model_report}, ensure_ascii=False, indent=2))
    else:
        print("Scientific validation report:", store.science_report_path)
        print("Model accuracy report:", store.model_report_path)
        print("Observer label template:", label_template)
        print("Recovery events:", science_report.get("n_recovery_events", 0))
        print("Recovery events with initial baseline:", science_report.get("n_recovery_events_with_initial_baseline", 0))
        print("Aligned model samples:", model_report.get("n_aligned_samples", 0))
        print("Balanced accuracy:", model_report.get("metrics", {}).get("balanced_accuracy", 0.0))
        print("Macro F1:", model_report.get("metrics", {}).get("macro_f1", 0.0))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
