"""Scientific validation and model-accuracy reporting helpers.

The module is intentionally lightweight and local-only.  It records the
minimum data needed to validate FocusGuardian against self-report,
attention-probe metrics, return-to-work outcomes, and human observer labels.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


APP_STATE_LABELS = [
    "ON_SCREEN_READING",
    "OFFSCREEN_WRITING",
    "PHONE_DISTRACTION",
    "DROWSY_FATIGUE",
    "AWAY",
    "UNCERTAIN",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _timestamp_iso(ts: Optional[float] = None) -> str:
    value = float(ts if ts is not None else time.time())
    return datetime.fromtimestamp(value).isoformat(timespec="seconds")


def _read_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except Exception:
        pass
    return []


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _append_csv(path: Path, fieldnames: Sequence[str], row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    final_fieldnames = list(fieldnames)
    if exists:
        try:
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                current_header = next(reader, [])
            if current_header:
                final_fieldnames = list(current_header)
                for field in fieldnames:
                    if field not in final_fieldnames:
                        final_fieldnames.append(field)
                if final_fieldnames != current_header:
                    rows = _read_csv(path)
                    with open(path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=final_fieldnames, extrasaction="ignore")
                        writer.writeheader()
                        for old_row in rows:
                            writer.writerow({key: old_row.get(key, "") for key in final_fieldnames})
            else:
                exists = False
        except Exception:
            final_fieldnames = list(fieldnames)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in final_fieldnames})


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _rank(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 3 or len(x) != len(y):
        return None
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    if den_x <= 1e-12 or den_y <= 1e-12:
        return None
    return num / (den_x * den_y)


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 3 or len(x) != len(y):
        return None
    return _pearson(_rank(x), _rank(y))


def _mean(values: Sequence[float]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(statistics.fmean(clean))


def _std(values: Sequence[float]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if len(clean) < 2:
        return None
    return float(statistics.stdev(clean))


def _cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> Optional[float]:
    if len(labels_a) != len(labels_b) or not labels_a:
        return None
    total = len(labels_a)
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / total
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    expected = 0.0
    for label in set(count_a) | set(count_b):
        expected += (count_a[label] / total) * (count_b[label] / total)
    if abs(1.0 - expected) <= 1e-12:
        return None
    return (observed - expected) / (1.0 - expected)


class ValidationDataStore:
    """Local store for validation events, predictions, labels and reports."""

    STATE_PREDICTION_FIELDS = [
        "timestamp",
        "timestamp_iso",
        "session_id",
        "profile_name",
        "app_state",
        "raw_state",
        "display_state",
        "confidence",
        "work_readiness",
        "raw_work_readiness",
        "face_present",
        "camera_quality",
        "status_modifier",
        "initial_work_readiness",
        "readiness_delta_from_start",
        "initial_fatigue_index",
        "fatigue_delta_from_start",
        "initial_distraction_risk",
        "distraction_delta_from_start",
        "initial_baseline_quality",
        "reason",
        "elapsed_session_seconds",
    ]

    SCIENCE_EVENT_FIELDS = [
        "timestamp",
        "timestamp_iso",
        "session_id",
        "profile_name",
        "event_type",
        "pre_work_readiness",
        "post_work_readiness",
        "readiness_delta",
        "initial_work_readiness",
        "readiness_delta_from_start",
        "post_readiness_delta_from_start",
        "recovery_to_initial_ratio",
        "initial_fatigue_index",
        "fatigue_delta_from_start",
        "initial_distraction_risk",
        "distraction_delta_from_start",
        "initial_baseline_quality",
        "fatigue_index",
        "distraction_risk",
        "self_report_ready",
        "game_attention_score",
        "attention_stability",
        "accuracy",
        "avg_reaction_time_ms",
        "reaction_variability_ms",
        "omission_errors",
        "commission_errors",
        "return_work_stable_ratio",
        "return_distraction_count",
        "return_drowsy_seconds",
        "return_away_seconds",
        "transfer_score",
        "recovery_success",
        "best_game",
        "weakest_game",
    ]

    OBSERVER_LABEL_FIELDS = [
        "session_id",
        "observer_id",
        "observer_label",
        "timestamp",
        "timestamp_iso",
        "start_timestamp",
        "end_timestamp",
        "note",
    ]

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path("analytics") / "validation"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.state_predictions_path = self.base_dir / "state_predictions.csv"
        self.observer_labels_path = self.base_dir / "observer_labels.csv"
        self.science_events_path = self.base_dir / "scientific_validation_events.json"
        self.science_events_csv_path = self.base_dir / "scientific_validation_events.csv"
        self.legacy_recovery_path = self.base_dir.parent / "focus_reset_recovery_validation.json"
        self.science_report_path = self.base_dir / "scientific_validation_report.json"
        self.model_report_path = self.base_dir / "model_accuracy_report.json"

    def ensure_observer_label_template(self) -> Path:
        if not self.observer_labels_path.exists():
            self.observer_labels_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.observer_labels_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.OBSERVER_LABEL_FIELDS).writeheader()
        return self.observer_labels_path

    def append_state_prediction(self, record: Dict[str, Any]) -> None:
        ts = _safe_float(record.get("timestamp"), time.time())
        row = dict(record)
        row["timestamp"] = f"{ts:.3f}"
        row.setdefault("timestamp_iso", _timestamp_iso(ts))
        _append_csv(self.state_predictions_path, self.STATE_PREDICTION_FIELDS, row)

    def append_scientific_event(self, record: Dict[str, Any]) -> None:
        ts = _safe_float(record.get("timestamp"), time.time())
        row = dict(record)
        row["timestamp"] = float(ts)
        row.setdefault("timestamp_iso", _timestamp_iso(ts))
        rows = _read_json_list(self.science_events_path)
        rows.append(row)
        _write_json(self.science_events_path, rows[-5000:])

        csv_row = dict(row)
        csv_row["timestamp"] = f"{ts:.3f}"
        _append_csv(self.science_events_csv_path, self.SCIENCE_EVENT_FIELDS, csv_row)

    def load_scientific_events(self) -> List[Dict[str, Any]]:
        return _read_json_list(self.science_events_path)

    def load_legacy_recovery_events(self) -> List[Dict[str, Any]]:
        """Load existing Focus Reset validation rows into the common event shape."""
        rows = _read_json_list(self.legacy_recovery_path)
        normalized: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            ts = self._row_timestamp(item, "timestamp")
            if ts is None:
                ts = time.time()
            item["timestamp"] = float(ts)
            item["timestamp_iso"] = str(row.get("timestamp", "") or _timestamp_iso(ts))
            item.setdefault("event_type", "break_recovery")
            item.setdefault("session_id", str(row.get("session_id", "") or ""))
            item.setdefault("profile_name", str(row.get("profile_name", "") or ""))
            if "recovery_success" not in item:
                stable = _safe_float(item.get("return_work_stable_ratio"), 0.0)
                post_wr = _safe_float(item.get("post_work_readiness"), 0.0)
                item["recovery_success"] = bool(stable >= 0.70 and post_wr >= 60.0)
            normalized.append(item)
        return normalized

    def build_scientific_report(self, extra_events: Optional[Iterable[Dict[str, Any]]] = None) -> Dict[str, Any]:
        events = self.load_scientific_events()
        events.extend(self.load_legacy_recovery_events())
        if extra_events:
            events.extend([dict(item) for item in extra_events if isinstance(item, dict)])
        events = self._dedupe_events(events)

        recovery_events = [row for row in events if str(row.get("event_type", "")) in ("break_recovery", "recovery")]
        deltas = [_safe_float(row.get("readiness_delta")) for row in recovery_events]
        transfer = [_safe_float(row.get("transfer_score")) for row in recovery_events]
        success_values = [1.0 if bool(row.get("recovery_success")) else 0.0 for row in recovery_events]

        correlations = self._correlation_table(
            recovery_events,
            pairs=[
                ("self_report_ready", "transfer_score"),
                ("self_report_ready", "return_work_stable_ratio"),
                ("game_attention_score", "transfer_score"),
                ("game_attention_score", "return_work_stable_ratio"),
                ("attention_stability", "return_work_stable_ratio"),
                ("accuracy", "return_work_stable_ratio"),
                ("reaction_variability_ms", "return_work_stable_ratio"),
                ("fatigue_index", "return_work_stable_ratio"),
                ("pre_work_readiness", "post_work_readiness"),
                ("initial_work_readiness", "post_work_readiness"),
                ("recovery_to_initial_ratio", "return_work_stable_ratio"),
            ],
        )

        baseline_events = [
            row for row in recovery_events
            if row.get("initial_work_readiness") not in (None, "")
        ]

        report = {
            "created_at": _timestamp_iso(),
            "n_events": len(events),
            "n_recovery_events": len(recovery_events),
            "n_recovery_events_with_initial_baseline": len(baseline_events),
            "readiness_delta_mean": _mean(deltas),
            "readiness_delta_sd": _std(deltas),
            "transfer_score_mean": _mean(transfer),
            "recovery_success_rate": _mean(success_values),
            "correlations_spearman": correlations,
            "interpretation_note": (
                "These statistics test whether FocusGuardian behavioral signals align with "
                "self-report, attention-probe metrics, and return-to-work outcomes. They are "
                "validation evidence, not proof of direct cognitive-state measurement."
            ),
        }
        _write_json(self.science_report_path, report)
        return report

    @staticmethod
    def _dedupe_events(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        output: List[Dict[str, Any]] = []
        for row in events:
            key = (
                str(row.get("event_type", "")),
                str(row.get("session_id", "")),
                str(row.get("timestamp_iso", "")),
                str(row.get("validated_at", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            output.append(dict(row))
        return output

    @staticmethod
    def _correlation_table(records: Sequence[Dict[str, Any]], pairs: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for x_key, y_key in pairs:
            xs: List[float] = []
            ys: List[float] = []
            for row in records:
                x_raw = row.get(x_key)
                y_raw = row.get(y_key)
                if x_raw in (None, "") or y_raw in (None, ""):
                    continue
                xs.append(_safe_float(x_raw))
                ys.append(_safe_float(y_raw))
            output[f"{x_key}__vs__{y_key}"] = {
                "n": len(xs),
                "spearman_r": spearman(xs, ys),
            }
        return output

    def build_model_accuracy_report(
        self,
        *,
        prediction_path: Optional[Path] = None,
        observer_label_path: Optional[Path] = None,
        tolerance_seconds: float = 1.5,
        labels: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        predictions = _read_csv(prediction_path or self.state_predictions_path)
        observer_rows = _read_csv(observer_label_path or self.observer_labels_path)
        label_order = list(labels or APP_STATE_LABELS)

        aligned = self._align_predictions_to_observer_labels(
            predictions,
            observer_rows,
            tolerance_seconds=max(0.1, float(tolerance_seconds)),
        )
        y_true = [row["observer_label"] for row in aligned]
        y_pred = [row["app_state"] for row in aligned]
        metrics = self._classification_metrics(y_true, y_pred, label_order)
        agreement = self._observer_agreement(observer_rows)

        report = {
            "created_at": _timestamp_iso(),
            "prediction_path": str(prediction_path or self.state_predictions_path),
            "observer_label_path": str(observer_label_path or self.observer_labels_path),
            "n_predictions": len(predictions),
            "n_observer_labels": len(observer_rows),
            "n_aligned_samples": len(aligned),
            "tolerance_seconds": tolerance_seconds,
            "labels": label_order,
            "metrics": metrics,
            "observer_agreement": agreement,
            "aligned_preview": aligned[:20],
        }
        _write_json(self.model_report_path, report)
        return report

    @staticmethod
    def _row_timestamp(row: Dict[str, Any], key: str = "timestamp") -> Optional[float]:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                try:
                    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
                except ValueError:
                    return None
        iso = str(row.get(f"{key}_iso", "") or row.get("timestamp_iso", "") or "").strip()
        if iso:
            try:
                return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return None
        return None

    @classmethod
    def _align_predictions_to_observer_labels(
        cls,
        predictions: Sequence[Dict[str, Any]],
        observer_rows: Sequence[Dict[str, Any]],
        *,
        tolerance_seconds: float,
    ) -> List[Dict[str, Any]]:
        labels_by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in observer_rows:
            label = str(row.get("observer_label", "") or "").strip()
            if not label:
                continue
            session_id = str(row.get("session_id", "") or "").strip()
            prepared = dict(row)
            prepared["_timestamp"] = cls._row_timestamp(row, "timestamp")
            prepared["_start"] = cls._row_timestamp(row, "start_timestamp")
            prepared["_end"] = cls._row_timestamp(row, "end_timestamp")
            labels_by_session[session_id].append(prepared)

        aligned: List[Dict[str, Any]] = []
        for pred in predictions:
            app_state = str(pred.get("app_state", "") or pred.get("display_state", "") or "").strip()
            if not app_state:
                continue
            ts = cls._row_timestamp(pred, "timestamp")
            if ts is None:
                continue
            session_id = str(pred.get("session_id", "") or "").strip()
            candidates = labels_by_session.get(session_id, []) or labels_by_session.get("", [])
            matched_labels: List[str] = []
            for row in candidates:
                start = row.get("_start")
                end = row.get("_end")
                point = row.get("_timestamp")
                if start is not None and end is not None and start <= ts <= end:
                    matched_labels.append(str(row.get("observer_label", "")).strip())
                elif point is not None and abs(ts - float(point)) <= tolerance_seconds:
                    matched_labels.append(str(row.get("observer_label", "")).strip())
            if not matched_labels:
                continue
            label_counts = Counter(matched_labels)
            observer_label = label_counts.most_common(1)[0][0]
            aligned.append(
                {
                    "timestamp": ts,
                    "session_id": session_id,
                    "observer_label": observer_label,
                    "app_state": app_state,
                }
            )
        return aligned

    @staticmethod
    def _classification_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, Any]:
        n = len(y_true)
        confusion = {true: {pred: 0 for pred in labels} for true in labels}
        for true, pred in zip(y_true, y_pred):
            if true not in confusion:
                confusion[true] = {label: 0 for label in labels}
            if pred not in confusion[true]:
                confusion[true][pred] = 0
            confusion[true][pred] += 1

        per_class: Dict[str, Any] = {}
        recalls: List[float] = []
        f1s: List[float] = []
        weighted_f1_sum = 0.0
        total_support = 0
        correct = 0

        all_pred_labels = set(labels) | set(y_pred)
        for label in labels:
            tp = confusion.get(label, {}).get(label, 0)
            fp = sum(confusion.get(other, {}).get(label, 0) for other in confusion if other != label)
            fn = sum(count for pred_label, count in confusion.get(label, {}).items() if pred_label != label)
            support = tp + fn
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / support if support else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            per_class[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
            if support:
                recalls.append(recall)
                f1s.append(f1)
                weighted_f1_sum += f1 * support
                total_support += support
            correct += tp

        # Keep unexpected predicted labels visible in the confusion matrix.
        for pred_label in all_pred_labels:
            for true_label in confusion:
                confusion[true_label].setdefault(pred_label, 0)

        return {
            "accuracy": correct / n if n else 0.0,
            "balanced_accuracy": statistics.fmean(recalls) if recalls else 0.0,
            "macro_f1": statistics.fmean(f1s) if f1s else 0.0,
            "weighted_f1": weighted_f1_sum / total_support if total_support else 0.0,
            "per_class": per_class,
            "confusion_matrix": confusion,
        }

    @classmethod
    def _observer_agreement(cls, observer_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        grouped: Dict[Tuple[str, str, str], Dict[str, str]] = defaultdict(dict)
        for row in observer_rows:
            observer_id = str(row.get("observer_id", "") or "").strip()
            label = str(row.get("observer_label", "") or "").strip()
            if not observer_id or not label:
                continue
            session_id = str(row.get("session_id", "") or "").strip()
            start = str(row.get("start_timestamp", "") or row.get("timestamp", "") or "").strip()
            end = str(row.get("end_timestamp", "") or row.get("timestamp", "") or "").strip()
            grouped[(session_id, start, end)][observer_id] = label

        observers = sorted({obs for row in grouped.values() for obs in row})
        pairwise: Dict[str, Any] = {}
        kappas: List[float] = []
        for i, obs_a in enumerate(observers):
            for obs_b in observers[i + 1 :]:
                labels_a: List[str] = []
                labels_b: List[str] = []
                for row in grouped.values():
                    if obs_a in row and obs_b in row:
                        labels_a.append(row[obs_a])
                        labels_b.append(row[obs_b])
                kappa = _cohen_kappa(labels_a, labels_b)
                key = f"{obs_a}__vs__{obs_b}"
                pairwise[key] = {"n": len(labels_a), "cohen_kappa": kappa}
                if kappa is not None:
                    kappas.append(kappa)

        return {
            "n_observers": len(observers),
            "pairwise": pairwise,
            "mean_pairwise_kappa": statistics.fmean(kappas) if kappas else None,
        }
