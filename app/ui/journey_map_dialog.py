"""Large Focus Journey map dialog.

This module keeps the optional web map isolated from MainWindow so the main
tracking UI can continue to fall back to the symbolic PyQt map when WebEngine is
not available.
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkDiskCache, QNetworkReply, QNetworkRequest
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

QWebEngineView = None
WEBENGINE_AVAILABLE = False


def _resolve_webengine_view():
    """Import QtWebEngine lazily for the optional Leaflet renderer."""
    global QWebEngineView, WEBENGINE_AVAILABLE
    if QWebEngineView is not None:
        WEBENGINE_AVAILABLE = True
        return QWebEngineView
    try:  # QtWebEngine is optional in some local installs.
        from PyQt6.QtWebEngineWidgets import QWebEngineView as WebEngineView

        QWebEngineView = WebEngineView
        WEBENGINE_AVAILABLE = True
        return QWebEngineView
    except Exception as exc:  # pragma: no cover - depends on local Qt packaging.
        WEBENGINE_AVAILABLE = False
        logger.warning("QtWebEngine unavailable, native satellite tile renderer remains available: %s", exc)
        return None


AIRPORT_COORDS: Dict[str, Dict[str, Any]] = {
    "DAD": {"name": "Da Nang", "lat": 16.0439, "lng": 108.1994},
    "SGN": {"name": "Ho Chi Minh City", "lat": 10.8188, "lng": 106.6519},
    "HAN": {"name": "Ha Noi", "lat": 21.2212, "lng": 105.8072},
    "HUI": {"name": "Hue", "lat": 16.4015, "lng": 107.7031},
    "CXR": {"name": "Cam Ranh", "lat": 12.2275, "lng": 109.1922},
    "DLI": {"name": "Da Lat", "lat": 11.7500, "lng": 108.3670},
    "VCA": {"name": "Can Tho", "lat": 10.0851, "lng": 105.7119},
    "PQC": {"name": "Phu Quoc", "lat": 10.1698, "lng": 103.9931},
    "BMV": {"name": "Buon Ma Thuot", "lat": 12.6683, "lng": 108.1203},
    "VII": {"name": "Vinh", "lat": 18.7376, "lng": 105.6708},
    "VCL": {"name": "Chu Lai", "lat": 15.4033, "lng": 108.7060},
    "BKK": {"name": "Bangkok", "lat": 13.6900, "lng": 100.7501},
    "SIN": {"name": "Singapore", "lat": 1.3644, "lng": 103.9915},
    "KUL": {"name": "Kuala Lumpur", "lat": 2.7456, "lng": 101.7072},
    "PNH": {"name": "Phnom Penh", "lat": 11.5466, "lng": 104.8441},
    "VTE": {"name": "Vientiane", "lat": 17.9883, "lng": 102.5633},
    "REP": {"name": "Siem Reap", "lat": 13.4107, "lng": 103.8128},
}


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def _haversine_km(origin: Dict[str, Any], destination: Dict[str, Any]) -> int:
    lat1 = math.radians(float(origin["lat"]))
    lon1 = math.radians(float(origin["lng"]))
    lat2 = math.radians(float(destination["lat"]))
    lon2 = math.radians(float(destination["lng"]))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return int(round(6371.0 * 2.0 * math.asin(math.sqrt(h))))


def _curve_bias(pair_key: str) -> float:
    seed = sum(ord(ch) for ch in pair_key or "FG")
    return (0.18 + (seed % 8) / 100.0) * (-1 if seed % 2 else 1)


def _curve_points(
    origin: Dict[str, Any],
    destination: Dict[str, Any],
    *,
    bias: float,
    steps: int = 72,
) -> List[List[float]]:
    lat1, lng1 = float(origin["lat"]), float(origin["lng"])
    lat2, lng2 = float(destination["lat"]), float(destination["lng"])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    length = max(0.0001, math.sqrt(dlat * dlat + dlng * dlng))
    normal_lat = -dlng / length
    normal_lng = dlat / length
    bow = max(-0.32, min(0.32, float(bias or 0.18))) * length

    points: List[List[float]] = []
    for i in range(max(2, steps) + 1):
        t = i / max(1, steps)
        lift = math.sin(math.pi * t) * bow
        lat = lat1 + dlat * t + normal_lat * lift
        lng = lng1 + dlng * t + normal_lng * lift
        points.append([round(lat, 6), round(lng, 6)])
    return points


def build_journey_model(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    data = dict(payload or {})
    from_code = str(data.get("route_from_code") or data.get("from_code") or "DAD").strip().upper() or "DAD"
    to_code = str(data.get("route_to_code") or data.get("to_code") or "SGN").strip().upper() or "SGN"
    origin = dict(AIRPORT_COORDS.get(from_code) or AIRPORT_COORDS["DAD"])
    destination = dict(AIRPORT_COORDS.get(to_code) or AIRPORT_COORDS["SGN"])
    origin["code"] = from_code
    destination["code"] = to_code

    duration = _safe_int(
        data.get("route_duration_minutes")
        or data.get("duration_minutes")
        or data.get("planned_minutes")
        or data.get("deadline_minutes"),
        25,
    )
    computed_distance = _haversine_km(origin, destination)
    distance = _safe_int(data.get("route_distance_km") or data.get("distance_km"), computed_distance)
    pair_key = f"{from_code}-{to_code}"
    bias = float(data.get("curve") or _curve_bias(pair_key))

    return {
        "from_code": from_code,
        "to_code": to_code,
        "from_name": str(data.get("route_from_name") or data.get("from_name") or origin.get("name") or from_code),
        "to_name": str(data.get("route_to_name") or data.get("to_name") or destination.get("name") or to_code),
        "duration_minutes": max(1, duration),
        "distance_km": max(1, distance),
        "origin": origin,
        "destination": destination,
        "curve_points": _curve_points(origin, destination, bias=bias),
    }


class LeafletJourneyMapWidget(QWidget):
    """Leaflet satellite map with a route-following flight camera."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._model = build_journey_model({})
        self._progress = 0.0
        self._remaining_seconds = int(self._model["duration_minutes"]) * 60
        self._distance_left_km = int(self._model["distance_km"])
        self._phase = "Boarding"
        self._ready = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        web_view_class = _resolve_webengine_view()
        if web_view_class is None:
            raise RuntimeError("QtWebEngine is not available")
        self.view = web_view_class(self)
        layout.addWidget(self.view)
        self.view.loadFinished.connect(self._on_load_finished)

    def set_journey_data(self, data: Dict[str, Any]) -> None:
        self._model = build_journey_model(data)
        if self._remaining_seconds <= 0:
            self._remaining_seconds = int(self._model["duration_minutes"]) * 60
        self._distance_left_km = int(round(float(self._model["distance_km"]) * (1.0 - self._progress)))
        self._ready = False
        self.view.setHtml(self._html(), QUrl("https://focusguardian.local/"))

    def update_progress(
        self,
        progress: float,
        remaining_seconds: int,
        distance_left_km: int,
        phase: str = "",
    ) -> None:
        self._progress = max(0.0, min(1.0, float(progress or 0.0)))
        self._remaining_seconds = max(0, int(remaining_seconds or 0))
        self._distance_left_km = max(0, int(distance_left_km or 0))
        self._phase = str(phase or self._phase or "")
        self._push_progress()

    def _on_load_finished(self, ok: bool) -> None:
        self._ready = bool(ok)
        self._push_progress()

    def _remaining_text(self) -> str:
        minutes = int(math.ceil(max(0, self._remaining_seconds) / 60.0))
        return f"{minutes} min"

    def _push_progress(self) -> None:
        if not self._ready:
            return
        args = json.dumps(
            [
                self._progress,
                self._remaining_text(),
                f"{self._distance_left_km} km",
                self._phase,
            ]
        )
        self.view.page().runJavaScript(f"window.updateJourneyProgress.apply(window, {args});")

    def _html(self) -> str:
        model_json = json.dumps(self._model, ensure_ascii=True)
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body, #map {{ height: 100%; width: 100%; margin: 0; background: #07111d; overflow: hidden; }}
    body {{ font-family: "Segoe UI", Inter, Arial, sans-serif; }}
    #map {{
      filter: saturate(0.88) contrast(1.04) brightness(0.88);
    }}
    .focus-vignette {{
      position: absolute; inset: 0; z-index: 600; pointer-events: none;
      background:
        linear-gradient(180deg, rgba(5,12,20,0.24), rgba(5,12,20,0.01) 34%, rgba(5,12,20,0.50)),
        radial-gradient(circle at 50% 44%, rgba(255,255,255,0.06), rgba(5,12,20,0.00) 42%);
    }}
    .flight-corridor {{
      display: none;
    }}
    .route-badge {{
      position: absolute; left: 50%; top: 22px; transform: translateX(-50%);
      z-index: 650; color: #f6fbff; background: rgba(5, 13, 23, 0.72);
      border: 1px solid rgba(130, 170, 205, 0.24); border-radius: 999px;
      padding: 9px 16px; letter-spacing: 0.02em; font-weight: 700;
      box-shadow: 0 14px 34px rgba(0,0,0,0.28); backdrop-filter: blur(14px);
    }}
    .phase-badge {{
      margin-left: 10px; color: #74f0dd; font-weight: 600; opacity: 0.92;
    }}
    .metric {{
      position: absolute; bottom: 70px; z-index: 650; color: #f8fbff;
      text-shadow: 0 3px 18px rgba(0,0,0,0.62); pointer-events: none;
    }}
    .metric.left {{ left: 30px; }}
    .metric.right {{ right: 34px; text-align: right; }}
    .metric .label {{ display: block; color: rgba(246,251,255,0.72); font-size: 15px; font-weight: 600; }}
    .metric .value {{ display: block; margin-top: 3px; font-size: 40px; line-height: 1.12; font-weight: 760; letter-spacing: 0; }}
    .airport-pin {{
      width: 42px; height: 34px; transform: translate(-50%, -50%);
      display: grid; place-items: center; color: #f9fdff; font-size: 11px; font-weight: 800;
      background: rgba(2, 10, 18, 0.82); border: 2px solid rgba(117, 236, 222, 0.86);
      border-radius: 12px; box-shadow: 0 8px 20px rgba(0,0,0,0.35);
    }}
    .airport-pin.destination {{ border-color: #ffd64d; color: #ffe577; }}
    .plane-shell {{
      width: 54px; height: 54px; display: grid; place-items: center; border-radius: 999px;
      background: radial-gradient(circle, rgba(255,255,255,0.22), rgba(117,236,222,0.08) 42%, rgba(117,236,222,0));
    }}
    .plane-arrow {{
      width: 0; height: 0; border-top: 8px solid transparent; border-bottom: 8px solid transparent;
      border-left: 24px solid #f8ffff; transform-origin: 9px 8px;
      filter: drop-shadow(0 2px 1px rgba(3,10,18,0.86)) drop-shadow(0 0 12px rgba(255,255,255,0.48));
    }}
    .leaflet-control-attribution {{
      background: rgba(5, 13, 23, 0.44) !important; color: rgba(255,255,255,0.62) !important;
      border-radius: 8px 8px 0 0; font-size: 10px;
    }}
    .leaflet-control-attribution a {{ color: rgba(255,255,255,0.78) !important; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="focus-vignette"></div>
  <div class="flight-corridor"></div>
  <div class="route-badge"><span id="routeText"></span><span id="phaseText" class="phase-badge"></span></div>
  <div class="metric left"><span class="label">Time Remaining</span><span id="remainingText" class="value"></span></div>
  <div class="metric right"><span class="label">Distance Remaining</span><span id="distanceText" class="value"></span></div>
  <script>
    const journeyModel = {model_json};
    const map = L.map('map', {{
      zoomControl: false,
      attributionControl: true,
      scrollWheelZoom: true,
      doubleClickZoom: true,
      zoomSnap: 0.25,
      zoomDelta: 0.5,
      preferCanvas: true
    }});
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      maxZoom: 18,
      attribution: 'Tiles &copy; Esri'
    }}).addTo(map);

    const routePoints = journeyModel.curve_points.map(p => [p[0], p[1]]);
    const baseLine = L.polyline(routePoints, {{
      color: 'rgba(8, 14, 22, 0.72)', weight: 6, opacity: 0.12, lineCap: 'round'
    }}).addTo(map);
    const routeLine = L.polyline(routePoints, {{
      color: 'rgba(236, 248, 255, 0.82)', weight: 2, opacity: 0.20, lineCap: 'round'
    }}).addTo(map);
    const activeLine = L.polyline([routePoints[0]], {{
      color: '#f7ffff', weight: 2.5, opacity: 0.30, lineCap: 'round'
    }}).addTo(map);

    function airportIcon(code, destination) {{
      return L.divIcon({{
        className: '',
        html: `<div class="airport-pin ${{destination ? 'destination' : ''}}">${{code}}</div>`,
        iconSize: [42, 34],
        iconAnchor: [21, 17]
      }});
    }}
    const originMarker = L.marker(
      [journeyModel.origin.lat, journeyModel.origin.lng],
      {{ icon: airportIcon(journeyModel.from_code, false), interactive: false }}
    ).addTo(map);
    const destMarker = L.marker(
      [journeyModel.destination.lat, journeyModel.destination.lng],
      {{ icon: airportIcon(journeyModel.to_code, true), interactive: false }}
    ).addTo(map);
    const planeIcon = L.divIcon({{
      className: '',
      html: '<div class="plane-shell"><div class="plane-arrow"></div></div>',
      iconSize: [54, 54],
      iconAnchor: [27, 27]
    }});
    const planeMarker = L.marker(routePoints[0], {{ icon: planeIcon, interactive: false, zIndexOffset: 500 }}).addTo(map);

    const bounds = L.latLngBounds(routePoints);
    const flightZoom = journeyModel.distance_km <= 240 ? 14.25 : (journeyModel.distance_km <= 620 ? 13.25 : 12.25);
    map.setView(routePoints[0], flightZoom, {{ animate: false }});
    document.getElementById('routeText').innerHTML = `${{journeyModel.from_code}} &rarr; ${{journeyModel.to_code}}`;

    function pointAt(progress) {{
      const p = Math.max(0, Math.min(1, Number(progress || 0)));
      const raw = p * (routePoints.length - 1);
      const index = Math.floor(raw);
      const next = Math.min(routePoints.length - 1, index + 1);
      const mix = raw - index;
      const a = routePoints[index];
      const b = routePoints[next];
      return [a[0] + (b[0] - a[0]) * mix, a[1] + (b[1] - a[1]) * mix, index, next];
    }}

    function updatePlaneRotation(index, next) {{
      const a = routePoints[index];
      const b = routePoints[next];
      const angle = Math.atan2(b[0] - a[0], b[1] - a[1]) * 180 / Math.PI;
      const marker = planeMarker.getElement();
      if (!marker) return;
      const arrow = marker.querySelector('.plane-arrow');
      if (arrow) arrow.style.transform = `rotate(${{90 - angle}}deg)`;
    }}

    window.updateJourneyProgress = function(progress, remainingText, distanceText, phaseText) {{
      const pos = pointAt(progress);
      planeMarker.setLatLng([pos[0], pos[1]]);
      updatePlaneRotation(pos[2], pos[3]);
      map.panTo([pos[0], pos[1]], {{ animate: true, duration: 0.85, easeLinearity: 0.18 }});
      const activeCount = Math.max(2, Math.ceil(Math.max(0, Math.min(1, progress || 0)) * (routePoints.length - 1)) + 1);
      activeLine.setLatLngs(routePoints.slice(0, activeCount));
      document.getElementById('remainingText').textContent = remainingText || '0 min';
      document.getElementById('distanceText').textContent = distanceText || '0 km';
      document.getElementById('phaseText').textContent = phaseText ? ` ${phaseText}` : '';
    }};
    window.updateJourneyProgress(0, `${{journeyModel.duration_minutes}} min`, `${{journeyModel.distance_km}} km`, 'Boarding');
  </script>
</body>
</html>"""


class FallbackJourneyMapWidget(QWidget):
    """Native PyQt satellite tile map used by the large journey dialog."""

    TILE_SIZE = 256

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._model = build_journey_model({})
        self._progress = 0.0
        self._target_progress = 0.0
        self._display_progress = 0.0
        self._anim_start_progress = 0.0
        self._anim_started_at = time.monotonic()
        self._anim_duration = 1.0
        self._progress_anchor = 0.0
        self._progress_anchor_at = time.monotonic()
        self._progress_rate_per_second = 0.0
        self._motion_paused = False
        self._last_progress_update_at: Optional[float] = None
        self._last_anim_tick_at = time.monotonic()
        self._last_repaint_at = 0.0
        self._target_frame_interval = 1.0 / 30.0
        self._tile_update_pending = False
        self._remaining_seconds = int(self._model["duration_minutes"]) * 60
        self._distance_left_km = int(self._model["distance_km"])
        self._phase = "Boarding"
        self._min_zoom = 9
        self._max_zoom = 17
        self._zoom = self._zoom_for_distance(int(self._model["distance_km"]))
        self._center_lat, self._center_lng = self._point_at_progress(self._display_progress)
        self._tile_cache: Dict[tuple[int, int, int], QPixmap] = {}
        self._pending_tiles: set[tuple[int, int, int]] = set()
        self._world_points_cache: Dict[int, List[QPointF]] = {}
        self._last_progress_signature: Optional[tuple] = None
        self._network = QNetworkAccessManager(self)
        self._install_tile_disk_cache()
        self._network.finished.connect(self._on_tile_finished)
        self._camera_timer = QTimer(self)
        self._camera_timer.setInterval(33)
        self._camera_timer.timeout.connect(self._tick_camera_animation)
        self._camera_timer.start()
        self.setMinimumSize(840, 540)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def _install_tile_disk_cache(self) -> None:
        try:
            cache_dir = Path(__file__).resolve().parents[2] / "analytics" / "journey_tile_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            disk_cache = QNetworkDiskCache(self)
            disk_cache.setCacheDirectory(str(cache_dir))
            disk_cache.setMaximumCacheSize(96 * 1024 * 1024)
            self._network.setCache(disk_cache)
            self._tile_disk_cache = disk_cache
        except Exception as exc:
            self._tile_disk_cache = None
            logger.debug("Journey tile disk cache unavailable: %s", exc)

    def set_journey_data(self, data: Dict[str, Any]) -> None:
        self._model = build_journey_model(data)
        self._progress = 0.0
        self._zoom = self._zoom_for_distance(int(self._model["distance_km"]))
        self._target_progress = 0.0
        self._display_progress = 0.0
        self._anim_start_progress = self._display_progress
        self._anim_started_at = time.monotonic()
        self._anim_duration = 1.0
        self._progress_anchor = 0.0
        self._progress_anchor_at = time.monotonic()
        self._progress_rate_per_second = 0.0
        self._motion_paused = False
        self._last_progress_update_at = None
        self._last_anim_tick_at = time.monotonic()
        self._last_repaint_at = 0.0
        self._last_progress_signature = None
        self._world_points_cache.clear()
        self._center_lat, self._center_lng = self._point_at_progress(self._display_progress)
        self._remaining_seconds = int(self._model["duration_minutes"]) * 60
        self._distance_left_km = int(self._model["distance_km"])
        self._schedule_tile_update()
        self.update()

    def update_progress(
        self,
        progress: float,
        remaining_seconds: int,
        distance_left_km: int,
        phase: str = "",
    ) -> None:
        now = time.monotonic()
        new_progress = max(0.0, min(1.0, float(progress or 0.0)))
        new_remaining = max(0, int(remaining_seconds or 0))
        new_distance_left = max(0, int(distance_left_km or 0))
        new_phase = str(phase or self._phase or "")
        signature = (round(new_progress, 5), new_remaining, new_distance_left, new_phase)
        if signature == self._last_progress_signature:
            return
        self._last_progress_signature = signature
        first_update = self._last_progress_update_at is None
        self._last_progress_update_at = now
        self._progress = new_progress
        if first_update:
            self._display_progress = new_progress

        total_seconds = max(1.0, float(self._model.get("duration_minutes", 25) or 25) * 60.0)
        self._progress_anchor = new_progress
        self._progress_anchor_at = now
        self._progress_rate_per_second = 0.0 if self._motion_paused or new_progress >= 1.0 else 1.0 / total_seconds
        self._target_progress = new_progress
        if new_progress < self._display_progress - 0.03:
            self._display_progress = new_progress

        self._anim_start_progress = self._display_progress
        self._anim_started_at = now
        self._anim_duration = 1.0
        self._remaining_seconds = new_remaining
        self._distance_left_km = new_distance_left
        self._phase = new_phase
        if self.isVisible() and not self._camera_timer.isActive():
            self._camera_timer.start()

    def set_motion_paused(self, paused: bool) -> None:
        now = time.monotonic()
        next_paused = bool(paused)
        if self._motion_paused == next_paused and self._camera_timer.isActive() == (not next_paused):
            return
        self._motion_paused = next_paused
        self._progress_anchor = self._display_progress
        self._progress_anchor_at = now
        total_seconds = max(1.0, float(self._model.get("duration_minutes", 25) or 25) * 60.0)
        self._progress_rate_per_second = 0.0 if self._motion_paused else 1.0 / total_seconds
        if self._motion_paused:
            if abs(self._target_progress - self._display_progress) <= 0.00001:
                self._camera_timer.stop()
        elif self.isVisible() and not self._camera_timer.isActive():
            self._camera_timer.start()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._motion_paused and not self._camera_timer.isActive():
            self._last_anim_tick_at = time.monotonic()
            self._camera_timer.start()

    def hideEvent(self, event) -> None:
        self._camera_timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
        grad.setColorAt(0.0, QColor("#0c1b2b"))
        grad.setColorAt(0.55, QColor("#15324a"))
        grad.setColorAt(1.0, QColor("#09131f"))
        painter.fillRect(rect, QBrush(grad))
        self._draw_satellite_tiles(painter, rect)
        self._draw_vignette(painter, rect)
        self._draw_route(painter, rect)
        self._draw_overlays(painter, rect)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        self.change_zoom(1 if delta > 0 else -1)
        event.accept()

    def change_zoom(self, delta: int) -> None:
        new_zoom = max(self._min_zoom, min(self._max_zoom, int(self._zoom) + int(delta)))
        if new_zoom == self._zoom:
            return
        self._zoom = new_zoom
        self.update()

    def zoom_in(self) -> None:
        self.change_zoom(1)

    def zoom_out(self) -> None:
        self.change_zoom(-1)

    def _tick_camera_animation(self) -> None:
        if not self.isVisible():
            self._camera_timer.stop()
            return
        now = time.monotonic()
        dt = max(0.0, min(0.09, now - self._last_anim_tick_at))
        self._last_anim_tick_at = now
        live_target = min(1.0, self._progress_anchor + max(0.0, now - self._progress_anchor_at) * self._progress_rate_per_second)
        if live_target < self._display_progress - 0.03:
            self._display_progress = live_target
        else:
            alpha = 1.0 - math.pow(0.925, max(1.0, dt / (1.0 / 60.0)))
            self._display_progress += (live_target - self._display_progress) * alpha
        self._center_lat, self._center_lng = self._point_at_progress(self._display_progress)
        should_repaint = abs(live_target - self._display_progress) > 0.000001
        if should_repaint and now - self._last_repaint_at >= self._target_frame_interval:
            self._last_repaint_at = now
            self.update()
        if self._motion_paused and not should_repaint:
            self._camera_timer.stop()

    @staticmethod
    def _zoom_for_distance(distance_km: int) -> int:
        if distance_km <= 180:
            return 14
        if distance_km <= 520:
            return 13
        return 12

    def _point_at_progress(self, progress: float) -> tuple[float, float]:
        points = self._model["curve_points"]
        if not points:
            origin = self._model["origin"]
            return float(origin["lat"]), float(origin["lng"])
        p = max(0.0, min(1.0, float(progress or 0.0)))
        raw = p * (len(points) - 1)
        index = int(math.floor(raw))
        next_index = min(len(points) - 1, index + 1)
        mix = raw - index
        a = points[index]
        b = points[next_index]
        lat = float(a[0]) + (float(b[0]) - float(a[0])) * mix
        lng = float(a[1]) + (float(b[1]) - float(a[1])) * mix
        return lat, lng

    @classmethod
    def _latlng_to_world(cls, lat: float, lng: float, zoom: int) -> tuple[float, float]:
        lat = max(-85.05112878, min(85.05112878, float(lat)))
        lng = float(lng)
        scale = cls.TILE_SIZE * (2 ** int(zoom))
        sin_lat = math.sin(math.radians(lat))
        x = (lng + 180.0) / 360.0 * scale
        y = (0.5 - math.log((1.0 + sin_lat) / max(1e-12, 1.0 - sin_lat)) / (4.0 * math.pi)) * scale
        return x, y

    def _project(self, lat: float, lng: float, rect: QRectF) -> QPointF:
        center_x, center_y = self._latlng_to_world(self._center_lat, self._center_lng, self._zoom)
        world_x, world_y = self._latlng_to_world(lat, lng, self._zoom)
        return QPointF(
            rect.center().x() + (world_x - center_x),
            rect.center().y() + (world_y - center_y),
        )

    def _world_points(self) -> List[QPointF]:
        points = self._world_points_cache.get(self._zoom)
        if points is not None:
            return points
        points = [
            QPointF(*self._latlng_to_world(float(point[0]), float(point[1]), self._zoom))
            for point in self._model["curve_points"]
        ]
        self._world_points_cache[self._zoom] = points
        return points

    @staticmethod
    def _world_point_at_progress(points: List[QPointF], progress: float) -> QPointF:
        if not points:
            return QPointF(0, 0)
        p = max(0.0, min(1.0, float(progress or 0.0)))
        raw = p * (len(points) - 1)
        index = int(math.floor(raw))
        next_index = min(len(points) - 1, index + 1)
        mix = raw - index
        a = points[index]
        b = points[next_index]
        return QPointF(a.x() + (b.x() - a.x()) * mix, a.y() + (b.y() - a.y()) * mix)

    @staticmethod
    def _project_world(point: QPointF, center_x: float, center_y: float, rect: QRectF) -> QPointF:
        return QPointF(
            rect.center().x() + (point.x() - center_x),
            rect.center().y() + (point.y() - center_y),
        )

    def _draw_satellite_tiles(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        center_x, center_y = self._latlng_to_world(self._center_lat, self._center_lng, self._zoom)
        top_left_x = center_x - rect.width() / 2.0
        top_left_y = center_y - rect.height() / 2.0
        start_x = math.floor(top_left_x / self.TILE_SIZE)
        end_x = math.floor((top_left_x + rect.width()) / self.TILE_SIZE)
        start_y = math.floor(top_left_y / self.TILE_SIZE)
        end_y = math.floor((top_left_y + rect.height()) / self.TILE_SIZE)
        max_tile = 2 ** self._zoom
        painted_any = False
        tile_jobs: List[tuple[float, int, int, tuple[int, int, int], QRectF]] = []
        center_tile_x = center_x / self.TILE_SIZE
        center_tile_y = center_y / self.TILE_SIZE

        for tile_x in range(start_x - 1, end_x + 2):
            for tile_y in range(start_y - 1, end_y + 2):
                if tile_y < 0 or tile_y >= max_tile:
                    continue
                wrapped_x = tile_x % max_tile
                key = (self._zoom, wrapped_x, tile_y)
                dest = QRectF(
                    rect.left() + tile_x * self.TILE_SIZE - top_left_x,
                    rect.top() + tile_y * self.TILE_SIZE - top_left_y,
                    self.TILE_SIZE,
                    self.TILE_SIZE,
                )
                dist = (tile_x - center_tile_x) ** 2 + (tile_y - center_tile_y) ** 2
                tile_jobs.append((dist, tile_x, tile_y, key, dest))

        requested_this_paint = 0
        for _dist, _tile_x, _tile_y, key, dest in sorted(tile_jobs, key=lambda item: item[0]):
            pixmap = self._tile_cache.get(key)
            if pixmap is None:
                if requested_this_paint < 10:
                    if self._request_tile(key):
                        requested_this_paint += 1
                continue
            painter.drawPixmap(dest.adjusted(-1.0, -1.0, 1.0, 1.0), pixmap, QRectF(pixmap.rect()))
            painted_any = True

        if not painted_any:
            painter.setPen(QPen(QColor(255, 255, 255, 22), 1))
            for i in range(1, 5):
                x = rect.left() + rect.width() * i / 5.0
                painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
                y = rect.top() + rect.height() * i / 5.0
                painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        painter.restore()

    def _request_tile(self, key: tuple[int, int, int]) -> bool:
        if key in self._pending_tiles or key in self._tile_cache:
            return False
        if len(self._tile_cache) > 720:
            self._tile_cache.pop(next(iter(self._tile_cache)))
        z, x, y = key
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        request = QNetworkRequest(QUrl(url))
        request.setRawHeader(b"User-Agent", b"FocusGuardian/1.0")
        reply = self._network.get(request)
        reply.setProperty("tileKey", f"{z}/{x}/{y}")
        self._pending_tiles.add(key)
        return True

    def _schedule_tile_update(self) -> None:
        if self._tile_update_pending:
            return
        self._tile_update_pending = True
        QTimer.singleShot(34, self._flush_tile_update)

    def _flush_tile_update(self) -> None:
        self._tile_update_pending = False
        self._last_repaint_at = time.monotonic()
        self.update()

    def _on_tile_finished(self, reply) -> None:
        key_text = str(reply.property("tileKey") or "")
        try:
            z, x, y = (int(part) for part in key_text.split("/"))
            key = (z, x, y)
        except ValueError:
            reply.deleteLater()
            return
        self._pending_tiles.discard(key)
        if reply.error() == QNetworkReply.NetworkError.NoError:
            pixmap = QPixmap()
            if pixmap.loadFromData(bytes(reply.readAll())):
                self._tile_cache[key] = pixmap
                self._schedule_tile_update()
        else:
            logger.debug("Satellite tile request failed for %s: %s", key, reply.errorString())
        reply.deleteLater()

    def _draw_vignette(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        shade = QLinearGradient(rect.topLeft(), rect.bottomRight())
        shade.setColorAt(0.0, QColor(5, 12, 20, 32))
        shade.setColorAt(0.58, QColor(5, 12, 20, 8))
        shade.setColorAt(1.0, QColor(5, 12, 20, 112))
        painter.fillRect(rect, QBrush(shade))
        painter.restore()

    def _draw_route(self, painter: QPainter, rect: QRectF) -> None:
        world_points = self._world_points()
        center_x, center_y = self._latlng_to_world(self._center_lat, self._center_lng, self._zoom)
        points = [self._project_world(point, center_x, center_y, rect) for point in world_points]
        if len(points) < 2:
            return

        path = QPainterPath(points[0])
        for point in points[1:]:
            path.lineTo(point)

        raw_index = self._display_progress * (len(points) - 1)
        active_index = max(0, min(len(points) - 2, int(math.floor(raw_index))))
        exact_world = self._world_point_at_progress(world_points, self._display_progress)
        exact_point = self._project_world(exact_world, center_x, center_y, rect)
        active = QPainterPath(points[0])
        for point in points[1:active_index + 1]:
            active.lineTo(point)
        active.lineTo(exact_point)

        painter.save()
        painter.setPen(QPen(QColor(245, 251, 255, 96), 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawPath(path)
        if self._display_progress > 0:
            painter.setPen(QPen(QColor(99, 230, 216, 62), 8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPath(active)
            painter.setPen(QPen(QColor("#63e6d8"), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPath(active)
        self._draw_airport(painter, points[0], self._model["from_code"], QColor("#9eb3c8"))
        self._draw_airport(painter, points[-1], self._model["to_code"], QColor("#ffd64d"))
        next_world = self._world_point_at_progress(world_points, min(1.0, self._display_progress + 0.0025))
        self._draw_plane(painter, exact_point, self._project_world(next_world, center_x, center_y, rect))
        painter.restore()

    def _draw_airport(self, painter: QPainter, point: QPointF, code: str, color: QColor) -> None:
        painter.save()
        painter.setPen(QPen(color, 2))
        painter.setBrush(QColor(5, 13, 23, 220))
        painter.drawRoundedRect(QRectF(point.x() - 22, point.y() - 17, 44, 34), 10, 10)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(color)
        painter.drawText(QRectF(point.x() - 22, point.y() - 17, 44, 34), Qt.AlignmentFlag.AlignCenter, code)
        painter.restore()

    def _draw_plane(self, painter: QPainter, point: QPointF, next_point: QPointF) -> None:
        angle = math.degrees(math.atan2(next_point.y() - point.y(), next_point.x() - point.x()))
        painter.save()
        painter.translate(point)
        painter.rotate(angle)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 42))
        painter.drawEllipse(QPointF(0, 0), 28, 28)
        painter.setBrush(QColor(99, 230, 216, 34))
        painter.drawEllipse(QPointF(0, 0), 20, 20)

        plane = QPainterPath()
        plane.moveTo(22, 0)
        plane.cubicTo(15, -3, 8, -4, 0, -4)
        plane.lineTo(-14, -17)
        plane.cubicTo(-17, -19, -19, -18, -18, -14)
        plane.lineTo(-12, -3)
        plane.lineTo(-23, -1)
        plane.cubicTo(-26, -1, -26, 1, -23, 1)
        plane.lineTo(-12, 3)
        plane.lineTo(-18, 14)
        plane.cubicTo(-19, 18, -17, 19, -14, 17)
        plane.lineTo(0, 4)
        plane.cubicTo(8, 4, 15, 3, 22, 0)
        plane.closeSubpath()

        painter.setPen(QPen(QColor(5, 13, 22, 210), 1.4))
        painter.setBrush(QColor("#f8ffff"))
        painter.drawPath(plane)
        painter.setPen(QPen(QColor("#63e6d8"), 1.1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(-15, 0), QPointF(14, 0))
        painter.restore()

    def _draw_overlays(self, painter: QPainter, rect: QRectF) -> None:
        route = f"{self._model['from_code']} -> {self._model['to_code']}"
        remaining_minutes = int(math.ceil(max(0, self._remaining_seconds) / 60.0))
        painter.save()
        painter.setPen(QColor("#f7fbff"))
        painter.setFont(QFont("Segoe UI", 15, QFont.Weight.DemiBold))
        painter.drawText(QRectF(0, 24, rect.width(), 34), Qt.AlignmentFlag.AlignCenter, f"{route}  {self._phase}")
        painter.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        painter.setPen(QColor(255, 255, 255, 178))
        painter.drawText(QRectF(30, rect.bottom() - 150, 300, 28), "Time Remaining")
        painter.drawText(QRectF(rect.right() - 330, rect.bottom() - 150, 300, 28), Qt.AlignmentFlag.AlignRight, "Distance Remaining")
        painter.setFont(QFont("Segoe UI", 34, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(QRectF(30, rect.bottom() - 118, 300, 80), f"{remaining_minutes} min")
        painter.drawText(QRectF(rect.right() - 330, rect.bottom() - 118, 300, 80), Qt.AlignmentFlag.AlignRight, f"{self._distance_left_km} km")
        painter.restore()


class JourneyTicketCheckOverlay(QWidget):
    """Interactive boarding ticket overlay shown on top of the live map."""

    checkedIn = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._payload: Dict[str, Any] = {}
        self._model = build_journey_model({})
        self._checked_in = False
        self._dragging = False
        self._tear_progress = 0.0
        self._fade_progress = 0.0
        self._dismiss_progress = 0.0
        self._tear_finish_active = False
        self._tear_rewind_active = False
        self._blink_on = False
        self._last_tick_at = time.monotonic()
        self._barcode_cache: Dict[tuple, QPixmap] = {}
        self._dotted_world_cache: Dict[tuple, QPixmap] = {}
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setVisible(False)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(420)
        self._blink_timer.timeout.connect(self._blink)

    def set_payload(self, payload: Dict[str, Any]) -> None:
        self._payload = dict(payload or {})
        self._model = build_journey_model(self._payload)
        self._checked_in = False
        self._dragging = False
        self._tear_progress = 0.0
        self._fade_progress = 0.0
        self._dismiss_progress = 0.0
        self._tear_finish_active = False
        self._tear_rewind_active = False
        self._last_tick_at = time.monotonic()
        self.setVisible(True)
        self._timer.start()
        self._blink_timer.start()
        self.update()

    def hideEvent(self, event) -> None:
        if hasattr(self, "_timer"):
            self._timer.stop()
        if hasattr(self, "_blink_timer"):
            self._blink_timer.stop()
        super().hideEvent(event)

    def _tick(self) -> None:
        now = time.monotonic()
        step = max(0.25, min(2.6, (now - self._last_tick_at) / (1.0 / 60.0)))
        self._last_tick_at = now
        changed = False
        if self._fade_progress < 1.0:
            self._fade_progress = min(1.0, self._fade_progress + 0.055 * step)
            changed = True
        if self._tear_finish_active and self._tear_progress < 1.0:
            self._tear_progress = min(1.0, self._tear_progress + max(0.012 * step, (1.0 - self._tear_progress) * 0.18 * step))
            self._tear_finish_active = self._tear_progress < 1.0
            changed = True
        if self._tear_rewind_active and self._tear_progress > 0.0:
            self._tear_progress = max(0.0, self._tear_progress * math.pow(0.72, step) - 0.012 * step)
            self._tear_rewind_active = self._tear_progress > 0.0
            changed = True
        if self._tear_progress >= 1.0:
            self._dismiss_progress = min(1.0, self._dismiss_progress + 0.045 * step)
            changed = True
            if self._dismiss_progress >= 1.0:
                self._timer.stop()
                self._blink_timer.stop()
                self.hide()
                self.checkedIn.emit()
                return
        if changed:
            self.update()

    def _blink(self) -> None:
        if self._checked_in and self._tear_progress < 1.0:
            self._blink_on = not self._blink_on
            self.update()

    def _ticket_rect(self) -> QRectF:
        width = min(540.0, self.width() * 0.46)
        height = min(360.0, self.height() * 0.56)
        return QRectF((self.width() - width) / 2.0, self.height() * 0.28, width, height)

    def _button_rect(self) -> QRectF:
        ticket = self._ticket_rect()
        width = min(560.0, self.width() * 0.48)
        return QRectF((self.width() - width) / 2.0, ticket.bottom() + 34, width, 58)

    def _tear_y(self, ticket: QRectF) -> float:
        return ticket.bottom() - ticket.height() * 0.21

    def _barcode_seed(self) -> int:
        text = (
            f"{self._model['from_code']}{self._model['to_code']}"
            f"{self._model['duration_minutes']}{self._model['distance_km']}"
            f"{self._payload.get('selected_seat', '')}"
        )
        return sum((index + 1) * ord(ch) for index, ch in enumerate(text))

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        ticket = self._ticket_rect()
        tear_y = self._tear_y(ticket)
        if not self._checked_in and self._button_rect().contains(event.position()):
            self._checked_in = True
            self._tear_finish_active = False
            self._tear_rewind_active = False
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.update()
            event.accept()
            return
        if self._checked_in and abs(event.position().y() - tear_y) <= 44:
            self._dragging = True
            self._tear_finish_active = False
            self._tear_rewind_active = False
            self.grabMouse()
            self._update_tear_progress(event.position().x(), ticket)
            event.accept()
            return
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self._update_tear_progress(event.position().x(), self._ticket_rect())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.releaseMouse()
            if self._tear_progress >= 0.88:
                self._tear_finish_active = True
                self._tear_rewind_active = False
                self._timer.start()
            else:
                self._tear_rewind_active = True
                self._tear_finish_active = False
                self._timer.start()
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _update_tear_progress(self, x: float, ticket: QRectF) -> None:
        left = ticket.left() + 34.0
        right = ticket.right() - 34.0
        self._tear_progress = max(0.0, min(1.0, (float(x) - left) / max(1.0, right - left)))
        self.update()

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        opacity = max(0.0, min(1.0, self._fade_progress)) * (1.0 - self._dismiss_progress)
        if opacity <= 0:
            return
        painter.setOpacity(opacity)
        painter.fillRect(QRectF(self.rect()), QColor(2, 7, 13, int(135 * opacity)))

        ticket = self._ticket_rect()
        if self._fade_progress < 1.0:
            ticket.translate(0, (1.0 - self._fade_progress) * 26.0)
        if self._dismiss_progress > 0:
            ticket.translate(0, -self._dismiss_progress * 38.0)
        self._draw_ticket(painter, ticket)
        if not self._checked_in and self._tear_progress < 1.0:
            self._draw_check_button(painter)

    def _draw_ticket(self, painter: QPainter, ticket: QRectF) -> None:
        tear_y = self._tear_y(ticket)
        top_rect = QRectF(ticket.left(), ticket.top(), ticket.width(), tear_y - ticket.top() + 2)
        bottom_rect = QRectF(ticket.left(), tear_y, ticket.width(), ticket.bottom() - tear_y)

        painter.save()
        self._draw_ticket_section(painter, ticket, top_rect, top=True)
        painter.restore()

        painter.save()
        fall = self._tear_progress * self._tear_progress * (3.0 - 2.0 * self._tear_progress)
        center = QPointF(bottom_rect.center().x(), bottom_rect.top())
        painter.translate(center)
        painter.rotate(-8.0 * fall)
        painter.translate(-center)
        painter.translate(0, fall * 88.0 + self._dismiss_progress * 90.0)
        self._draw_ticket_section(painter, ticket, bottom_rect, top=False)
        painter.restore()

        self._draw_perforation(painter, ticket, tear_y)

    def _draw_ticket_section(self, painter: QPainter, ticket: QRectF, clip_rect: QRectF, *, top: bool) -> None:
        painter.save()
        clip = QPainterPath()
        if top:
            clip.addRoundedRect(QRectF(ticket.left(), ticket.top(), ticket.width(), ticket.height()), 14, 14)
            painter.setClipPath(clip)
            painter.setClipRect(clip_rect, Qt.ClipOperation.IntersectClip)
        else:
            clip.addRoundedRect(QRectF(ticket.left(), clip_rect.top() - 10, ticket.width(), clip_rect.height() + 10), 12, 12)
            painter.setClipPath(clip)

        bg = QLinearGradient(ticket.topLeft(), ticket.bottomRight())
        bg.setColorAt(0.0, QColor("#222426"))
        bg.setColorAt(1.0, QColor("#101214"))
        painter.setPen(QPen(QColor(255, 255, 255, 22), 1))
        painter.setBrush(QBrush(bg))
        painter.drawRoundedRect(ticket, 14, 14)
        self._draw_dotted_world(painter, ticket)

        if top:
            self._draw_ticket_text(painter, ticket)
        else:
            barcode_rect = QRectF(ticket.left() + 34, clip_rect.top() + 24, ticket.width() - 68, clip_rect.height() - 42)
            self._draw_barcode(painter, barcode_rect)
        painter.restore()

    def _draw_ticket_text(self, painter: QPainter, ticket: QRectF) -> None:
        from_code = str(self._model["from_code"])
        to_code = str(self._model["to_code"])
        seat = str(self._payload.get("selected_seat") or "01A")
        painter.setFont(QFont("Segoe UI", 36, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(QRectF(ticket.left() + 34, ticket.top() + 36, 150, 58), Qt.AlignmentFlag.AlignLeft, from_code)
        painter.drawText(QRectF(ticket.right() - 184, ticket.top() + 36, 150, 58), Qt.AlignmentFlag.AlignRight, to_code)

        painter.setFont(QFont("Segoe UI", 12))
        painter.setPen(QColor(210, 214, 218, 180))
        painter.drawText(QRectF(ticket.left() + 36, ticket.top() + 100, 170, 24), Qt.AlignmentFlag.AlignLeft, str(self._model["from_name"]))
        painter.drawText(QRectF(ticket.right() - 206, ticket.top() + 100, 170, 24), Qt.AlignmentFlag.AlignRight, str(self._model["to_name"]))
        painter.drawText(QRectF(ticket.center().x() - 50, ticket.top() + 100, 100, 24), Qt.AlignmentFlag.AlignCenter, f"{self._model['duration_minutes']}m")
        self._draw_small_plane(painter, QPointF(ticket.center().x(), ticket.top() + 82), QColor(160, 166, 170, 90))

        label = QColor(210, 214, 218, 150)
        value = QColor("#ffffff")
        painter.setFont(QFont("Segoe UI", 11))
        painter.setPen(label)
        painter.drawText(QRectF(ticket.left() + 36, ticket.top() + 150, 120, 22), "Seat")
        painter.drawText(QRectF(ticket.left() + 36, ticket.top() + 210, 120, 22), "Boarding")
        painter.drawText(QRectF(ticket.right() - 176, ticket.top() + 150, 140, 22), Qt.AlignmentFlag.AlignRight, "Distance")
        painter.drawText(QRectF(ticket.right() - 176, ticket.top() + 210, 140, 22), Qt.AlignmentFlag.AlignRight, "Date")
        painter.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
        painter.setPen(value)
        painter.drawText(QRectF(ticket.left() + 36, ticket.top() + 178, 120, 24), seat)
        painter.drawText(QRectF(ticket.left() + 36, ticket.top() + 238, 120, 24), "Now")
        painter.drawText(QRectF(ticket.right() - 176, ticket.top() + 178, 140, 24), Qt.AlignmentFlag.AlignRight, f"{self._model['distance_km']} km")
        painter.drawText(QRectF(ticket.right() - 176, ticket.top() + 238, 140, 24), Qt.AlignmentFlag.AlignRight, datetime.now().strftime("%Y/%m/%d"))

    def _draw_perforation(self, painter: QPainter, ticket: QRectF, y: float) -> None:
        alpha = 180 if self._blink_on else 95
        line_color = QColor(95, 228, 212, alpha if self._checked_in else 76)
        painter.setPen(QPen(line_color, 1.5, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(ticket.left() + 34, y), QPointF(ticket.right() - 34, y))
        if self._checked_in and self._tear_progress < 1.0:
            tear_x = ticket.left() + 34 + (ticket.width() - 68) * self._tear_progress
            painter.setPen(QPen(QColor(95, 228, 212, 210), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPointF(ticket.left() + 34, y), QPointF(tear_x, y))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(235, 250, 255, 235))
            painter.drawEllipse(QPointF(tear_x, y), 13, 13)
            painter.setBrush(QColor(16, 22, 30, 220))
            painter.drawEllipse(QPointF(tear_x, y), 20, 20)
            painter.setBrush(QColor("#ffffff"))
            painter.drawEllipse(QPointF(tear_x, y), 11, 11)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(27, 73, 88, 210))
        painter.drawEllipse(QPointF(ticket.left(), y), 16, 16)
        painter.drawEllipse(QPointF(ticket.right(), y), 16, 16)

    def _draw_check_button(self, painter: QPainter) -> None:
        button = self._button_rect()
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(248, 251, 255, 236))
        painter.drawRoundedRect(button, 29, 29)
        painter.setFont(QFont("Segoe UI", 16, QFont.Weight.DemiBold))
        painter.setPen(QColor("#111111"))
        painter.drawText(button, Qt.AlignmentFlag.AlignCenter, "Check in")
        painter.restore()

    def _draw_barcode(self, painter: QPainter, rect: QRectF) -> None:
        width = max(1, int(rect.width()))
        height = max(1, int(rect.height()))
        seed = self._barcode_seed()
        key = (width, height, seed)
        pixmap = self._barcode_cache.get(key)
        if pixmap is None:
            pixmap = QPixmap(width, height)
            pixmap.fill(Qt.GlobalColor.transparent)
            barcode_painter = QPainter(pixmap)
            barcode_painter.setPen(Qt.PenStyle.NoPen)
            barcode_painter.setBrush(QColor("#ffffff"))
            barcode_painter.drawRect(QRectF(0, 0, width, height))
            x = 8.0
            max_x = width - 8.0
            local_seed = seed
            while x < max_x:
                local_seed = (local_seed * 1103515245 + 12345) & 0x7FFFFFFF
                bar_width = 1 + (local_seed % 5)
                gap = 1 + ((local_seed >> 3) % 3)
                barcode_painter.setBrush(QColor("#111111"))
                barcode_painter.drawRect(QRectF(x, 6, bar_width, max(1, height - 12)))
                x += bar_width + gap
            barcode_painter.end()
            if len(self._barcode_cache) > 8:
                self._barcode_cache.clear()
            self._barcode_cache[key] = pixmap
        painter.drawPixmap(rect, pixmap, QRectF(pixmap.rect()))

    def _draw_dotted_world(self, painter: QPainter, ticket: QRectF) -> None:
        width = max(1, int(ticket.width()))
        height = max(1, int(ticket.height()))
        key = (width, height)
        pixmap = self._dotted_world_cache.get(key)
        if pixmap is None:
            pixmap = QPixmap(width, height)
            pixmap.fill(Qt.GlobalColor.transparent)
            dot_painter = QPainter(pixmap)
            dot_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            dot_painter.setPen(Qt.PenStyle.NoPen)
            dot_painter.setBrush(QColor(255, 255, 255, 22))
            local = QRectF(0, 0, width, height)
            clusters = [
                QRectF(local.left() + 118, local.top() + 22, 124, 108),
                QRectF(local.left() + 158, local.top() + 118, 116, 136),
                QRectF(local.center().x() - 42, local.top() + 54, 150, 122),
                QRectF(local.right() - 196, local.top() + 44, 134, 150),
            ]
            for cluster in clusters:
                for yy in range(int(cluster.top()), int(cluster.bottom()), 6):
                    for xx in range(int(cluster.left()), int(cluster.right()), 6):
                        cx = (xx - cluster.center().x()) / max(1.0, cluster.width() / 2)
                        cy = (yy - cluster.center().y()) / max(1.0, cluster.height() / 2)
                        if cx * cx + cy * cy < 0.82:
                            dot_painter.drawEllipse(QPointF(xx, yy), 1.2, 1.2)
            dot_painter.end()
            if len(self._dotted_world_cache) > 4:
                self._dotted_world_cache.clear()
            self._dotted_world_cache[key] = pixmap
        painter.drawPixmap(ticket, pixmap, QRectF(pixmap.rect()))

    def _draw_small_plane(self, painter: QPainter, point: QPointF, color: QColor) -> None:
        painter.save()
        painter.translate(point)
        painter.rotate(90)
        plane = QPainterPath()
        plane.moveTo(0, -10)
        plane.lineTo(4, 3)
        plane.lineTo(10, 6)
        plane.lineTo(2, 7)
        plane.lineTo(0, 12)
        plane.lineTo(-2, 7)
        plane.lineTo(-10, 6)
        plane.lineTo(-4, 3)
        plane.closeSubpath()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPath(plane)
        painter.restore()


class ClickableVolumeSlider(QSlider):
    """Horizontal slider that jumps to the clicked/dragged position."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSingleStep(1)
        self.setPageStep(5)
        self.setMouseTracking(True)

    def _set_value_from_x(self, x: float) -> None:
        span = max(1.0, float(self.width() - 2))
        ratio = max(0.0, min(1.0, (float(x) - 1.0) / span))
        value = self.minimum() + round(ratio * (self.maximum() - self.minimum()))
        self.setValue(max(self.minimum(), min(self.maximum(), value)))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.setSliderDown(True)
            self._set_value_from_x(event.position().x())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._set_value_from_x(event.position().x())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._set_value_from_x(event.position().x())
            self.setSliderDown(False)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class JourneySoundPanel(QFrame):
    """Overlay sound picker for the large journey map."""

    TRACK_CARDS = (
        ("Airplane Sound", "brown_noise"),
        ("Raindrop", "rain_light"),
        ("Ocean Waves", "stream"),
        ("Forest", "forest"),
    )

    def __init__(
        self,
        *,
        config: Optional[dict] = None,
        audio_manager=None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.config = config if config is not None else {}
        self.audio_manager = audio_manager
        self._track_buttons: Dict[str, QPushButton] = {}
        self.setObjectName("journeySoundPanel")
        self.setVisible(False)
        self.setStyleSheet(
            """
            QFrame#journeySoundPanel {
                background-color: rgba(14, 22, 32, 238);
                border: 1px solid rgba(132, 170, 205, 0.28);
                border-radius: 18px;
            }
            QLabel#soundTitle {
                color: white;
                font: 850 24px "Segoe UI";
            }
            QLabel#soundSection {
                color: rgba(255,255,255,0.90);
                font: 760 15px "Segoe UI";
            }
            QLabel#volumeLabel {
                color: rgba(255,255,255,0.70);
                font: 700 12px "Segoe UI";
            }
            QLabel#volumeValue {
                color: #f7fbff;
                background: rgba(255,255,255,0.10);
                border: 1px solid rgba(255,255,255,0.16);
                border-radius: 10px;
                padding: 4px 8px;
                font: 800 12px "Segoe UI";
            }
            QPushButton#soundToggle {
                background-color: rgba(255,255,255,0.15);
                color: white;
                border: 1px solid rgba(255,255,255,0.26);
                border-radius: 18px;
                padding: 0 14px;
                font: 850 14px "Segoe UI";
            }
            QPushButton#soundToggle:hover {
                background-color: rgba(255,255,255,0.20);
            }
            QPushButton#soundCard {
                background-color: rgba(22, 22, 22, 0.48);
                color: rgba(255,255,255,0.94);
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 12px;
                padding: 10px 14px;
                text-align: left;
                font: 780 14px "Segoe UI";
            }
            QPushButton#soundCard:checked {
                background-color: rgba(68, 125, 155, 0.58);
                border-color: rgba(132, 213, 255, 0.62);
            }
            QPushButton#soundCard:disabled {
                color: rgba(255,255,255,0.46);
                background-color: rgba(22,22,22,0.28);
            }
            QSlider#journeyVolumeSlider {
                min-height: 36px;
            }
            QSlider#journeyVolumeSlider::groove:horizontal {
                height: 8px;
                border-radius: 4px;
                background-color: rgba(255,255,255,0.10);
                border: none;
            }
            QSlider#journeyVolumeSlider::sub-page:horizontal {
                border-radius: 4px;
                background-color: rgba(89, 213, 192, 0.95);
            }
            QSlider#journeyVolumeSlider::add-page:horizontal {
                border-radius: 4px;
                background-color: rgba(255,255,255,0.10);
            }
            QSlider#journeyVolumeSlider::handle:horizontal {
                width: 22px;
                height: 22px;
                margin: -7px 0px;
                border-radius: 11px;
                border: 2px solid rgba(9, 18, 28, 0.92);
                background-color: #f7fbff;
            }
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(13)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Sound")
        title.setObjectName("soundTitle")
        top.addWidget(title)
        self.toggle_button = QPushButton()
        self.toggle_button.setObjectName("soundToggle")
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.setFixedSize(96, 38)
        self.toggle_button.clicked.connect(self._toggle_audio)
        top.addStretch(1)
        top.addWidget(self.toggle_button)
        root.addLayout(top)

        volume_row = QHBoxLayout()
        volume_row.setContentsMargins(0, 0, 0, 0)
        volume_row.setSpacing(10)
        volume_label = QLabel("Volume")
        volume_label.setObjectName("volumeLabel")
        volume_row.addWidget(volume_label)
        self.volume_slider = ClickableVolumeSlider(self)
        self.volume_slider.setObjectName("journeyVolumeSlider")
        self.volume_slider.setRange(0, 100)
        self.volume_slider.valueChanged.connect(self._set_volume)
        volume_row.addWidget(self.volume_slider, 1)
        self.volume_value_label = QLabel("0%")
        self.volume_value_label.setObjectName("volumeValue")
        self.volume_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.volume_value_label.setFixedWidth(52)
        volume_row.addWidget(self.volume_value_label)
        root.addLayout(volume_row)

        section = QLabel("White Noise")
        section.setObjectName("soundSection")
        root.addWidget(section)

        for label, track_key in self.TRACK_CARDS:
            button = QPushButton(f"{label}    OFF")
            button.setObjectName("soundCard")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(58)
            button.clicked.connect(lambda checked=False, key=track_key: self._select_track(key))
            self._track_buttons[track_key] = button
            root.addWidget(button)

        focus_label = QLabel("Focus Sound")
        focus_label.setObjectName("soundSection")
        root.addWidget(focus_label)
        coming = QPushButton("Coming Soon    OFF")
        coming.setObjectName("soundCard")
        coming.setEnabled(False)
        coming.setMinimumHeight(58)
        root.addWidget(coming)
        root.addStretch(1)
        self.refresh_state()

    def refresh_state(self) -> None:
        enabled = bool(self.config.get("enable_focus_audio", False))
        if self.audio_manager is not None:
            enabled = bool(self.audio_manager.is_enabled())
        track = str(self.config.get("focus_audio_track", "rain_light") or "rain_light")
        if self.audio_manager is not None:
            track = str(self.audio_manager.current_track_key())
        try:
            volume = int(self.config.get("focus_audio_volume", 30))
        except (TypeError, ValueError):
            volume = 30
        if self.audio_manager is not None:
            volume = int(self.audio_manager.current_volume())

        self.toggle_button.setText("\u266a  ON" if enabled else "\u266a  OFF")
        self.volume_slider.blockSignals(True)
        self.volume_slider.setValue(max(0, min(100, volume)))
        self.volume_slider.blockSignals(False)
        self.volume_value_label.setText(f"{max(0, min(100, volume))}%")

        for key, button in self._track_buttons.items():
            selected = key == track
            button.setChecked(selected)
            label = next((name for name, item_key in self.TRACK_CARDS if item_key == key), key)
            button.setText(f"{label}    {'ON' if selected and enabled else 'OFF'}")

    def _toggle_audio(self) -> None:
        enabled = not bool(self.config.get("enable_focus_audio", False))
        if self.audio_manager is not None:
            enabled = not self.audio_manager.is_enabled()
            self.audio_manager.set_enabled(enabled)
            self.config.update(self.audio_manager.to_config())
        else:
            self.config["enable_focus_audio"] = enabled
        self.refresh_state()

    def _set_volume(self, value: int) -> None:
        value = max(0, min(100, int(value)))
        self.volume_value_label.setText(f"{value}%")
        if self.audio_manager is not None:
            self.audio_manager.set_volume(value)
            self.config.update(self.audio_manager.to_config())
        else:
            self.config["focus_audio_volume"] = int(value)
        self.refresh_state()

    def _select_track(self, track_key: str) -> None:
        if self.audio_manager is not None:
            self.audio_manager.set_track(track_key)
            if self.audio_manager.is_enabled():
                self.audio_manager.play(track_key)
            self.config.update(self.audio_manager.to_config())
        else:
            self.config["focus_audio_track"] = track_key
        self.refresh_state()


class FocusJourneyMapDialog(QDialog):
    """Full map dialog opened from the Focus Journey card."""

    pauseRequested = pyqtSignal()

    def __init__(
        self,
        *,
        config: Optional[dict] = None,
        audio_manager=None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.config = config or {}
        self.audio_manager = audio_manager
        self._ticket_checked = False
        self._pending_progress: Optional[tuple[float, int, int, str]] = None
        self._current_payload: Dict[str, Any] = {}
        self._map_window_maximized = False
        self._normal_geometry = None
        self._dragging_window = False
        self._drag_start_global = None
        self._drag_origin = None
        self._checked_route_key = None
        self._current_route_key = None
        self._paused = False
        self.setWindowTitle("Focus Journey Map")
        self.setMinimumSize(940, 620)
        self.resize(1040, 700)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Use the native tile renderer by default. It avoids the common
        # QtWebEngine packaging mismatch and still renders real satellite tiles.
        self.map_widget = FallbackJourneyMapWidget(self)
        self.map_widget.installEventFilter(self)
        layout.addWidget(self.map_widget)

        self.window_controls = self._make_window_controls()
        self.sound_button = self._make_map_button("\u266a")
        self.sound_button.clicked.connect(self._toggle_sound_panel)
        self.pause_button = self._make_map_button("II")
        self.pause_button.clicked.connect(self.pauseRequested.emit)
        self.ticket_overlay = JourneyTicketCheckOverlay(self)
        self.ticket_overlay.installEventFilter(self)
        self.ticket_overlay.checkedIn.connect(self._complete_ticket_check)
        self.sound_panel = JourneySoundPanel(config=self.config, audio_manager=self.audio_manager, parent=self)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.sound_button.move(18, 18)
        self.pause_button.move(18, 74)
        self.window_controls.move(self.width() - self.window_controls.width() - 22, 25)
        self.ticket_overlay.setGeometry(self.rect())
        self._position_sound_panel()
        if self.ticket_overlay.isVisible():
            self.ticket_overlay.raise_()
        if self.sound_panel.isVisible():
            self.sound_panel.raise_()
        self.sound_button.raise_()
        self.pause_button.raise_()
        self.window_controls.raise_()

    def _position_sound_panel(self) -> None:
        panel_width = min(380, max(320, self.width() - 132))
        panel_height = max(360, self.height() - 36)
        self.sound_panel.setGeometry(86, 18, panel_width, panel_height)

    def eventFilter(self, obj, event) -> bool:
        if obj in (self.map_widget, self.ticket_overlay):
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                if self._is_window_drag_zone(event.position()):
                    self._dragging_window = True
                    self._drag_start_global = event.globalPosition().toPoint()
                    self._drag_origin = self.pos()
                    event.accept()
                    return True
            if event.type() == QEvent.Type.MouseMove and self._dragging_window:
                if self._drag_start_global is not None and self._drag_origin is not None:
                    self.move(self._drag_origin + event.globalPosition().toPoint() - self._drag_start_global)
                event.accept()
                return True
            if event.type() == QEvent.Type.MouseButtonRelease and self._dragging_window:
                self._dragging_window = False
                self._drag_start_global = None
                self._drag_origin = None
                event.accept()
                return True
        return super().eventFilter(obj, event)

    def _is_window_drag_zone(self, pos: QPointF) -> bool:
        if self._map_window_maximized:
            return False
        margin = 24.0
        return (
            pos.y() <= 86.0
            or pos.x() <= margin
            or pos.x() >= self.width() - margin
            or pos.y() >= self.height() - margin
        )

    def _make_window_controls(self) -> QWidget:
        host = QWidget(self)
        host.setFixedSize(88, 24)
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(8)
        specs = (
            ("#f2c94c", self._minimize_app),
            ("#37d67a", self._toggle_max_restore),
            ("#ff6b5f", self.close),
        )
        for color, slot in specs:
            button = QToolButton(host)
            button.setFixedSize(17, 17)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(slot)
            button.setStyleSheet(
                f"QToolButton {{ background-color: {color}; border: 1px solid rgba(0,0,0,0.24); border-radius: 8px; }}"
                f"QToolButton:hover {{ border-color: rgba(255,255,255,0.72); }}"
            )
            row.addWidget(button)
        host.raise_()
        return host

    def _minimize_app(self) -> None:
        self.showMinimized()
        parent_window = self.parentWidget().window() if self.parentWidget() is not None else None
        if parent_window is not None and parent_window is not self:
            parent_window.showMinimized()

    def _toggle_max_restore(self) -> None:
        if self._map_window_maximized:
            if self._normal_geometry is not None:
                self.setGeometry(self._normal_geometry)
            else:
                self.showNormal()
            self._map_window_maximized = False
            return

        self._normal_geometry = self.geometry()
        screen = self.windowHandle().screen() if self.windowHandle() is not None else None
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.availableGeometry())
        else:
            self.showMaximized()
        self._map_window_maximized = True

    def _make_map_button(self, text: str) -> QToolButton:
        button = QToolButton(self)
        button.setText(text)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFixedSize(48, 48)
        button.setStyleSheet(
            """
            QToolButton {
                background: rgba(6, 15, 25, 0.74);
                color: #f7fbff;
                border: 1px solid rgba(145, 180, 210, 0.28);
                border-radius: 24px;
                font: 850 22px "Segoe UI";
            }
            QToolButton:hover {
                background: rgba(18, 35, 52, 0.88);
                border-color: rgba(99, 230, 216, 0.58);
            }
            """
        )
        button.raise_()
        return button

    def _toggle_sound_panel(self) -> None:
        self.sound_panel.refresh_state()
        self._position_sound_panel()
        self.sound_panel.setVisible(not self.sound_panel.isVisible())
        self.sound_panel.raise_()
        self.window_controls.raise_()
        self.sound_button.raise_()
        self.pause_button.raise_()

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)
        self.pause_button.setText(">" if self._paused else "II")
        if hasattr(self.map_widget, "set_motion_paused"):
            self.map_widget.set_motion_paused(self._paused or not self._ticket_checked)

    def _route_session_key(self, data: Dict[str, Any]) -> tuple:
        return (
            str(data.get("route_from_code") or data.get("from_code") or ""),
            str(data.get("route_to_code") or data.get("to_code") or ""),
            int(data.get("planned_minutes") or data.get("route_duration_minutes") or 0),
            int(data.get("route_distance_km") or data.get("distance_km") or 0),
            int(float(data.get("journey_session_id") or 0)),
        )

    def set_journey_data(self, data: Dict[str, Any]) -> None:
        self._current_payload = dict(data or {})
        self._current_route_key = self._route_session_key(self._current_payload)
        stored_key = str(self.config.get("_focus_journey_checked_route_key", ""))
        self._ticket_checked = self._checked_route_key == self._current_route_key or stored_key == repr(self._current_route_key)
        self._pending_progress = None
        self.map_widget.set_journey_data(data)
        model = build_journey_model(data)
        self.map_widget.update_progress(
            0.0,
            int(model.get("duration_minutes", 25) or 25) * 60,
            int(model.get("distance_km", 0) or 0),
            "Boarding",
        )
        self.ticket_overlay.setGeometry(self.rect())
        if self._ticket_checked:
            self.ticket_overlay.hide()
            if hasattr(self.map_widget, "set_motion_paused"):
                self.map_widget.set_motion_paused(self._paused)
        else:
            if hasattr(self.map_widget, "set_motion_paused"):
                self.map_widget.set_motion_paused(True)
            self.ticket_overlay.set_payload(self._current_payload)
            self.ticket_overlay.raise_()
        self.sound_button.raise_()
        self.pause_button.raise_()
        self.window_controls.raise_()

    def update_progress(
        self,
        progress: float,
        remaining_seconds: int,
        distance_left_km: int,
        phase: str = "",
    ) -> None:
        if not self._ticket_checked:
            self._pending_progress = (
                max(0.0, min(1.0, float(progress or 0.0))),
                max(0, int(remaining_seconds or 0)),
                max(0, int(distance_left_km or 0)),
                str(phase or "Boarding"),
            )
            model = build_journey_model(self._current_payload)
            self.map_widget.update_progress(
                0.0,
                int(model.get("duration_minutes", 25) or 25) * 60,
                int(model.get("distance_km", 0) or 0),
                "Boarding",
            )
            if hasattr(self.map_widget, "set_motion_paused"):
                self.map_widget.set_motion_paused(True)
            return
        self.map_widget.update_progress(progress, remaining_seconds, distance_left_km, phase)

    def _complete_ticket_check(self) -> None:
        self._ticket_checked = True
        self._checked_route_key = self._current_route_key
        self.config["_focus_journey_checked_route_key"] = repr(self._current_route_key)
        pending = self._pending_progress
        if pending is None:
            model = build_journey_model(self._current_payload)
            pending = (
                0.0,
                int(model.get("duration_minutes", 25) or 25) * 60,
                int(model.get("distance_km", 0) or 0),
                "Boarding",
            )
        self.map_widget.update_progress(*pending)
        if hasattr(self.map_widget, "set_motion_paused"):
            self.map_widget.set_motion_paused(self._paused)
