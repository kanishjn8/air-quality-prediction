"""Utility helpers for storing and loading a rolling live history window.

Why this exists
---------------
The training dataset in this repo ends in 2023, but live inference happens in
real-time. Using a fixed historical last-7 window from 2023 makes lag/rolling
features inconsistent with "today" (live) values.

This module maintains a small on-disk buffer under data/live/ that accumulates
live observations daily (or whenever 06_live_ingest.py runs).

File format
-----------
`data/live/live_history.json` is a dict:

{
  "Delhi": [ {"timestamp": "...", "pm2_5": 100, ...}, ... ],
  ...
}

We keep only the most recent `max_len` entries per city.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


LIVE_HISTORY_PATH = "data/live/live_history.json"


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def load_live_history(path: str = LIVE_HISTORY_PATH) -> Dict[str, List[Dict[str, Any]]]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    # ensure list values
    out: Dict[str, List[Dict[str, Any]]] = {}
    for city, rows in data.items():
        if isinstance(rows, list):
            out[city] = [r for r in rows if isinstance(r, dict)]
    return out


def save_live_history(history: Dict[str, List[Dict[str, Any]]], path: str = LIVE_HISTORY_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def append_live_readings(
    readings: Dict[str, Optional[Dict[str, Any]]],
    *,
    max_len: int = 7,
    path: str = LIVE_HISTORY_PATH,
    timestamp: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Append the latest live readings and keep only the most recent `max_len`.

    `readings` is the object written by 06_live_ingest.py.

    Returns the updated history.
    """

    ts = timestamp or datetime.utcnow().isoformat()
    hist = load_live_history(path)

    for city, r in readings.items():
        if not r:
            continue

        row: Dict[str, Any] = {
            "timestamp": ts,
            # core fields the model expects (raw features used in city_last7)
            "aqi": r.get("aqi"),
            "pm2_5": _safe_float(r.get("pm2_5")),
            "pm10": _safe_float(r.get("pm10")),
            "co": _safe_float(r.get("co")),
            "no2": _safe_float(r.get("no2")),
            "o3": _safe_float(r.get("o3")),
            "so2": _safe_float(r.get("so2")),
            "no": _safe_float(r.get("no")),
            "nh3": _safe_float(r.get("nh3")),
            "temperature": _safe_float(r.get("temperature")),
            "wind_speed": _safe_float(r.get("wind_speed")),
            "rainfall": _safe_float(r.get("rainfall")),
            "pressure": _safe_float(r.get("pressure")),
        }

        rows = hist.get(city, [])
        rows.append(row)
        hist[city] = rows[-max_len:]

    save_live_history(hist, path)
    return hist
