"""Step 05 — Live data ingest (WAQI)

Writes:
  - data/live/latest_reading.json

Env:
  - WAQI_TOKEN

This is the plan-vNext ingest script. It intentionally stores *raw* values.
Feature scaling happens in Step 06.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import requests
from dotenv import load_dotenv


CITY_SLUGS = {
    "Delhi": "delhi",
    "Bengaluru": "bangalore",
    "Kolkata": "kolkata",
    "Hyderabad": "hyderabad",
}

OUT_PATH = "data/live/latest_reading.json"


def _safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def fetch_city(slug: str, token: str) -> dict:
    url = f"https://api.waqi.info/feed/{slug}/?token={token}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    d = r.json()
    if d.get("status") != "ok":
        raise ValueError(f"API failure for {slug}: {d}")

    iaqi = d["data"].get("iaqi", {})
    out = {
        "aqi": _safe_float(d["data"].get("aqi")),
        "pm2_5": _safe_float(iaqi.get("pm25", {}).get("v")),
        "pm10": _safe_float(iaqi.get("pm10", {}).get("v")),
        "co": _safe_float(iaqi.get("co", {}).get("v")),
        "no2": _safe_float(iaqi.get("no2", {}).get("v")),
        "o3": _safe_float(iaqi.get("o3", {}).get("v")),
        "so2": _safe_float(iaqi.get("so2", {}).get("v")),
        "temperature": _safe_float(iaqi.get("t", {}).get("v")),
        "wind_speed": _safe_float(iaqi.get("w", {}).get("v")),
        "pressure": _safe_float(iaqi.get("p", {}).get("v")),
        # Not in live API
        "rainfall": 0.0,
        "fetched_at": datetime.utcnow().isoformat(),
        "api_timestamp": d["data"].get("time", {}).get("s"),
    }
    return out


def main() -> None:
    load_dotenv()
    token = os.getenv("WAQI_TOKEN")
    if not token:
        print("ERROR: WAQI_TOKEN not set. Copy .env.example to .env and set your token.")
        sys.exit(1)

    os.makedirs("data/live", exist_ok=True)

    print("=" * 60)
    print("Step 05 — Live ingest (WAQI)")
    print("=" * 60)

    readings: dict[str, dict | None] = {}
    ok = 0
    for city, slug in CITY_SLUGS.items():
        try:
            readings[city] = fetch_city(slug, token)
            ok += 1
            print(f"  ✓ {city:<10s} pm2_5={readings[city].get('pm2_5')} aqi={readings[city].get('aqi')}")
        except Exception as e:
            readings[city] = None
            print(f"  ✗ {city:<10s} failed: {e}")

    with open(OUT_PATH, "w") as f:
        json.dump(readings, f, indent=2)

    print(f"\n✓ Saved: {OUT_PATH} ({ok}/{len(CITY_SLUGS)} cities)")


if __name__ == "__main__":
    main()
