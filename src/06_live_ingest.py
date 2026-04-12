"""
Step 06 — Live Data Ingestion
Fetches current air quality readings from the WAQI API for 5 Indian cities
and saves them to data/live/latest_reading.json.

Usage:
    uv run python src/06_live_ingest.py
"""

import os
import json
import sys
from datetime import datetime

import requests
from dotenv import load_dotenv

from live_history import append_live_readings


def aqi_to_training_scale(aqi_val):
    """Convert WAQI AQI (0–500-ish) to the project's historical 1–5 scale.

    Historical `merged.csv` uses an AQI category 1..5. We map the live AQI
    to the same buckets to avoid out-of-distribution inputs.
    """
    if aqi_val is None:
        return None
    try:
        aqi = float(aqi_val)
    except (TypeError, ValueError):
        return None
    if aqi <= 50:
        return 1
    if aqi <= 100:
        return 2
    if aqi <= 200:
        return 3
    if aqi <= 300:
        return 4
    return 5


def sanitize_live_reading(result: dict) -> dict:
    """Sanitize live sensor values to match training distribution/units.

    - pressure: training is ~990–1025 hPa; treat values outside 850–1100 as missing
    - wind_speed: training max ~30; clamp to [0, 40] and treat absurd values as missing
    - temperature: clamp to a conservative [ -5, 55 ]
    """

    # Convert AQI to the 1–5 training scale
    result["aqi_raw"] = result.get("aqi")
    result["aqi"] = aqi_to_training_scale(result.get("aqi"))

    # NOTE on units:
    # The historical dataset used for training has pollutant magnitudes that
    # often differ from WAQI live values (especially CO). We keep raw values
    # at ingestion, and additional robustness / clipping happens in Step 07.

    def _to_float(x):
        try:
            if x is None:
                return None
            return float(x)
        except (TypeError, ValueError):
            return None

    # Convert common numeric fields to floats early
    for k in ["pm2_5", "pm10", "co", "no2", "o3", "so2", "humidity", "dew_point"]:
        if k in result:
            result[k] = _to_float(result.get(k))

    # Pressure sanity
    p = _to_float(result.get("pressure"))
    if p is None or p < 850 or p > 1100:
        result["pressure"] = None
    else:
        result["pressure"] = p

    # Wind sanity
    w = _to_float(result.get("wind_speed"))
    if w is None or w < 0 or w > 40:
        result["wind_speed"] = None
    else:
        result["wind_speed"] = w

    # Temperature sanity
    t = _to_float(result.get("temperature"))
    if t is None:
        result["temperature"] = None
    else:
        result["temperature"] = max(-5.0, min(55.0, t))

    return result

# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────
load_dotenv()
TOKEN = os.getenv("WAQI_TOKEN")

if not TOKEN:
    print("ERROR: WAQI_TOKEN not set. Create a .env file with WAQI_TOKEN=your_token_here")
    sys.exit(1)

CITY_SLUGS = {
    "Delhi":     "delhi",
    "Bengaluru": "bengaluru",
    "Kolkata":   "kolkata",
    "Hyderabad": "hyderabad",
}

OUTPUT_PATH = "data/live/latest_reading.json"
os.makedirs("data/live", exist_ok=True)

print("=" * 60)
print("Step 06 — Live Data Ingestion (WAQI API)")
print("=" * 60)


def fetch_city(slug):
    """Fetch live AQI data for a single city from WAQI API."""
    url = f"https://api.waqi.info/feed/{slug}/?token={TOKEN}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    d = r.json()
    if d["status"] != "ok":
        raise ValueError(f"API error for {slug}: {d}")

    iaqi = d["data"].get("iaqi", {})

    result = {
        "aqi":         d["data"].get("aqi"),
        "pm2_5":       iaqi.get("pm25", {}).get("v"),
        "pm10":        iaqi.get("pm10", {}).get("v"),
        "co":          iaqi.get("co",   {}).get("v"),
        "no2":         iaqi.get("no2",  {}).get("v"),
        "o3":          iaqi.get("o3",   {}).get("v"),
        "so2":         iaqi.get("so2",  {}).get("v"),
        "temperature": iaqi.get("t",    {}).get("v"),
        "wind_speed":  iaqi.get("w",    {}).get("v"),
        "pressure":    iaqi.get("p",    {}).get("v"),
        "humidity":    iaqi.get("h",    {}).get("v"),
        "dew_point":   iaqi.get("dew",  {}).get("v"),
        # Fields absent from live API — will be imputed during prediction
        "no":          None,
        "nh3":         None,
        "rainfall":    None,
        "fetched_at":  datetime.utcnow().isoformat(),
        "api_timestamp": d["data"].get("time", {}).get("s"),
    }

    # Extract forecast data if available
    forecast = d["data"].get("forecast", {}).get("daily", {})
    if forecast:
        result["forecast_pm25"] = forecast.get("pm25", [])
        result["forecast_pm10"] = forecast.get("pm10", [])

    return sanitize_live_reading(result)


# ═══════════════════════════════════════════════════════════════
# Fetch all cities
# ═══════════════════════════════════════════════════════════════
readings = {}
success_count = 0

for city, slug in CITY_SLUGS.items():
    try:
        readings[city] = fetch_city(slug)
        success_count += 1
        pm = readings[city].get("pm2_5", "N/A")
        aqi = readings[city].get("aqi", "N/A")
        print(f"  ✓ {city:<12s}  PM2.5={pm}  AQI={aqi}")
    except Exception as e:
        print(f"  ✗ {city:<12s}  FAILED: {e}")
        readings[city] = None

# Save
with open(OUTPUT_PATH, "w") as f:
    json.dump(readings, f, indent=2)

# Maintain a rolling 7-day live history buffer for inference lag features
try:
    append_live_readings(readings, max_len=7)
except Exception as e:
    # Non-fatal: live prediction can still fall back to models/city_last7.json
    print(f"  [WARN] Failed to update live history buffer: {e}")

print(f"\n{'=' * 60}")
print(f"✓ Live readings saved to {OUTPUT_PATH}")
print(f"  {success_count}/{len(CITY_SLUGS)} cities fetched successfully")
print(f"{'=' * 60}")
