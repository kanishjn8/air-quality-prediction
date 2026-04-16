from datetime import datetime
from pathlib import Path

import pandas as pd
from meteostat import Point, Daily

# Paths
CITIES_PATH = Path("data/raw/cities.csv")
OUTPUT_DIR = Path("data/raw/weather")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Date range
start = datetime(2015, 1, 1)
end = datetime(2024, 12, 31)

# Load cities
cities = pd.read_csv(CITIES_PATH)

for _, row in cities.iterrows():
    city = row["city"]
    lat = row["latitude"]
    lon = row["longitude"]

    location = Point(lat, lon)
    data = Daily(location, start, end).fetch()
    print(city, data.columns.tolist())

    if data.empty:
        print(f"No data found for {city}")
        continue

    data = data.reset_index()

    data = data[["time", "tavg", "wspd", "prcp", "pres"]]#We are keeping this column and eliminating others
    data.columns = ["date", "temperature", "wind_speed", "rainfall", "pressure"]#renaming of columns

    out_file = OUTPUT_DIR / f"{city.lower()}_weather_2015.csv"
    data.to_csv(out_file, index=False)

    print(f"Saved weather data for {city}")