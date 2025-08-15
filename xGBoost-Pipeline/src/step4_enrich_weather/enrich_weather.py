import os
import sys
import json
import asyncio
import pandas as pd
from tqdm import tqdm
from hashlib import md5
from datetime import datetime
import argparse
import logging
import requests
import time

# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("weather_async_log.txt")
    ]
)

# ---------- Caching Helpers ----------
def cache_key(lat, lon, date_or_range):
    return md5(f"{lat}-{lon}-{date_or_range}".encode()).hexdigest()

def load_from_cache(lat, lon, date_or_range):
    key = cache_key(lat, lon, date_or_range)
    file = f"cache/{key}.json"
    if os.path.exists(file):
        with open(file) as f:
            return json.load(f)
    return None

def save_to_cache(lat, lon, date_or_range, data):
    os.makedirs("cache", exist_ok=True)
    key = cache_key(lat, lon, date_or_range)
    with open(f"cache/{key}.json", "w") as f:
        json.dump(data, f)

# ------------- Fetch Weather for Date Range -------------
def fetch_weather_range(lat, lon, start, end, retries=3, backoff_factor=5,):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "PRECTOTCORR,WS2M,T2M",
        "start": start,
        "end": end,
        "latitude": lat,
        "longitude": lon,
        "format": "JSON",
        "community": "RE"
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 429:
                wait = backoff_factor * (2 ** attempt)
                print(f"[429] Rate limited for {lat},{lon}. Waiting {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()

            data = response.json()
            parameters = data.get("properties", {}).get("parameter", {})

            if not parameters:
                print(f"[Warning] No data returned for {lat}, {lon}")
                return {}

            result = {}
            for date in parameters.get("T2M", {}):
                result[date] = {
                    "lat": lat,
                    "lon": lon,
                    "date": date,
                    "temperature": parameters.get("T2M", {}).get(date),
                    "wind_speed": parameters.get("WS2M", {}).get(date),
                    "rainfall": parameters.get("PRECTOTCORR", {}).get(date),
                }

            return result

        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                print(f"[Error] Failed after {retries} attempts for {lat}, {lon} — {str(e)}")
                return {}
            wait = backoff_factor * (2 ** attempt)
            print(f"[Retry] Attempt {attempt+1} failed for {lat}, {lon}: {e}. Retrying in {wait}s...")
            time.sleep(wait)

def enrich_with_weather(input_csv, output_csv):
    print(" Starting enrichment...")

    # Load and clean
    df = pd.read_csv(input_csv)
    df["date"] = pd.to_datetime(df["capture_date"], errors="coerce").dt.normalize()
    df["date"] = df["date"].dt.tz_localize(None) 

    df = df.dropna(subset=["lat", "lon", "date"])
    df["date_str"] = df["date"].dt.strftime("%Y%m%d")
    df["lat_r"] = df["lat"].round(2)  # Coarser binning to reduce duplicates
    df["lon_r"] = df["lon"].round(2)

    # Create cache-aware key set
    unique_requests = df.groupby(["lat_r", "lon_r"]).agg(
        start_date=("date_str", "min"),
        end_date=("date_str", "max")
    ).reset_index()

    all_weather_records = []
    print(f" Total unique location-date ranges: {len(unique_requests)}")


    for _, row in tqdm(unique_requests.iterrows(), total=len(unique_requests), desc="Weather fetch"):
        
        lat = row["lat_r"]
        lon = row["lon_r"]
        start = row["start_date"]
        end = row["end_date"]

        # Check full-range cache
        cache_result = load_from_cache(lat, lon, f"{start}_{end}")
        if cache_result:
            weather_by_date = cache_result
        else:
            weather_by_date = fetch_weather_range(lat, lon, start, end)
            save_to_cache(lat, lon, f"{start}_{end}", weather_by_date)

        for _, weather in weather_by_date.items():
            all_weather_records.append(weather)
       

    # Weather DataFrame
    weather_df = pd.DataFrame(all_weather_records)
    weather_df["date"] = pd.to_datetime(weather_df["date"], format="%Y%m%d", errors="coerce").dt.normalize()
    weather_df["date"] = weather_df["date"].dt.tz_localize(None) 

    print("weather_df count:",len(weather_df))
    print(weather_df.head())



    # Merge and save
    weather_df = weather_df.rename(columns={"lat": "lat_r", "lon": "lon_r"})
    enriched_df = df.merge(weather_df, on=["lat_r", "lon_r", "date"], how="left")

    print("enriched_df count:",len(enriched_df))
    print(enriched_df.head())

    enriched_df.to_csv(output_csv, index=False)
    print(f" Weather-enriched CSV saved to: {output_csv}")


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_enrich_csv", type=str, required=True)
    parser.add_argument("--output_weather_csv", type=str, required=True)
    args = parser.parse_args()

    enrich_with_weather(args.input_enrich_csv, args.output_weather_csv)