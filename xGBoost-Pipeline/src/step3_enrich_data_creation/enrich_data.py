import argparse
import pandas as pd
import requests
from collections import defaultdict
from tqdm import tqdm

# -----------------------------
# Parse arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--merged_csv", type=str)
parser.add_argument("--enriched_csv", type=str)
parser.add_argument("--opencage_api_key", type=str)
args = parser.parse_args()
# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(args.merged_csv)

# -----------------------------
# Geocoding Function
# -----------------------------
def get_region(lat, lon, key):
    url = "https://api.opencagedata.com/geocode/v1/json"
    params = {"q": f"{lat},{lon}", "key": key}

    try:
        resp = requests.get(url, params=params).json()
        results = resp.get("results", [])
        if not results:
            return {"country": None, "country_code": None}

        components = results[0].get("components", {})
         # Fallback logic for region
        region = (
            components.get("region") or
            components.get("state") or
            components.get("county") or
            components.get("state_district") or
            components.get("municipality")
        )
        return {
            "country": components.get("country"),
            "region": region,
            "country_code": components.get("ISO_3166-1_alpha-3") or components.get("country_code")
        }

    except Exception as e:
        print("Error:", e)
        return {"country": None,"region":None, "country_code": None}

# -----------------------------
# Geocode Once Per Disaster
# -----------------------------
disaster_locations = df.groupby("disaster")[["lat", "lon"]].first().to_dict("index")

disaster_region_map = {}
for disaster, coords in tqdm(disaster_locations.items(), desc="Geocoding disasters"):
    lat, lon = coords["lat"], coords["lon"]
    disaster_region_map[disaster] = get_region(lat, lon, args.opencage_api_key)

# -----------------------------
# Apply Geocoding Results
# -----------------------------
df["country"] = df["disaster"].map(lambda d: disaster_region_map[d]["country"])
df["country_code"] = df["disaster"].map(lambda d: disaster_region_map[d]["country_code"])
df["region"] = df["disaster"].map(lambda d: disaster_region_map[d]["region"])
# -----------------------------
# Save Enriched Data
# -----------------------------
df.to_csv(args.enriched_csv, index=False)
print("Enrichment complete: country, region and country_code added.")
