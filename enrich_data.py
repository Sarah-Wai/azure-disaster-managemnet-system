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
# Load data
# -----------------------------
df = pd.read_csv(args.merged_csv)

# -----------------------------
# Geocoding Functions
# -----------------------------
def get_region(lat, lon, key):
    url = "https://api.opencagedata.com/geocode/v1/json"
    params = {"q": f"{lat},{lon}", "key": key}

    try:
        resp = requests.get(url, params=params).json()
        results = resp.get("results", [])
        if not results:
            return {"country": None,  "country_code": None}

        components = results[0].get("components", {})
        return {
            "country": components.get("country"),
            #"region": components.get("state", components.get("region", None)),
            "country_code": components.get("ISO_3166-1_alpha-3") or components.get("country_code")
        }

    except Exception as e:
        print("Error:", e)
        return {"country": None, "region": None, "country_code": None}

# -----------------------------
# Caching Setup
# -----------------------------
region_cache = defaultdict(dict)

def get_region_with_cache(disaster, lat, lon, key):
    cache_key = (lat, lon)
    if cache_key in region_cache[disaster]:
        return region_cache[disaster][cache_key]

    result = get_region(lat, lon, key)
    region_cache[disaster][cache_key] = result
    return result

# -----------------------------
# Apply Geocoding
# -----------------------------
tqdm.pandas(desc="Geocoding with caching")

df[['country', 'country_code']] = df.progress_apply(
    lambda row: pd.Series(get_region_with_cache(row['disaster'], row['lat'], row['lon'], args.opencage_api_key)),
    axis=1
)

# -----------------------------
# Save enriched data
# -----------------------------
df.to_csv(args.enriched_csv, index=False)
print("✅ Enrichment complete: country, region, and country_code added.")
