import pandas as pd
import argparse
import os
import sys
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from tqdm import tqdm
import logging
import gc

# --- Setup Azure ML-friendly logging ---
logging.basicConfig(level=logging.INFO)

def download_tif(country_code, year, download_dir):
    tif_filename = f"{country_code}_{year}.tif"
    local_path = os.path.join(download_dir, tif_filename)

    if os.path.exists(local_path):
        logging.debug(f"File already exists: {tif_filename}")
        return local_path

    url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020/{year}/{country_code}/{country_code.lower()}_ppp_{year}.tif"
    try:
        response = requests.get(url, stream=True, timeout=20)
        if response.status_code == 200:
            os.makedirs(download_dir, exist_ok=True)
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            logging.info(f"Downloaded: {tif_filename}")
            return local_path
        else:
            logging.error(f"Failed to download {url} (Status code: {response.status_code})")
            return None
    except Exception as e:
        logging.error(f"Download error for {url}: {e}")
        return None

def get_tif_path(country_code, year, tif_input_folder, status):
    tif_filename = f"{country_code}_ppp_{year}.tif"
    if status == "Local":
        path = os.path.join(tif_input_folder, tif_filename)
        logging.info(f"{tif_filename} Path : {path}")
        if os.path.exists(path):
            logging.info(f"{tif_filename} exists in path.")
            return path
        else:
            logging.warning(f"TIF not found in tif_input_folder: {path}")
            return None
    else:  # Remote mode
        return download_tif(country_code, year, download_dir="worldpop_cache")

def main(input_csv, output_csv, tif_input_folder=None, status="Remote"):
    logging.info(f"Reading input CSV: {input_csv}")
    df_prediction = pd.read_csv(input_csv)

    # Extract year and clean invalid rows
    df_prediction['year'] = pd.to_datetime(df_prediction['capture_date'], errors='coerce').dt.year
    df = df_prediction.dropna(subset=["lat", "lon", "year", "country_code"])
    logging.info(f"Original rows: {len(df_prediction)}, Cleaned rows: {len(df)}")

    df = df.drop_duplicates(subset=["image_name"]).reset_index(drop=True)
    logging.info(f"Unique images: {len(df)}")

    df['population_density'] = None

    grouped = df.groupby(["country_code", "year"])

    for (country_code, year), group in tqdm(grouped, desc="Processing population", file=sys.stdout):
        tif_path = get_tif_path(country_code, year, tif_input_folder, status)
        if not tif_path:
            logging.warning(f"Skipping group for {country_code} {year} due to missing TIF")
            continue

        try:
            with rasterio.open(tif_path) as src:
                for idx in group.index:
                    lat, lon = df.at[idx, "lat"], df.at[idx, "lon"]
                    try:
                        row, col = src.index(lon, lat)
                        # Only read 1x1 pixel window to save memory
                        pop = src.read(1, window=((row, row+1), (col, col+1)), resampling=Resampling.bilinear)[0, 0]
                        df.at[idx, "population_density"] = round(float(pop), 4)
                    except Exception as e:
                        logging.warning(f"Failed to read pixel at ({lat}, {lon}): {e}")
        except Exception as e:
            logging.error(f"Failed to open {tif_path}: {e}")

        # Manual garbage collection to release memory
        gc.collect()

    df.to_csv(output_csv, index=False)
    logging.info(f"Saved output CSV: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_enrich_weather_csv", type=str, required=True, help="Path to input enriched CSV")
    parser.add_argument("--output_population_csv", type=str, required=True, help="Path to output CSV with population")
    parser.add_argument("--tif_input_folder", type=str, required=False, help="Path to local/mounted folder with .tif files")
    parser.add_argument("--status", type=str, default="Remote", help="'Local' to use tif_input_folder, 'Remote' to download TIFs")
    args = parser.parse_args()

    main(
        input_csv=args.input_enrich_weather_csv,
        output_csv=args.output_population_csv,
        tif_input_folder=args.tif_input_folder,
        status=args.status
    )
