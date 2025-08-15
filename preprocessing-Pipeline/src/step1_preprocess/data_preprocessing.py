import os
import json
import argparse
from glob import glob
from PIL import Image, UnidentifiedImageError
from shapely.wkt import loads
import csv

# Damage subtype to class mapping
DAMAGE_MAP = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3
}

def get_damage_class(json_path):
    """Extract max damage class from xBD JSON label."""
    with open(json_path, "r") as f:
        data = json.load(f)
    damages = [
        DAMAGE_MAP.get(feature["properties"].get("subtype"), -1)
        for feature in data.get("features", {}).get("xy", [])
    ]
    if not damages:
        return -1
    return max(damages)

def process_images(images_dir, labels_dir, output_dir):
    """
    Process post-disaster images: resize to 224x224 and
    save into folders by damage class.
    """
    os.makedirs(output_dir, exist_ok=True)

    for img_path in glob(os.path.join(images_dir, "*_post_disaster.png")):
        filename = os.path.basename(img_path)
        base_name = filename.replace("_post_disaster.png", "")
        label_path = os.path.join(labels_dir, base_name + "_post_disaster.json")

        if not os.path.exists(label_path):
            print(f"Skipping {filename}: JSON label not found.")
            continue

        damage_class = get_damage_class(label_path)
        if damage_class == -1:
            print(f"Skipping {filename}: No valid damage class found.")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except (OSError, UnidentifiedImageError) as e:
            print(f"Skipping {filename}, cannot open image. Error: {e}")
            continue

        img_resized = img.resize((224, 224))

        # Save into class folder
        class_dir = os.path.join(output_dir, f"class_{damage_class}")
        os.makedirs(class_dir, exist_ok=True)
        save_path = os.path.join(class_dir, filename)
        img_resized.save(save_path)

    print(f"Images processed and saved to {output_dir}")

def create_train_metadata(images_dir, labels_dir, csv_output):
    """
    Create metadata CSV for training set, including lat/lon and source field.
    """
    csv_data = []

    for image_filename in os.listdir(images_dir):
        if not image_filename.endswith(".png"):
            continue

        img_path = os.path.join(images_dir, image_filename)
        label_filename = image_filename.replace(".png", ".json")
        label_path = os.path.join(labels_dir, label_filename)

        entry = {
            "image_name": image_filename,
            "disaster": None,
            "disaster_type": None,
            "capture_date": None,
            "damage_class": -1,
            "lat": -1,
            "lon": -1,
            "source": "train"
        }

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                label_data = json.load(f)

            metadata = label_data.get("metadata", {})
            entry["disaster"] = metadata.get("disaster")
            entry["disaster_type"] = metadata.get("disaster_type")
            entry["capture_date"] = metadata.get("capture_date")

            # Damage classification
            features = label_data.get("features", {}).get("xy", [])
            damage_level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            for feat in features:
                props = feat.get("properties", {})
                damage = props.get("subtype")
                if damage == "no-damage":
                    damage_level_counts[0] += 1
                elif damage == "minor-damage":
                    damage_level_counts[1] += 1
                elif damage == "major-damage":
                    damage_level_counts[2] += 1
                elif damage == "destroyed":
                    damage_level_counts[3] += 1

            if sum(damage_level_counts.values()) > 0:
                entry["damage_class"] = max(damage_level_counts, key=damage_level_counts.get)

            # Latitude and longitude from WKT geometry
            features_latlon = label_data.get("features", {}).get("lng_lat", [])
            if features_latlon:
                for feat in features_latlon:
                    if "wkt" in feat:
                        try:
                            polygon = loads(feat["wkt"])
                            centroid = polygon.centroid
                            entry["lon"] = centroid.x
                            entry["lat"] = centroid.y
                            break  # Use first valid polygon
                        except Exception as e:
                            print(f"Error loading WKT for {image_filename}: {e}")

        else:
            print(f"Label not found for {image_filename}")

        csv_data.append(entry)

    if csv_data:
        with open(csv_output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"CSV created: {csv_output} ({len(csv_data)} rows)")
    else:
        print("No data to write.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_xBDdata", type=str, required=True, help="Path to raw xBD data containing train/ and test/ folders")
    parser.add_argument("--train_clean_data", type=str, required=True, help="Output folder for cleaned train images")
    parser.add_argument("--test_clean_data", type=str, required=True, help="Output folder for cleaned test images")
    parser.add_argument("--oridinary_meta_data", type=str, required=True, help="Output CSV for training metadata")

    args = parser.parse_args()

    # Define train and test directories from raw data
    train_images_dir = os.path.join(args.raw_xBDdata, "train/images")
    train_labels_dir = os.path.join(args.raw_xBDdata, "train/labels")
    test_images_dir = os.path.join(args.raw_xBDdata, "test/images")
    test_labels_dir = os.path.join(args.raw_xBDdata, "test/labels")

    # Process and save cleaned datasets
    process_images(train_images_dir, train_labels_dir, args.train_clean_data)
    process_images(test_images_dir, test_labels_dir, args.test_clean_data)

    # Create metadata CSV for train set
    create_train_metadata(train_images_dir, train_labels_dir, args.oridinary_meta_data)

    print("Pipeline job completed successfully.")
