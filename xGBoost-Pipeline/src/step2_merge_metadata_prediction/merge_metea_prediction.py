import argparse
import pandas as pd

def main(prediction_csv, original_csv, merged_csv):
    pred_df = pd.read_csv(prediction_csv)
    meta_df = pd.read_csv(original_csv)

     # Clean any trailing spaces from image_name columns to avoid merge issues
    meta_df['image_name'] = meta_df['image_name'].str.strip()
    pred_df['image_name'] = pred_df['image_name'].str.strip()

    
    # Inner join on image_name to combine metadata and predictions
    merged_df = pd.merge(meta_df, pred_df, on="image_name", how="inner")

    merged_df.to_csv(merged_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_csv', type=str, required=True)
    parser.add_argument('--original_csv', type=str, required=True)
    parser.add_argument('--merged_csv', type=str, required=True)
    args = parser.parse_args()
    main(args.prediction_csv, args.original_csv, args.merged_csv)
