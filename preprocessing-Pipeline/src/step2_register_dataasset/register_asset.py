import argparse

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_clean_data", type=str, required=True)
    parser.add_argument("--test_clean_data", type=str, required=True)
    parser.add_argument("--oridinary_meta_data", type=str, required=True, help="Output csv for Dataset")
    parser.add_argument("--train_asset_uri", type=str)
    parser.add_argument("--test_asset_uri", type=str)
    parser.add_argument("--original_meta_uri", type=str)
  
    args = parser.parse_args()

    train_asset_uri = "train_asset_uri"
    test_asset_uri = "train_asset_uri"
    train_asset_uri = "original_meta_uri"

    # Write the asset ID to the output path
    with open(args.train_asset_uri, "w") as f:
        f.write(train_asset_uri)
    
    with open(args.test_asset_uri, "w") as f:
        f.write(test_asset_uri)

    with open(args.original_meta_uri, "w") as f:
        f.write(train_asset_uri)