import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data

def main(predictions_path: str):
  
    # Connect to Azure ML workspace
    ml_client = MLClient(
         credential=DefaultAzureCredential(),
             subscription_id="***",
            resource_group_name="r***",
            workspace_name="uofr-***t"
    )

    data_asset = Data(
        name="prediction_csv_asset",
        description="Train+Test combined predictions from ResNet50",
        path=os.path.join(predictions_path, "combined_predictions.csv"),
        type="uri_file"
    )

    registered_asset=ml_client.data.create_or_update(data_asset)
    print(" Registered prediction_csv_asset")

    return registered_asset
   


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--registered_asset_uri", type=str)
    args = parser.parse_args()

    registered_asset = main(args.predictions_path)

    # Write the asset ID to the output path
    with open(args.registered_asset_uri, "w") as f:
        f.write(registered_asset.id)
