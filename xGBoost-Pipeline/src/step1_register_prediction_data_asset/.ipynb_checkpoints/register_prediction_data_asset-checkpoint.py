#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
import argparse
from azureml.core import Workspace
import os

# Define args
parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str)
parser.add_argument("--asset_name", type=str)
args = parser.parse_args()

# Connect to ML workspace

ws = Workspace.from_config()
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=ws.subscription_id,
    resource_group_name=ws.resource_group,
    workspace_name=ws.name
)

# Register DataAsset
data_asset = Data(
    path=args.file_path,  # Full Azure Blob path
    type="uri_file",
    name=args.asset_name,
    description="Prediction Output CSV registered as data asset",
)

ml_client.data.create_or_update(data_asset)


output_path = os.environ["AZUREML_OUTPUT_registered_asset_uri"]
with open(output_path, "w") as f:
    f.write(data_asset.path)  # or asset name or resource ID

