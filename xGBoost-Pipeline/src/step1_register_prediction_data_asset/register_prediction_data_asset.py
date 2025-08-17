#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
import argparse
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str)
parser.add_argument("--asset_name", type=str)
parser.add_argument("--registered_asset_uri", type=str)
args = parser.parse_args()

# Connect to Azure ML workspace
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="***",
    resource_group_name="r***",
    workspace_name="uofr-***t"
)

# Register the data asset
data_asset = Data(
    path=args.file_path,
    type="uri_file",
    name=args.asset_name,
    description="Prediction Output CSV registered as data asset",
)

registered_asset = ml_client.data.create_or_update(data_asset)

# Write the asset ID to the output path
with open(args.registered_asset_uri, "w") as f:
    f.write(registered_asset.id)
