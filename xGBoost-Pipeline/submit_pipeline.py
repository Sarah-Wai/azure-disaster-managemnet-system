#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace
from pipeline import xgboost_pipeline  # pipeline.py should export this function

# Connect to your ML workspace
ws = Workspace.from_config()
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=ws.subscription_id,
    resource_group_name=ws.resource_group,
    workspace_name=ws.name
)
# Instantiate the pipeline
pipeline_job = xgboost_pipeline()

# Submit the job to Azure ML
submitted_job = ml_client.jobs.create_or_update(pipeline_job)

print(f"Pipeline submitted. View it at: https://ml.azure.com/runs/{submitted_job.name}")

