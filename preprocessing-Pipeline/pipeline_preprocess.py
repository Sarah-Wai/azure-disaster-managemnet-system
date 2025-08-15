#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from azure.ai.ml.entities import CommandJob
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, load_component,Output,command
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

# -------------------------------
# 1. Load component from YAML
# -------------------------------
data_preprocessing_component = load_component(
    source="src/step1_preprocess/data_preprocessing_component.yaml"
)
register_asset_component = load_component(
    source="src/step2_register_dataasset/register_asset_component.yaml"
)


@pipeline(default_compute="wai-cpu")
def preprocessing_pipeline():
    preprocess_step = data_preprocessing_component(
        raw_xBDdata=Input(type="uri_folder", path="azureml:xDBRawDate:1"),
    ) 
    register_predict_step = register_asset_component(
        train_clean_data=preprocess_step.outputs.train_clean_data, 
        test_clean_data=preprocess_step.outputs.test_clean_data,
        oridinary_meta_data=preprocess_step.outputs.oridinary_meta_data,
    ) 
    return {
        "train_asset_response": register_predict_step.outputs.train_asset_uri,
        "test_asset_response": register_predict_step.outputs.test_asset_uri,
        "oridinary_meta_asset_response": register_predict_step.outputs.original_meta_uri
    }



# -------------------------------
# 3. Run pipeline
# -------------------------------
if __name__ == "__main__":
    # Load workspace config
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())

    # Create pipeline job
    pipeline_job = preprocessing_pipeline()

    # Submit the pipeline job
    submitted_job = ml_client.jobs.create_or_update(pipeline_job)
    print(f"Job submitted. Name: {submitted_job.name}")

