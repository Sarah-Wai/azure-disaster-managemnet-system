#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
from azure.ai.ml.entities import CommandJob
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, load_component,Output,command
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

# -------------------------------
# 1. Load component from YAML
# -------------------------------
register_prediction_component = load_component(
    source="src/step1_register_prediction_data_asset/component.yaml"
)
merge_component = load_component(
    source="src/step2_merge_metadata_prediction/component.yaml"
)
enrich_country_component = load_component(
    source="src/step3_enrich_data_creation/enrich_component.yaml"
)
enrich_weather_component = load_component(
    source="src/step4_enrich_weather/enrich_weather_component.yaml"
)
enrich_population_component = load_component(
    source="src/step5_enrich_population/enrich_population_component.yaml"
)
train_xgboost_component=load_component(
    source="src/step6_xGBoost_Train/train_xgboost_component.yaml"
)
push_to_powerbi_component=load_component(
    source="src/step7_a_deploy_to_powerbi/push_to_powerbi_component.yaml"
)
train_aks_component=load_component(
    source="src/step7_b_deploy_AKS/step7_deploy_to_aks.yaml"
)
generate_sas_component=load_component(
    source="src/step7_c_deploy_git/step7_c_deploy_git_component.yaml"
) 
github_token = "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" # Make sure this is set in your Azure ML environment

# Construct authenticated GitHub URL
github_username = "Sarah-Wai"
repo_name = "disaster-dashboard-repo"
authenticated_repo_url = f"https://{github_token}@github.com/{github_username}/{repo_name}.git"

# -------------------------------
# 2. Define pipeline
# -------------------------------
@pipeline(default_compute="wai-cpu")
def xgboost_pipeline():
    
    merge_step = merge_component(
        prediction_csv=Input(type="uri_file", path="azureml:prediction_csv_asset:3"), #register_step.outputs.registered_asset_uri,
        original_csv=Input(type="uri_file", path="azureml:xBD_test_train:1"),
    ) 
    enrich_country_step = enrich_country_component(
        merged_csv=merge_step.outputs.merged_csv,
        opencage_api_key="da5a0ea687f24539af2d0225609e1a5a"  # Ideally from a secure source
    ) 
   
    enrich_weather_step = enrich_weather_component(
        input_enrich_csv=enrich_country_step.outputs.enriched_csv, #enrich_population_step.outputs.output_population_csv,
    )

    enrich_population_step = enrich_population_component(
        input_enrich_weather_csv=enrich_weather_step.outputs.output_weather_csv,
        tif_input_folder=Input(
            type="uri_folder",
            path="azureml:population_tiff_data:1"  # Replace with your TIF folder DataAsset
        ),
        status="Local"  # Can be "Remote" if no .tif files are provided
    ) 
     
    xgboost_train_step = train_xgboost_component(
        training_data=enrich_population_step.outputs.output_population_csv
    )

    train_aks_step=train_aks_component(
        model_folder=xgboost_train_step.outputs.model_dir,
        model_name="xgboost_risk_model",
        aks_name="xgboost-aks",
        endpoint_name="xgboost-risk-endpoint"

    )
    push_powerbi_step = push_to_powerbi_component(
    input_csv=xgboost_train_step.outputs.powerbi_output,
    power_bi_url="https://api.powerbi.com/beta/3233ffa1-ea05-4445-8958-6b6744723147/datasets/158c89bb-8c07-49b3-bff9-40963d2c3e33/rows?key=nNDJDOb4WBeZmUdNc1zQ3gyEAHXAFvQa3ggM3fbba1AVGgEmz2FTmuRgS8fa2ysJlGgAKqMwTeeHEFNiKnBy+A=="
    )
 
    sas_github_step = generate_sas_component(
        pipeline_output_csv_url="https://uofrmlstudent1972267660.blob.core.windows.net/azureml-blobstore-c1ea77a6-69dd-40f4-b128-0361949bd439/azureml/fc31878a-b500-4766-924f-627c9ecb64ee/powerbi_output",
        github_branch="main",
        sas_expiry_days=7,
        github_username=github_username,
        repo_name=repo_name,
        github_token=github_token
    )



    return {
        "powerbi_response": push_powerbi_step.outputs.response_message,
        "endpoint_uri": train_aks_step.outputs.endpoint_uri,
        "sas_url_file": sas_github_step.outputs.sas_url_file
    }

    #"registered_data": merge_step.outputs.merged_csv,
    #"country_enriched_data": enrich_country_step.outputs.enriched_csv,
    #"weather_enriched_data": enrich_weather_step.outputs.output_weather_csv,
    #"population_enriched_data": enrich_population_step.outputs.output_population_csv,
    #"trained_output": xgboost_train_step.outputs.powerbi_output,
    #"powerbi_response": push_powerbi_step.outputs.response_message,
    #"endpoint_uri":train_aks_step.outputs.endpoint_uri
    ''' register_step = register_prediction_component(
       file_path=Input(
        type="uri_file",
        path="azureml://subscriptions/88b4cc90-e6c1-421c-9f5e-adaa14a0ed73/resourcegroups/rs-ml/workspaces/uofr-ml-student/datastores/damage_data_store/paths/predictions_data.csv"
    ),
    asset_name="prediction_csv_asset"
    )
    '''

# Load your environment variable securely

@pipeline(default_compute="wai-cpu")
def merge_only_pipeline():
    sas_step = generate_sas_component(
        pipeline_output_csv_url="https://uofrmlstudent1972267660.blob.core.windows.net/azureml-blobstore-c1ea77a6-69dd-40f4-b128-0361949bd439/azureml/fc31878a-b500-4766-924f-627c9ecb64ee/powerbi_output",
        github_branch="main",
        sas_expiry_days=7,
        github_username=github_username,
        repo_name=repo_name,
        github_token=github_token
    )
    return {"sas_step_output": sas_step.outputs.sas_url_file}
   


# -------------------------------
# 3. Run pipeline
# -------------------------------
if __name__ == "__main__":
    # Load workspace config
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())
    
    
    # Create pipeline job
    pipeline_job = xgboost_pipeline()
    
   
    # Submit the pipeline job
    submitted_job = ml_client.jobs.create_or_update(pipeline_job)
    print(f"Job submitted. Name: {submitted_job.name}")
    
    '''
    # Submit the pipeline
    merge_only_job = ml_client.jobs.create_or_update(
        merge_only_pipeline(), 
        experiment_name="sas_github_step")
    print(f"Submitted pipeline job: {merge_only_job.name}")
    '''

