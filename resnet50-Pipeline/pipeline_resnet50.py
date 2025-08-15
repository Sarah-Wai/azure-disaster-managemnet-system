from azure.ai.ml.entities import CommandJob, AmlCompute
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, load_component, command
from azure.ai.ml.dsl import pipeline

# -------------------------------
# 0. Set compute name
# -------------------------------
compute_name = "gpu-cluster"

# -------------------------------
# 1. Authenticate and load ML client
# -------------------------------
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# -------------------------------
# 2. Ensure compute exists or create it
# -------------------------------
try:
    compute = ml_client.compute.get(compute_name)
    print(f"Found existing compute cluster: {compute_name}")
except ResourceNotFoundError:
    print(f"Compute cluster '{compute_name}' not found. Creating it now...")
    compute = AmlCompute(
        name=compute_name,
        size="STANDARD_NC4as_T4_v3",
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=300
    )
    compute = ml_client.compute.begin_create_or_update(compute).result()
    print(f"Created new compute cluster: {compute_name}")

# -------------------------------
# 3. Load components from YAML
# -------------------------------
resnet_train_component = load_component(
    source="src/train_model/train_ensemble_component.yaml"
)
register_predictions_component = load_component(
    source="src/post_register/register_asset_component.yaml"
)

# -------------------------------
# 4. Define pipeline
# -------------------------------
@pipeline(default_compute="wai-cpu")
def resnet50_pipeline():
    train_step = resnet_train_component(
        train_data=Input(type="uri_folder", path="azureml:damage_classification_data:1"),
        test_data=Input(type="uri_folder", path="azureml:damage_classification_test_data:3"),
        epoch=20,
        batch_size=64,
        learning_rate=1e-3
    )

   
    train_step.compute = compute_name
    ''''
    register_predict_step = register_predictions_component(
        predictions_path=train_step.outputs.output_dir
    )
  
    register_predict_step.compute = "wai-cpu"  # Use CPU for lightweight task
    
    return {
        "resnet50_response": register_predict_step.outputs.registered_asset_uri
    }'''

# -------------------------------
# 5. Run pipeline
# -------------------------------
if __name__ == "__main__":
    pipeline_job = resnet50_pipeline()
    submitted_job = ml_client.jobs.create_or_update(pipeline_job)
    print(f"Job submitted successfully. Name: {submitted_job.name}")
