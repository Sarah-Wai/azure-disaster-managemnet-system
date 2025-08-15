from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace

ws = Workspace.from_config()
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=ws.subscription_id,
    resource_group_name=ws.resource_group,
    workspace_name=ws.name
)

custom_env = Environment(
    name="custom_sklearn_1.0_azure_ai_ml",
    description="Custom env with azure-ai-ml",
    conda_file="custom_env.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
)

ml_client.environments.create_or_update(custom_env)
