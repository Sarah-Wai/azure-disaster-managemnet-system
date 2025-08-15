import os
import argparse
from azureml.core import Workspace, Environment, Model
from azureml.core.compute import AksCompute
from azureml.core.webservice import AksWebservice, Webservice
from azureml.core.model import InferenceConfig

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--aks_name", type=str, required=True)
parser.add_argument("--endpoint_name", type=str, required=True)
parser.add_argument("--endpoint_uri", type=str, required=False)
args = parser.parse_args()

def main():
    print("Connecting to Azure ML Workspace...")
    ws = Workspace(
        subscription_id="88b4cc90-e6c1-421c-9f5e-adaa14a0ed73",
        resource_group="rs-ml",
        workspace_name="uofr-ml-student"
    )

    print(f"Retrieving AKS compute target: {args.aks_name}")
    aks_target = AksCompute(ws, args.aks_name)

    print(f"Registering model '{args.model_name}' from folder '{args.model_folder}'...")
    model = Model.register(
        workspace=ws,
        model_path=args.model_folder,
        model_name=args.model_name
    )
    print(f"Model '{args.model_name}' registered.")

    env_name = "xgboost-pipeline-env"
    try:
        env = Environment.get(workspace=ws, name=env_name)
        print(f"Environment '{env_name}' found.")
    except Exception:
        print(f"Environment '{env_name}' not found. Creating and registering...")
        env = Environment.from_conda_specification(env_name, "environment.yml")
        env.register(workspace=ws)
        print(f"Environment '{env_name}' registered.")

    inference_config = InferenceConfig(
        entry_script="score.py",
        environment=env,
        source_directory="."
    )

    deployment_config = AksWebservice.deploy_configuration(cpu_cores=1, memory_gb=3)

    # Delete existing service if it exists
    try:
        service = Webservice(ws, name=args.endpoint_name)
        print(f"Deleting existing webservice '{args.endpoint_name}'...")
        service.delete()
        print("Deleted existing service successfully.")
    except Exception:
        print("No existing service found, continuing with deployment...")

    # Deploy new service
    print(f"Deploying new webservice '{args.endpoint_name}' to AKS...")
    service = Model.deploy(
        workspace=ws,
        name=args.endpoint_name,
        models=[model],
        inference_config=inference_config,
        deployment_config=deployment_config,
        deployment_target=aks_target,
        overwrite=True
    )
    service.wait_for_deployment(show_output=True)

    print(f"Service state: {service.state}")
    if service.state != "Healthy":
        print("Service is unhealthy. Logs:")
        print(service.get_logs())
        raise RuntimeError(f"Deployment failed or unhealthy. State: {service.state}")

    print(f"Endpoint scoring URI: {service.scoring_uri}")

    if args.endpoint_uri:
        os.makedirs(os.path.dirname(args.endpoint_uri), exist_ok=True)
        with open(args.endpoint_uri, "w") as f:
            f.write(service.scoring_uri)
        print(f"Endpoint URI saved to: {args.endpoint_uri}")

if __name__ == "__main__":
    main()
    
