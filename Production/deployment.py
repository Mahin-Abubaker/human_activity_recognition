import uuid
import logging
import argparse
from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential
from azure.ai.ml.entities import Model, ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.ai.ml.constants import AssetTypes

logging.basicConfig(level=logging.DEBUG)

class ModelDeployer:
    def __init__(self, subscription_id, resource_group, workspace_name, credential=None):
        # Initialize the ML Client
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.credential = credential or AzureCliCredential()  # Default to Azure CLI credentials if not provided

        self.client = MLClient(
            credential=self.credential,
            subscription_id=self.subscription_id,
            resource_group_name=self.resource_group,
            workspace_name=self.workspace_name,
        )

    def register_model(self, job_name):
        # Register the model from the given job
        model_path = f"azureml://jobs/{job_name}/outputs/artifacts/paths/model/"
        model = Model(
            path=model_path,
            name="human_action_classify_model",
            description="Model created from the job run.",
            type=AssetTypes.MLFLOW_MODEL,
        )

        # Register or update the model
        self.client.models.create_or_update(model)
        print(f"Model '{model.name}' successfully registered.")
        return model.name

    def get_latest_model_version(self, model_name):
        # Retrieve the latest version of the registered model
        versions = [int(m.version) for m in self.client.models.list(name=model_name)]
        latest_version = max(versions)
        print(f"Fetched latest model version: {latest_version}")
        return latest_version

    def get_or_create_endpoint(self):
        # Generate a unique endpoint name using a UUID
        endpoint_name = "hac-endpoint"  # You can use a static name or a UUID-based name

        # Check if the endpoint already exists
        try:
            endpoint = self.client.online_endpoints.get(name=endpoint_name)
            print(f"Endpoint '{endpoint_name}' already exists.")
        except Exception as e:
            # If the endpoint doesn't exist, create it
            print(f"Endpoint '{endpoint_name}' does not exist. Creating a new endpoint.")
            endpoint = ManagedOnlineEndpoint(
                name=endpoint_name,
                description="This is the online endpoint for human activity classification.",
                auth_mode="key",
                tags={"dataset": "credit_defaults"},
            )
            endpoint = self.client.online_endpoints.begin_create_or_update(endpoint).result()
            print(f"Endpoint '{endpoint.name}' created with state: {endpoint.provisioning_state}")
        
        return endpoint

    def deploy_model_to_endpoint(self, endpoint_name, model_name, model_version):
        # Get the model to deploy
        model = self.client.models.get(name=model_name, version=model_version)

        # Define the deployment configuration
        deployment = ManagedOnlineDeployment(
            name="hac-model-blue",
            endpoint_name=endpoint_name,
            model=model,
            instance_type="Standard_D2as_v4",
            instance_count=1,
        )

        # Deploy the model
        deployment = self.client.online_deployments.begin_create_or_update(deployment).result()

        # Set 100% traffic to the "blue" deployment
        endpoint = self.client.online_endpoints.get(name=endpoint_name)
        endpoint.traffic = {"blue": 100}
        self.client.online_endpoints.begin_create_or_update(endpoint).result()

        print(f"Model deployed to endpoint '{endpoint_name}' with traffic routed to 'blue'.")
        return deployment

if __name__ == "__main__":
    # Get the arguments for job name
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True, help="Experiment name to register the model")
    args = parser.parse_args()

    # Azure ML Workspace details
    subscription_id = '52874332-1ce3-4393-a130-2534bbfd30f4'
    resource_group = "har_resource_group"
    workspace_name = "har_workspace"

    # Initialize the ModelDeployer class
    deployer = ModelDeployer(subscription_id, resource_group, workspace_name)

    # Register the model
    model_name = deployer.register_model(args.job_name)

    # Get the latest version of the model
    latest_version = deployer.get_latest_model_version(model_name)

    # Get or create an online endpoint
    endpoint = deployer.get_or_create_endpoint()

    # Deploy the model to the endpoint
    deployer.deploy_model_to_endpoint(endpoint.name, model_name, latest_version)
