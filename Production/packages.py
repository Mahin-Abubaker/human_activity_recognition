from azureml.core import Environment, Workspace
from azureml.core.conda_dependencies import CondaDependencies

# Create a custom environment
myenv = Environment(name="custom-environment")

# Get the workspace
wrksp = Workspace.get(name="har_workspace",
                      subscription_id="52874332-1ce3-4393-a130-2534bbfd30f4",
                      resource_group="har_resource_group")

# Create a conda dependencies object
conda_dep = CondaDependencies()

# Add required dependencies
conda_dep.add_conda_package("scikit-learn")
conda_dep.add_conda_package("pandas")
conda_dep.add_pip_package("azure-ai-ml")

# Set the conda dependencies for the environment
myenv.python.conda_dependencies = conda_dep

# Register the environment with the workspace
myenv.register(workspace=wrksp)
