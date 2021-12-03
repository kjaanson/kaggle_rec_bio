from azureml.core import Experiment
from azureml.core import Workspace, Run
from azureml.core import Environment
from azureml.core import Dataset, Datastore

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import ScriptRunConfig
from azureml.core.compute import ComputeInstance

workspace = Workspace.from_config()
instance = ComputeTarget(workspace=workspace, name="cpu-compute-low")
dataset_v5 = Dataset.get_by_name(workspace, name="recursionbio_zip", version=5)
dataset_target = Dataset.get_by_name(workspace, name="recursionbio_zip")
tf_env = Environment.get(workspace=workspace, name="AzureML-TensorFlow-2.3-GPU")
tf_env = tf_env.clone(new_name="extraction")
tf_env.python.conda_dependencies.add_pip_package("tqdm")

extract_script = ScriptRunConfig(
    source_directory="./scripts",
    script="extract_data.py",
    arguments=[
        "--data-file",
        dataset_v5.as_mount(),
        "--output-path",
        dataset_target.as_mount(),
    ],
    compute_target=instance,
    environment=tf_env,
)

Experiment(workspace=workspace, name="data-extraction").submit(extract_script)
