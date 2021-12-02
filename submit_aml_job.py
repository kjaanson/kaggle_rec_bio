# %%
from azureml.core import Experiment
from azureml.core import Workspace, Run
from azureml.core import Environment
from azureml.core import Dataset, Datastore

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import ScriptRunConfig

# %%
from azureml.core.compute import ComputeInstance

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser("aml_submit")
    parser.add_argument(
        "--aml_compute_target",
        type=str,
        help="Name of the Azure ML compute target to use",
        dest="aml_compute_target",
        default="gpu-v100-low"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the Azure ML dataset to use",
        dest="dataset",
        default="recusionbio"
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        help="Name of the conda environment file to use",
        dest="conda_env_file",
        default="conda_env.yml"
    )
    parser.add_argument(
        "--script",
        type=str,
        help="Name of the script to run",
        dest="script",
        default="training.py"
    )
    parser.add_argument(
        "--source-directory",
        type=str,
        help="Name of the source directory to use",
        dest="source_directory",
        default="./scripts"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name of the experiment to use",
        dest="experiment_name",
        default="recusionbio"
    )
    
    args = parser.parse_args()


    workspace = Workspace.from_config()
    instance = ComputeTarget(workspace=workspace, name=args.aml_compute_target)

    dataset = Dataset.get_by_name(workspace, name='recursionbio')

    tf_env = Environment.from_conda_specification(name="regbio-tf-env", file_path=args.conda_env_file)

    train_scr = ScriptRunConfig(
        source_directory=args.source_directory,
        script=args.script,
        arguments=[
                '--data-path', dataset.as_mount(),
                '--epochs', 2000,
                '--batch', 24,
                ],
        compute_target=instance,
        environment=tf_env
    )



    run = Experiment(workspace=workspace, name=args.experiment_name).submit(train_scr)

    run.wait_for_completion(show_output=True)




