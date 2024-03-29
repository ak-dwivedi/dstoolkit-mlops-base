"""Functions to orchestrate mlops pipeline"""

from aml_utils import workspace, compute, config
from azureml.core import Environment, Experiment , Datastore
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline, StepSequence, PipelineData
from dotnetcore2 import runtime


def main(model_name,model_name1,model_name2,service_name,compute_config_file,environment_path,aks_target_name, pipeline_name, compute_name, account_name, config_path, env_path):
    """orchestrating pipeline in this function"""
    runtime.version = ("18", "10", "0")

    ws_aml = workspace.retrieve_workspace()
    datastore_name = "workspaceblobstore"
    base_path = config.get_root_path()
    config_path = base_path / config_path
    env_path = base_path / env_path
    compute_config_file = base_path / compute_config_file
    environment_path = base_path / environment_path
    compute_target = compute.get_compute_target(ws_aml, compute_name, config_path)

    env = Environment.from_conda_specification(name='reco-data-image', file_path=env_path)
    run_config = RunConfiguration()
    run_config.environment = env

    # datastore = Datastore.get(ws_aml, datastore_name) if datastore_name else ws_aml.get_default_datastore()
    datastore = ws_aml.get_default_datastore()

    print("connected to workspace successfully")

    src_path = base_path

    anonymouse_model = PythonScriptStep(
        name="deploy_anonymouse_model",
        source_directory=src_path,
        script_name="deploy_model_anonymous_usr.py",
        compute_target=compute_target,
        arguments=[
            '--model_name', model_name,
            '--model_name1', model_name1,
            '--model_name2', model_name2,
            '--service_name', service_name,
            '--compute_config_file', compute_config_file,
            '--environment_path', environment_path,
            '--aks_target_name', aks_target_name,
            '--account_name', account_name
        ],
        runconfig=run_config,
        allow_reuse=False
    )

    registered_model = PythonScriptStep(
        name="deploy_registered_model",
        source_directory=src_path,
        script_name="deploy_model_reg_user.py",
        compute_target=compute_target,
        arguments=[
            '--model_name', model_name,
            '--model_name1', model_name1,
            '--model_name2', model_name2,
            '--service_name', service_name,
            '--compute_config_file', compute_config_file,
            '--environment_path', environment_path,
            '--aks_target_name', aks_target_name,
            '--account_name', account_name
        ],
        runconfig=run_config,
        allow_reuse=False
    )


    pipeline_steps = StepSequence(
        steps=[
            anonymouse_model,
            registered_model
        ]
    )

    pipeline = Pipeline(
        workspace=ws_aml,
        steps=pipeline_steps
    )

    pipeline.validate()

    run = Experiment(ws_aml, name=pipeline_name).submit(pipeline)
    # run.wait_for_completion(show_output=True)


if __name__ == "__main__":
    main(
        pipeline_name=config.get_env_var("AML_TRAINING_PIPELINE5"),
        compute_name=config.get_env_var("AML_TRAINING_COMPUTE"),
        config_path=config.get_env_var("TRAINING_CONFIG_PATH"),
        env_path=config.get_env_var("AML_TRAINING_ENV_PATH"),
        model_name=config.get_env_var("AML_MODEL_NAME"),
        model_name1=config.get_env_var("AML_MODEL_NAME1"),
        model_name2=config.get_env_var("AML_MODEL_NAME2"),
        service_name=config.get_env_var("AML_WEBSERVICE"),
        compute_config_file=config.get_env_var("CONFIG_PATH"),
        environment_path=config.get_env_var("AML_REALTIMEINFERENCE_ENV_PATH"),
        aks_target_name=config.get_env_var("AKS_COMPUTE"),
        account_name=config.get_env_var("BLOB_ACCOUNT_NAME")
        )
