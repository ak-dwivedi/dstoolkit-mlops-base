import pandas as pd
import pyarrow.parquet as pq
from azure.storage.blob import BlobServiceClient
import argparse
import logging
import logging.config
from pathlib import Path
import pandas as pd
from azureml.core import Run, Dataset
from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient
# logger = logging.getLogger()
from azureml.core import Workspace
# import logging


def main(blob_name_reco, container_name_reco, dataset_name_reco, blob_name_anonymous, container_name_anonymous, dataset_name_anonymous,account_name):
    run = Run.get_context()
    ws_aml = run.experiment.workspace
    keyvault = ws_aml.get_default_keyvault()
    account_key = keyvault.get_secret("blobaccountkey")
    recodf = load_blob_data(account_name, account_key, blob_name_reco,container_name_reco)
    anonymous_user = load_blob_data(account_name, account_key, blob_name_anonymous,container_name_anonymous)
    datastore = ws_aml.get_default_datastore()
    register_data(recodf, datastore, dataset_name_reco)
    register_data(anonymous_user, datastore, dataset_name_anonymous)


def load_blob_data(account_name, account_key, blob_name, container_name):
    ITEM = "sessiontag_contentid"

    # Create BlobServiceClient
    blob_service_client = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=account_key)

    # Get a reference to the container
    container_client = blob_service_client.get_container_client(container_name)

    # Download Parquet file from blob to local
    local_parquet_file = "downloaded_parquet_file.parquet"
    blob_client = container_client.get_blob_client(blob_name)
    with open(local_parquet_file, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

    # Read Parquet file into Pandas DataFrame
    df = pq.read_table(local_parquet_file).to_pandas()

    # Cleanup: Remove downloaded Parquet file
    import os
    os.remove(local_parquet_file)
    return df


def register_data(data, datastore, dataset_name):
    """This function is used Register the dataset to aml"""
    # dataset_name= "test_data_1"
    ds_name = Dataset.Tabular.register_pandas_dataframe(
            dataframe = data, 
            name = dataset_name, 
            description = 'Databricks dataset',
            target = datastore
        )

    # Display information about the dataset
    print(ds_name.name + " v" + str(ds_name.version) + ' (ID: ' + ds_name.id + ")")


def parse_args(args_list=None):
    """This function is used to pass the arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--blob_name_reco', type=str, required = False)
    parser.add_argument('--container_name_reco', type=str, required = False)
    parser.add_argument('--dataset_name_reco', type=str, required = False)
    parser.add_argument('--blob_name_anonymous', type=str, required = False)
    parser.add_argument('--container_name_anonymous', type=str, required = False)
    parser.add_argument('--dataset_name_anonymous', type=str, required = False)
    parser.add_argument('--account_name', type=str, required = False)
    args_parsed = parser.parse_args(args_list)
    return args_parsed


if __name__ == '__main__':
    args = parse_args()
    main(blob_name_reco = args.blob_name_reco, container_name_reco = args.container_name_reco, dataset_name_reco = args.dataset_name_reco, blob_name_anonymous = args.blob_name_anonymous, container_name_anonymous = args.container_name_anonymous, dataset_name_anonymous = args.dataset_name_anonymous,account_name = args.account_name)

