"""
Script to collect data from MLFlow

Author: Vivek Katial
"""

import os
import argparse
import mlflow
import yaml
import logging
import pandas as pd

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Initialize logging
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

    # Parsing arguments from CLI
    parser = argparse.ArgumentParser(description="Collect data from MLFlow.")
    parser.add_argument("-e", "--experiment", type=str, required=True, help="Name of the experiment")
    args = parser.parse_args()
    experiment_name = args.experiment

    # Log experiment details
    logging.info(f"Experiment: {experiment_name}")
    logging.info(f"MLFlow Tracking Server URI: {mlflow.get_tracking_uri()}")
    logging.info(f"MLFlow Version: {mlflow.version.VERSION}")
    logging.info(f"MLFlow Timeout: {os.getenv('MLFLOW_HTTP_REQUEST_TIMEOUT')}")

    # Read data from configuration file
    with open("config/mlflow-tracking-server.yml", 'r') as file:
        mlflow_config = yaml.safe_load(file)

    # Connect to MLflow
    tracking_uri = mlflow_config["mlflow"]["tracking_server_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    logging.info(f'Connecting to MLFlow Tracking Server at URI: "{tracking_uri}" on Experiment: "{experiment_name}"')

    # Find experiment by name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logging.error(f"Experiment '{experiment_name}' not found.")
        return

    logging.info(f"Downloading data from Experiment: {experiment_name}")

    instance_classes = [
        "four_regular_graph", "geometric", "nearly_complete_bi_partite", 
        "power_law_tree", "three_regular_graph", "uniform_random", "watts_strogatz_small_world"
    ]

    # Prepare DataFrame container for all runs
    all_runs = []

    # Loop through each instance class and instance sizes (4, 6, 8, 10, 12, 14)
    for instance_size in [4, 6, 8, 10, 12, 14]:
        for instance_class in instance_classes:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
                filter_string=f"params.instance_size = '{instance_size}' and params.instance_class='{instance_class}'",
                max_results=5000
            )
            if not runs.empty:
                all_runs.append(runs)
                logging.info(f"Found {len(runs)} runs for instance_size: {instance_size} and instance_class: {instance_class}")

    # Concatenate all runs into a single DataFrame, if not empty
    if all_runs:
        df_results = pd.concat(all_runs, ignore_index=True)
        df_results_filename = f"data/d_{experiment_name}_all_runs.csv"
        df_results.to_csv(df_results_filename, index=False)
        logging.info(f'Writing runs data to "{df_results_filename}"')
    else:
        logging.info(f"No runs found for Experiment: {experiment_name}")

if __name__ == "__main__":
    main()
