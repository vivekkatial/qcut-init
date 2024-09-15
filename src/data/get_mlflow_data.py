"""
Script to collect data from MLFlow and download optimized_parameters.json

Author: Vivek Katial (modified)
"""

import os
import argparse
import mlflow
from mlflow.tracking import MlflowClient
import logging
import json
import shutil
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

def process_runs(client, runs, output_dir):
    for run in runs:
        run_id = run.info.run_id

        os.makedirs(output_dir, exist_ok=True)

        # Download the artifact
        artifact_path = "optimized_parameters.json"
        output_path = os.path.join(output_dir, f"{run_id}_optimized_parameters.json")
        
        # If the file already exists, skip downloading
        if os.path.exists(output_path):
            logging.info(f"File already exists: {output_path}")
            continue
        try:
            local_path = client.download_artifacts(run_id, artifact_path, output_dir)
            # Rename the file to include the run_id
            shutil.move(local_path, output_path)
            logging.info(f"Downloaded and saved artifact for run {run_id} to {output_path}")

        except Exception as e:
            logging.error(f"Failed to download artifact for run {run_id}: {e}")

def main():
    # Initialize logging
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

    # Parsing arguments from CLI
    parser = argparse.ArgumentParser(description="Collect data from MLFlow and download optimized_parameters.json.")
    parser.add_argument("-e", "--experiment", type=str, required=True, help="Name of the experiment")
    parser.add_argument("-o", "--output", type=str, default="optimized_parameters", help="Output directory for downloaded files")
    parser.add_argument("-n", "--num_nodes", type=int, default=12, help="Number of nodes to filter")
    args = parser.parse_args()

    experiment_name = args.experiment
    output_dir = args.output
    num_nodes = args.num_nodes

    # Log experiment details
    logging.info(f"Experiment: {experiment_name}")
    logging.info(f"MLFlow Tracking Server URI: {mlflow.get_tracking_uri()}")
    logging.info(f"MLFlow Version: {mlflow.version.VERSION}")
    logging.info(f"MLFlow Timeout: {os.getenv('MLFLOW_HTTP_REQUEST_TIMEOUT')}")

    # Initialize MLflow client
    client = MlflowClient()

    # Find experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logging.error(f"Experiment '{experiment_name}' not found.")
        return

    experiment_id = experiment.experiment_id
    logging.info(f"Retrieved experiment ID: {experiment_id}")

    all_runs = []
    next_page_token = None
    while True:
        logging.info(f"Fetching runs with token: {next_page_token}")
        page_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            max_results=1000,
            page_token=next_page_token
        )
        all_runs.extend(page_runs)
        next_page_token = page_runs.token
        if not next_page_token:
            break

    logging.info(f"Total runs fetched: {len(all_runs)}")
    # Filter out evolved runs
    logging.info(f"Processing {len(all_runs)}")
    process_runs(client, all_runs, output_dir)
    # Convert MLFlow runs to a DataFrame
    runs_data = []
    for run in all_runs:
        run_id = run.info.run_id
        run_data = run.data
        run_params = run_data.params
        run_metrics = run_data.metrics
        run_tags = run_data.tags

        # Filter runs with num_nodes
        if "num_nodes" in run_params:
            if int(run_params["num_nodes"]) == num_nodes:
                runs_data.append({
                    "run_id": run_id,
                    "num_nodes": run_params["num_nodes"],
                    "params": run_params,
                    "metrics": run_metrics,
                    "tags": run_tags
                })
    
    runs = pd.DataFrame(runs_data)
    # Save the runs csv in `data` directory
    runs.to_csv(f"data/raw/{experiment_name}_runs.csv")

    logging.info("Finished processing all runs.")

if __name__ == "__main__":
    main()