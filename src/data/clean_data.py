import click
import logging
import re
import pandas as pd
import numpy as np
import warnings

from pathlib import Path
from dotenv import find_dotenv, load_dotenv


warnings.filterwarnings("ignore")

def convert_and_sort_columns(columns, prefix):
    # Define the regex pattern to extract layer number, parameter type, and parameter index
    pattern = re.compile(r'metrics\.QAOA_(\d+)_layers_optimal_([a-z]+)_(\d+)')

    # Sorting function using regex to extract sort keys
    def sort_key(col):
        match = pattern.search(col)
        if match:
            layer_num = int(match.group(1))  # Layer number as integer for numerical sorting
            param_type = match.group(2)  # Parameter type as is
            param_index = int(match.group(3))  # Parameter index as integer for numerical sorting
            return layer_num, param_type, param_index
        return (0, '', 0)  # Default sort value if pattern doesn't match

    # Sorting columns based on the defined sort key
    sorted_columns = sorted(columns, key=sort_key)

    # Renaming columns based on the new naming convention
    converted_names = []
    for seq, col in enumerate(sorted_columns, start=1):
        match = pattern.search(col)
        if match:
            layer_num = match.group(1)
            param_type = match.group(2)
            param_index = match.group(3)
            new_name = f"{prefix}_QAOA_L{layer_num}_{param_type}_{param_index}_Seq{str(seq).zfill(2)}"
            converted_names.append(new_name)
    
    return converted_names



def extract_real(x: str) -> float:
    """
    Extracts the real part of a complex number represented as a string, 
    or directly converts a numeric string to a float.

    Parameters:
    - x (str): A string representing a complex number (e.g., "(1+2j)") or a real number.

    Returns:
    - float: The real part of the complex number as a float, or the number itself if it's already real.
             Returns None if the conversion fails.

    Note:
    This function assumes the input string for complex numbers is in the form "(real+imaginaryj)".
    It will remove the imaginary part and convert the remaining real part to a float.
    """
    try:
        # Check if the string contains "+0j"
        if not re.search(r"\+.*j", x):
            return float(x)
        
        # Remove the brackets
        x = x.replace("(", "").replace(")", "")
        # Remove the imaginary part
        x = re.sub(r"\+.*$", "", x)
        
        # Convert to numeric
        x = float(x)
    except Exception as e:
        print(f"Failed to convert {x} to a real number: {e}")
        # If the conversion fails, return None
        x = None
    
    # Return the result
    return x


def apply_extract_real_to_df(data):
    """
    Applies extract_real function to each applicable cell in a DataFrame.
    
    Parameters:
    - data: pd.DataFrame, the DataFrame to process.
    
    Returns:
    - pd.DataFrame with the modifications applied.
    """
    # Define a row-wise operation function
    def apply_extract_real(row):
        for col in data.columns:
            if col.startswith("params.") and data[col].dtype == 'object':
                row[col] = extract_real(row[col])
        return row

    # Apply the function row-wise
    return data.apply(apply_extract_real, axis=1)

def find_max_column(row, approx_ratio_cols):
    # Filter to approximation ratio columns and find max value
    max_val = row[approx_ratio_cols].max()
    
    # Get all columns with the max value
    max_cols = [col for col in approx_ratio_cols if row[col] == max_val]
    
    # Select one randomly if there's more than one
    return np.random.choice(max_cols)

def fill_beta_gamma(row, columns):
    algorithm_prefix = "metrics.QAOA_"
    algorithm_suffix = "_approximation_ratio"
    algorithm = row['max_value_column'].replace(algorithm_prefix, "").replace(algorithm_suffix, "")
    
    for i in range(20):
        beta_col_name = f'metrics.QAOA_{algorithm}_optimal_beta_{i}'
        gamma_col_name = f'metrics.QAOA_{algorithm}_optimal_gamma_{i}'
        
        row[f'beta_{i}'] = row[beta_col_name] if beta_col_name in columns else np.nan
        row[f'gamma_{i}'] = row[gamma_col_name] if gamma_col_name in columns else np.nan
    
    return row



def clean_data_for_layer(data: pd.DataFrame, layer: int) -> pd.DataFrame:
    """ Clean the data for a specific layer"""
    # Identifying feature columns: All params except those with 'QAOA'
    features = [col for col in data.columns if col.startswith("params.") and "QAOA" not in col]

    
    # Initializing lists for layer-specific columns and targets
    layer_columns = []
    targets = []

    for col in data.columns:
        if col.startswith(f"metrics.QAOA_{layer}_layers_optimal_"):
            targets.append(col)
        # Check for preceding layers only if layer > 1
        elif layer > 1:
            for previous_layer in range(1, layer):
                if col.startswith(f"metrics.QAOA_{previous_layer}_layers_optimal_"):
                    layer_columns.append(col)
                    break

    converted_and_sorted_layer_columns = convert_and_sort_columns(layer_columns, "dynamic")
    converted_and_sorted_targets = convert_and_sort_columns(targets, "target")


    pattern = f"metrics\.QAOA_{layer}_layers_approximation_ratio"
    approx_ratio_cols = [col for col in data.columns if re.match(pattern, col)]
    p_success_col = f"metrics.QAOA_{layer}_layers_success_probability"


    # Ensure there are targets for the specified layer
    if not targets:
        print(f"Layer {layer} has no target gamma/beta info. Skipping.")
        return

    # Combining columns to include in the filtered DataFrame
    columns_to_include = features + layer_columns + targets + approx_ratio_cols + [p_success_col]
    filtered_data = data[columns_to_include].copy()
    # Rename feature columns to remove 'params.' prefix and have "static_feature_" prefix
    filtered_data.rename(columns=lambda x: x.replace("params.", "static_feature_"), inplace=True)

    # Rename layer columns based on converted_and_sorted_layer_columns
    filtered_data.rename(columns=dict(zip(layer_columns, converted_and_sorted_layer_columns)), inplace=True)
    # Rename target columns based on converted_and_sorted_targets
    filtered_data.rename(columns=dict(zip(targets, converted_and_sorted_targets)), inplace=True)

    # Rename the approximation ratio columns to remove the 'metrics.QAOA_' prefix and have "approx_ratio_" prefix
    # ONLY apply this to the `approx_ratio_cols` list
    filtered_data.rename(columns=lambda x: x.replace(f"metrics.QAOA_{layer}_layers_approximation_ratio", "approx_ratio"), inplace=True)

    # Rename the success probability column to have "success_probability" prefix
    filtered_data.rename(columns={p_success_col: "success_probability"}, inplace=True)


    # Return the cleaned data
    return filtered_data

def save_data(data: pd.DataFrame, output_filepath: str):
    """ Save the data to a file"""
    data.to_csv(output_filepath, index=False)




@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Load the data
    data_raw = pd.read_csv(input_filepath)
    logger.info(f'Data loaded from {input_filepath}')

    # Set the layer to clean up to
    layer = 20

    # Clean the data for all layers up to the specified layer
    for layer in range(1, layer+1):
        # Clean the data for the specified layer
        data = clean_data_for_layer(data_raw, layer)
        # If the data is None, skip to the next layer
        if data is None:
            continue
        # Write the cleaned data to a file
        data.to_csv(f"data/processed/QAOA_{layer}_layers.csv", index=False)
        # Log the completion of the cleaning process
        logger.info(f'Layer {layer} data cleaned and saved to data/processed/QAOA_{layer}_layers.csv')

    
    

    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
