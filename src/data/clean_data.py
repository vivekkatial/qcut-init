import click
import logging
import re
import pandas as pd
import numpy as np
import warnings

from pathlib import Path
from dotenv import find_dotenv, load_dotenv


warnings.filterwarnings("ignore")


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
    
    for i in range(5):
        beta_col_name = f'metrics.QAOA_{algorithm}_optimal_beta_{i}'
        gamma_col_name = f'metrics.QAOA_{algorithm}_optimal_gamma_{i}'
        
        row[f'beta_{i}'] = row[beta_col_name] if beta_col_name in columns else np.nan
        row[f'gamma_{i}'] = row[gamma_col_name] if gamma_col_name in columns else np.nan
    
    return row

def clean_data(data: pd.DataFrame)->pd.DataFrame:
    """ Clean the data"""

    # Filter for the rows where status is `FINISHED`
    data = data[data['status'] == 'FINISHED']

    # Select feature columns based on conditions
    feature_cols = [
        col for col in data.columns 
        if col.startswith('params.') and 
        'most_likely_solution' not in col and 
        'initial_point' not in col
    ]

    # Include 'run_id' separately
    feature_cols.append('run_id')

    # List out cols with approximation_ratio in the name
    approx_ratio_cols = [col for col in data.columns if 'approximation_ratio' in col]
    data['max_value_column'] = data.apply(find_max_column, axis=1, approx_ratio_cols=approx_ratio_cols)

    # Create 10 new columns for the beta / gamma values for the max value column (5 for each)
    for i in range(5):
        data.loc[:, f'beta_{i}'] = np.nan
        data.loc[:, f'gamma_{i}'] = np.nan


    data = data.apply(fill_beta_gamma, axis=1, columns=data.columns)

    # Rename feature columns: add 'feat_' prefix and remove 'params.' prefix
    feature_cols_renamed = {col: 'feat_' + col.split('.')[-1] for col in feature_cols}
    data = data.rename(columns=feature_cols_renamed)

    # Create a list for the beta_i and gamma_i columns
    beta_gamma_cols = [f'beta_{i}' for i in range(5)] + [f'gamma_{i}' for i in range(5)]

    # Select feature columns and beta/gamma columns
    feature_cols = list(feature_cols_renamed.values())
    data = data[feature_cols + beta_gamma_cols]

    # Apply the extract_real function to the data
    data = apply_extract_real_to_df(data)

    return data

def save_data(data: pd.DataFrame, output_filepath: str):
    """ Save the data to a file"""
    data.to_csv(output_filepath, index=False)




@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Load the data
    data = pd.read_csv(input_filepath)

    # Clean the data
    data = clean_data(data)

    # Save the data
    save_data(data, output_filepath)
    logger.info(f'Data saved to {output_filepath}')

    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
