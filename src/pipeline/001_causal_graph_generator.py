# %%
import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import re
import os
import random
from datetime import datetime
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

import tigramite
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys

from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from pathlib import Path
import argparse
from statsmodels.tsa.stattools import adfuller
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from causalnex.structure.dynotears import from_pandas_dynamic
import tigramite.plotting as tp
import time
import os

from lingam import VARLiNGAM
from itertools import product

import ast
import copy

np.random.seed(123)
random.seed(123)
random_state = 123


# %%
# Define the path to the current file's location
current_path = Path(__file__).resolve().parent if '__file__' in globals() else Path().resolve()
#current_path

# %%
################# CODE FOR RUNNING STANDALONE, WITHOUT THE COMPLETE PIPELINE####################
print('starting at:', datetime.now())

# Default values of not given arguments
experiment_name = 'experiment_1'

# Select the data fractions to evaluate (1 means the complete train set)
fractions = [1, 0.8, 0.6, 0.4, 0.2]

method_discovery = {
            '1_PC': [0.01],
            '2_edit_graph_keep_others': [0.01], # will take the graph of the first CD method given (PC in this case)
            '3_edit_graph_delete_others': [0.01], # will take the graph of the first CD method given (PC in this case)

            '4_PCMCI': [0.01],
            '5_edit_graph_keep_others': [0.01], # will take the graph of the first CD method given (PCMCI in this case)
            '6_edit_graph_delete_others': [0.01], # will take the graph of the first CD method given (PCMCI in this case)

            '13_DYNOTEARS': [0.01],
            '14_VARLINGAM': [0.03],

        }

suffix = "prepare"

tau = 5
# Select the name of the folders, target varaibles in the datasets, and intervention variables
# Select the name of the folders, target varaibles in the datasets, and intervention variables
config = {
        f'simulation{suffix}': {
        'predict': [
            'In_Temp', 
            'ITE_Ener'
        ],
        'interventions_columns': ['Cool_Set'],
        'fractions': fractions,
        'tau': tau,
        'method_discovery': method_discovery,
        'edition_graph': 'yes', 
        'df_extra_info_search': "yes",

    },

    f'minidc{suffix}': {
        'predict': ['water_temp_house'],
        'interventions_columns': ['dc_fan', 'dc_pump', 'house_fan', 'house_pump', 'stress_ctrl'],
        'fractions': fractions,
        'tau': tau,
        'method_discovery': method_discovery,
        'edition_graph': 'yes', 
        'df_extra_info_search': "no",

    },
    # The original data centers datasets is confidential, so the expeiremtns cannot be repeated with the original data. We present the compelte resutls for this datasets.
    # f'dc1{suffix}': {
    #     'predict': ['flow--t1',  'flow--t2', 'flow--t3_temp_customer_flow', 'flow--t4_temp_customer_return_flow'],
    #     'interventions_columns': ['flow2--valve_3_tar', 'flow2--valve_4_tar'],
    #     'fractions': fractions,
    #     'tau': tau,
    #     'method_discovery': method_discovery,
    #     'edition_graph': 'yes', 
    #     'df_extra_info_search': "no",

    # },
    # f'dc2_1{suffix}': {
    #     'predict': ['temp_in_U4', 'temp_out_U4'],
    #     'interventions_columns': ['cool_val_U4'],
    #     'fractions': fractions,
    #     'tau': tau,
    #     'method_discovery': method_discovery,
    #     'edition_graph': 'yes', 
    #     'df_extra_info_search': "yes",


    # },
    # f'dc2_2{suffix}': {
    #     'predict': ['temp_in_U4', 'temp_out_U4'],
    #     'interventions_columns': ['cool_val_U4'],
    #     'fractions': fractions,
    #     'tau': tau,
    #     'method_discovery': method_discovery,
    #     'edition_graph': 'yes', 
    #     'df_extra_info_search': "yes",


    # },

    }
################# CODE FOR RUNNING STANDALONE, WITHOUT THE COMPLETE PIPELINE####################

# %%

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default=experiment_name)
parser.add_argument("--config", type=str, default=str(config))
def list_of_strings(arg):
    return arg.split(',')

# Add an argument for the list of strings
args, _ = parser.parse_known_args()

# Define the root directory containing the subdirectories
experiment_name = Path(args.experiment_name)
config = ast.literal_eval(args.config)

folder_gold = current_path.parent.parent / 'data' / 'gold' / experiment_name

folder_gold.mkdir(parents=True, exist_ok=True)

fixed_tau = True

def find_columns_by_prefix(prefixes, dataset_columns):
    """
    Find columns that either exactly match or start with the given prefixes.

    Parameters:
    - prefixes: list of string prefixes to search for
    - dataset_columns: list of column names in the dataset

    Returns:
    - list: flattened list of all matching columns
    """
    matched_columns = []
    
    for prefix in prefixes:
        matches = [col for col in dataset_columns if col == prefix or col.startswith(prefix)]
        matched_columns.extend(matches)
    
    return matched_columns


# %%

for folder_name_silver, characteristics in config.items():
    parquet_folder_load = current_path.parent.parent / 'data' / 'silver' / folder_name_silver
    datasets = [f for f in parquet_folder_load.iterdir() if f.suffix == '.parquet']
    
    # Load graph_manual.json if it exists
    graph_manual_path = parquet_folder_load / 'graph_manual.json'
    if graph_manual_path.exists():
        with open(graph_manual_path, 'r') as f:
            graph_manual = json.load(f)
    else:
        graph_manual = None

    if fixed_tau:
    # Fixed tau value
        tau = characteristics['tau']

    for dataset, fraction in product(datasets, characteristics['fractions']):

        method_alpha_combinations = characteristics['method_discovery']

        print(f"Processing {dataset} with fraction {fraction}")

        extra_info = dataset
        dataset_name = dataset.name.replace('.parquet', '')

        df_extra_info_search = characteristics['df_extra_info_search']
        edition_graph = graph_manual


        if df_extra_info_search == "yes":
            # get extra info that represent names of columns form the dataset
            df_extra_info = dataset_name.rsplit('_', 1)[-1]
        else:
            df_extra_info = "nn"

        fraction_data_for_experiment =  fraction
        fraction_data_for_generation = 1 

        # %%
        env_path = dataset

        # %%
        # Load data
        df = pd.read_parquet(env_path).reset_index(drop=True)

        interventions_columns = find_columns_by_prefix(config[folder_name_silver]['interventions_columns'], df.columns)
        predict_columns = find_columns_by_prefix(config[folder_name_silver]['predict'], df.columns)


        # %%
        # Create a date column from the time column
        df['date'] = pd.to_datetime(df['time']).dt.date

        # Drop the date column
        df = df.drop(columns=['date'])


        df_raw = df.copy()#.reset_index()
        df_raw#.head()

        # %%
        # First visualization

        df = df_raw
            
        timestep_column = 'time'

        df = df.sort_values(by=timestep_column)
        df_not_normalized = df.copy()

        # # Normalize all columns except 'timestep'
        columns_to_normalize = [column for column in df.columns if column not in [timestep_column]]
        df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].min()) / (df[columns_to_normalize].max() - df[columns_to_normalize].min())
        columns_to_normalize

        # Melt the DataFrame to long format
        df_melted = df.melt(id_vars=[timestep_column], value_vars=df.columns, var_name='variable', value_name='value')

        columns_to_visualize = ['']
        # Plot all variables in one visual

        if False:
            fig = px.line(df_melted, x=timestep_column, y='value', color='variable', title="Normalized Variables vs. Timestep")
            fig.show()

        if False:
            for column in columns_to_visualize :
                if column != timestep_column:
                    fig = px.line(df_not_normalized, x=timestep_column, y=column, title=f"{column} vs. Timestep", line_shape='linear')
                    fig.show()



        # # Do a traditional split per time

        # %%
        df_raw.columns

        # %%
        df = df_raw.copy()


         # Create intervention columns
        df['intervention'] = df[interventions_columns].astype(str).agg('_'.join, axis=1)
        df['intervention_point'] = (df['intervention'] != df['intervention'].shift()).astype(int)
        df['intervention_id'] = pd.factorize(df['intervention'])[0]
        df_with_intervention = df.copy()
        df_with_intervention

        # %%

        # get the rows for training generation

        # Run the fixed solution
        df = df_with_intervention.copy()

        # Ensure the DataFrame is sorted by time
        df = df.sort_values(by='time').reset_index(drop=True)

        # Calculate the split index
        split_index = int(len(df) * 0.7)

        # Split the DataFrame into 70% and 30%
        df_train = df.iloc[:split_index].sort_values('time', ascending=True)  # First 70%

        n_rows = int(len(df_train) * fraction_data_for_experiment)
        df_train = df_train.iloc[:n_rows]
        df_test = df.iloc[split_index:].sort_values('time', ascending=True)  # Remaining 30%

        df_train.drop(columns=['intervention', 'intervention_point'], inplace=True)
        df_test.drop(columns=['intervention', 'intervention_point'], inplace=True)

        # %%
        changepoint_removal = True

        # get the rows for graph generation
        n_rows = int(len(df_train) * fraction_data_for_generation)

        if changepoint_removal is True:

            df_to_graph_generation = df_train.iloc[:n_rows].drop(columns='intervention_id').copy()
            
        else:
            df_to_graph_generation = df_train.iloc[:n_rows].drop(columns='intervention_id').copy()

        #print('df for generation:', df_to_graph_generation.shape)

        # %%
        df_to_graph_generation.columns

        # %%
        na_counts = df_to_graph_generation.isna().sum()

        # Convert the Series to a DataFrame
        na_counts_df = na_counts.reset_index()
        na_counts_df.columns = ['Column', 'NaN_Count']

        # Sort the DataFrame by the number of NaN values if necesary
        na_counts_df.sort_values(by='NaN_Count', ascending=False, inplace=True)

        # Identify columns to drop
        columns_to_drop = na_counts_df[na_counts_df['NaN_Count'] > 2000]['Column']
        len(columns_to_drop)



        # %%
        # Visualize the timeseries with tigramite

        df =df_to_graph_generation.dropna().drop(columns=timestep_column)

        df_tri = pp.DataFrame(df.to_numpy(), var_names=df_train.columns)

        tp.plot_timeseries(df_tri, figsize = (15, 10)); plt.show()

        df =df_test.dropna()

        # %%
        # Function to check stationarity
        def check_stationarity(timeseries):
            result = adfuller(timeseries.dropna(), autolag='AIC')
            p_value = result[1]
            return p_value < 0.01  # If p-value is less than 0.01, the series is stationary

        # Function to make the series stationary
        def make_stationary(df, max_diff=1):

            """
            Transforms the columns of a DataFrame to be stationary using iterative differencing.

            For each column in the input DataFrame, this function:
            - Ignores columns with constant values.
            - Checks if the series is already stationary via `check_stationarity()`; if so, includes as-is.
            - Otherwise, applies differencing up to `max_diff` times, stopping if the series becomes stationary.
            - Includes the resulting stationary series (if any) in the output DataFrame.

            Parameters
            ----------
            df : pd.DataFrame
                Input DataFrame with time series columns to be made stationary.
            max_diff : int, default=1
                Maximum number of differencing steps to attempt for each column.

            Returns
            -------
            stationary_df : pd.DataFrame
                DataFrame containing transformed stationary columns (possibly with fewer rows due to differencing).
                Columns that are constant are omitted.

            Notes
            -----
            - Requires the function `check_stationarity(series)` to be defined elsewhere; this should return True if the input series is stationary.
            - The output may have missing values at the beginning of each column due to differencing, which are retained as NaN or may be trimmed by dropna.

            Examples
            --------
            >>> stationary_df = make_stationary(df, max_diff=2)
            """
            stationary_df = pd.DataFrame()
            
            # Check stationarity of each column
            for column in df.columns:
                series = df[column].dropna()
                
                # Check if the series is constant
                # Constant columsna are dropped
                if series.nunique() == 1:
                    continue
                
                # Check if the series is already stationary
                if check_stationarity(series):
                    stationary_df[column] = series
                    continue
                
                differenced_series = series.copy()
                
                # Apply differencing up to max_diff times
                for d in range(1, max_diff + 1):
                    differenced_series = differenced_series.diff().dropna()
                    if check_stationarity(differenced_series):
                        break
                
                # Store the stationary series in the DataFrame
                stationary_df[column] = differenced_series
            
            return stationary_df


        # %%
        # make stationary and create a tigramite DataFrame
        df = df_to_graph_generation.drop(columns=timestep_column)
        df_stationary = make_stationary(df)
        df_stationary = df_stationary.dropna(how='any')

        # %% [markdown]
        # # Store the dataframes

        # %%
        # get the last experiment and set a new one
        def get_latest_experiment_number(experiments_folder=folder_gold):
            """
            Get the latest experiment number from the experiments folder.
            
            Parameters:
            - experiments_folder: str - Path to the folder containing experiment directories.
            
            Returns:
            - int - Latest experiment number.
            """
            # List all directories in the experiments folder
            directories = [d for d in os.listdir(experiments_folder) if os.path.isdir(os.path.join(experiments_folder, d))]
            
            # Extract numbers from directory names using regular expressions
            experiment_numbers = []
            for directory in directories:
                match = re.match(r'exp_(\d+)', directory)
                if match:
                    experiment_numbers.append(int(match.group(1)))
            
            # Return the latest experiment number or None if no valid experiments are found
            return max(experiment_numbers, default=0)

        experiment_number = get_latest_experiment_number(experiments_folder=folder_gold) + 1
        experiment_number

        # %%
        experiment_name = f'exp_{experiment_number}_{folder_name_silver}__{dataset_name}__f{str(fraction_data_for_experiment).replace(".", "")}_g{str(fraction_data_for_generation).replace(".", "")}'
        folder = folder_gold / experiment_name


        # %%

        # Ensure the folder exists
        if not os.path.exists(folder):
            os.makedirs(folder)

        # File paths for the Parquet files
        train_file_path = os.path.join(folder, 'train.parquet')
        test_file_path = os.path.join(folder, 'test.parquet')
        graph_generator_file_path = os.path.join(folder, 'graph_generator.parquet')


        # Save the DataFrames to Parquet files
        df_train.drop(columns='time').to_parquet(train_file_path, engine='pyarrow')
        df_test.drop(columns='time').to_parquet(test_file_path, engine='pyarrow')
        df_to_graph_generation.to_parquet(graph_generator_file_path, engine='pyarrow')

        # %%
        df_test

        # %%
        if fixed_tau is not True:
            parcorr = ParCorr()
            pcmci = PCMCI(
                dataframe=df_tri, 
                cond_ind_test=parcorr,
                verbosity=1)

            correlations = pcmci.get_lagged_dependencies(tau_max=24, val_only=True)['val_matrix']
            matrix_lags = np.argmax(np.abs(correlations), axis=2)
            tau = int(matrix_lags.mean())
        #print(tau)

        # %% [markdown]
        # # Graph generator

        # %%
        ################### Block of functions for the causal discovery and graph edition ####################
        def convert_to_serializable(obj):
            """Convert non-serializable types to serializable types."""
            if isinstance(obj, dict):
                return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_serializable(i) for i in obj)
            elif isinstance(obj, np.int64):
                return int(obj)  # Convert numpy.int64 to int
            elif isinstance(obj, np.float64):
                return float(obj)  # Convert numpy.float64 to float
            # Add more type checks if necessary
            else:
                return obj

        def save_to_json(variable_data, folder, name = 'graph.json'):
            # Convert the data to a serializable format
            serializable_data = convert_to_serializable(variable_data)
            
            # Create the folder path
            folder_path = os.path.join(folder)
            
            # Ensure the base folder exists
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            # Create the folder with the given number
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            # Define the file path
            file_path = os.path.join(folder_path, name)
            
            # Save the variable data to the JSON file
            with open(file_path, 'w') as f:
                json.dump(serializable_data, f, indent=4)

        def dynotears_to_tigramite_format(G, var_names, max_lag, include_contemporaneous=False):
            """
            Convert DYNOTEARS NetworkX graph to Tigramite format with robust parsing
            
            Parameters:
            G: NetworkX DiGraph from DYNOTEARS
            var_names: list of variable names
            max_lag: maximum lag to consider
            include_contemporaneous: bool, whether to include lag-0 (contemporaneous) relationships
            """
            N = len(var_names)
            graph = np.zeros((N, N, max_lag + 1), dtype=int)
            val_matrix = np.zeros((N, N, max_lag + 1))
            var_to_idx = {var: i for i, var in enumerate(var_names)}
            
            #print("=== DEBUGGING DYNOTEARS GRAPH ===")
            #print(f"Total nodes: {len(G.nodes())}")
            #print(f"Total edges: {len(G.edges())}")
            #print(f"Variable names: {var_names}")
            #print()
            
            # Try different parsing strategies
            edges_processed = 0
            
            for edge in G.edges(data=True):
                source, target = edge[0], edge[1]
                weight = edge[2].get('weight', 0)
                
                if abs(weight) < 1e-10:  # Skip very small weights
                    continue
                    
                # Strategy 1: Direct variable name matching (contemporaneous)
                if source in var_to_idx and target in var_to_idx and include_contemporaneous:
                    source_idx = var_to_idx[source]
                    target_idx = var_to_idx[target]
                    graph[target_idx, source_idx, 0] = 1
                    val_matrix[target_idx, source_idx, 0] = weight
                    edges_processed += 1
                    #print(f"Contemporaneous: {source} -> {target} (weight: {weight:.3f})")
                    continue
                
                # Strategy 2: Parse lag notation 'var lag-X' or 'var_lag-X'
                source_var, source_lag = parse_node_name(source, var_names)
                target_var, target_lag = parse_node_name(target, var_names)
                
                if source_var and target_var:
                    # Calculate actual lag difference
                    lag_diff = source_lag - target_lag
                    
                    # Skip contemporaneous relationships if not wanted
                    if lag_diff == 0 and not include_contemporaneous:
                        continue
                    
                    if 0 <= lag_diff <= max_lag:
                        source_idx = var_to_idx[source_var]
                        target_idx = var_to_idx[target_var]
                        
                        # Tigramite format: graph[i,j,tau] means j->i at lag tau
                        graph[target_idx, source_idx, lag_diff] = 1
                        val_matrix[target_idx, source_idx, lag_diff] = weight
                        edges_processed += 1
            
            # Make lag-zero connections symmetric (required by Tigramite) - only if including contemporaneous
            if include_contemporaneous:
                #print("\n=== MAKING LAG-ZERO SYMMETRIC ===")
                lag_zero_before = np.sum(graph[:, :, 0])
                
                # For lag 0, make both graph and val_matrix symmetric
                graph_lag0 = graph[:, :, 0]
                val_lag0 = val_matrix[:, :, 0]
                
                # Create symmetric versions
                graph_symmetric = np.logical_or(graph_lag0, graph_lag0.T).astype(int)
                val_symmetric = np.maximum(np.abs(val_lag0), np.abs(val_lag0.T))
                
                # Keep the sign from the stronger connection
                stronger_connection = np.abs(val_lag0) >= np.abs(val_lag0.T)
                val_symmetric = np.where(stronger_connection, val_lag0, val_lag0.T)
                val_symmetric = np.maximum(val_symmetric, val_symmetric.T)
                
                graph[:, :, 0] = graph_symmetric
                val_matrix[:, :, 0] = val_symmetric
                
                lag_zero_after = np.sum(graph[:, :, 0])
            else:
                #print("\n=== SKIPPING CONTEMPORANEOUS RELATIONSHIPS ===")
                # Ensure lag-0 is empty
                graph[:, :, 0] = 0
                val_matrix[:, :, 0] = 0
                #print("Lag-0 connections set to zero")
            
            
            return {'graph': graph, 'val_matrix': val_matrix}

        def parse_node_name(node_name, var_names):
            """
            Parse node name to extract variable name and lag
            Returns (variable_name, lag) or (None, None) if parsing fails
            """
            node_str = str(node_name)
            
            # Try different parsing patterns
            patterns = [
                ' lag-',  # 'variable lag-1'
                '_lag-',  # 'variable_lag-1'
                ' lag',   # 'variable lag1'
                '_lag',   # 'variable_lag1'
                '_t-',    # 'variable_t-1'
                '_t+',    # 'variable_t+1'
            ]
            
            for pattern in patterns:
                if pattern in node_str:
                    parts = node_str.split(pattern)
                    if len(parts) == 2:
                        var_name = parts[0].strip()
                        try:
                            lag_part = parts[1].strip()
                            # Handle negative signs
                            if lag_part.startswith('-'):
                                lag = -int(lag_part[1:])
                            elif lag_part.startswith('+'):
                                lag = int(lag_part[1:])
                            else:
                                lag = int(lag_part)
                            
                            # Check if variable name exists
                            if var_name in var_names:
                                return var_name, lag
                        except ValueError:
                            continue
            
            # If no pattern matches, check if it's a direct variable name
            if node_str in var_names:
                return node_str, 0
            
            # Try removing common suffixes/prefixes
            for var in var_names:
                if var in node_str:
                    # Extract lag from remaining part
                    remaining = node_str.replace(var, '').strip('_- ')
                    if remaining.isdigit():
                        return var, int(remaining)
                    elif remaining.startswith('t') and remaining[1:].isdigit():
                        return var, int(remaining[1:])
            
            return None, None

        def run_dynotears_analysis(df, var_names, tau_min, tau_max, w_threshold=0):
            """
            Run DYNOTEARS analysis and return results in PCMCI-compatible format
            
            Parameters:
            df: pandas DataFrame with stationary data
            var_names: list of variable names
            tau_min: minimum lag (horizon)
            tau_max: maximum lag
            w_threshold: weight threshold for DYNOTEARS
            
            Returns:
            dict with 'graph', 'val_matrix', and 'p_matrix' keys
            """
            # Calculate lag parameter for DYNOTEARS
            lag = tau_max
            
            # Run DYNOTEARS
            sm_dyn = from_pandas_dynamic(
                df, 
                p=lag,
                w_threshold=w_threshold # use the standard thresholding
            )
            
            # Convert to Tigramite format
            dyno_results = dynotears_to_tigramite_format(
                sm_dyn, var_names, tau_max, include_contemporaneous=False
            )
            

            p_matrix = np.ones_like(dyno_results['val_matrix'])
            p_matrix[dyno_results['graph'] == 1] = 0.01
            
            return {
                'graph': dyno_results['graph'],
                'val_matrix': dyno_results['val_matrix'],
                'p_matrix': p_matrix
            }

        def create_parents_dict_from_graph(graph, val_matrix):
            """
            Create parents dictionary from graph matrix (mimicking PCMCI format)
            
            Parameters:
            graph: numpy array of shape (N, N, tau_max+1)
            val_matrix: numpy array of shape (N, N, tau_max+1)
            
            Returns:
            dict: {variable_index: [(parent_variable_index, lag), ...]}
            """
            N, _, tau_max_plus_one = graph.shape
            parents_dict = {}
            
            for i in range(N):
                parents_list = []
                for j in range(N):
                    for tau in range(tau_max_plus_one):
                        if graph[i, j, tau] == 1:
                            parents_list.append((j, -tau))
                parents_dict[i] = parents_list
            
            return parents_dict

        def tigramite_parents_dict(graph, val_matrix, min_lag=1):
            """Returns dict: target index -> list of (parent index, lag) given Graph and matrix"""
            n_variables = graph.shape[0]
            n_lags = graph.shape[2]
            parent_dict = {}
            for i in range(n_variables):  # target (current node)
                links = []
                # Start from min_lag (usually 1, skip lag 0/contemporaneous)
                for j in range(n_variables):  # source
                    for lag in range(min_lag, n_lags):  # lag 1..max
                        if graph[i, j, lag]:
                            links.append((j, lag))
                parent_dict[i] = links
            return parent_dict


        def lingam_to_tigramite_graph(df_stationary, lags=4, threshold=0):
            """Fits VARLiNGAM to dataframe and returns a (graph, val_matrix, var_names)
            tuple in Tigramite format."""

            # Remove non-numeric columns
            var_names = [c for c in df_stationary.columns if np.issubdtype(df_stationary[c].dtype, np.number)]
            df = df_stationary[var_names]
            
            model = VARLiNGAM(lags=lags)
            model.fit(df)
            n_vars = len(var_names)
            
            # Create arrays with lag 0 included
            graph = np.zeros((n_vars, n_vars, lags + 1), dtype=int)
            val_matrix = np.zeros((n_vars, n_vars, lags + 1), dtype=float)
            
            # Fill lag 0 (contemporaneous) with zeros
            
            # Check how many adjacency matrices we actually have
            n_adjacency_matrices = len(model.adjacency_matrices_)
            #print(f"Number of adjacency matrices from VARLiNGAM: {n_adjacency_matrices}")
            #print(f"Expected lags: {lags}")
            
            # Fill lags 1...min(lags, n_adjacency_matrices)
            max_available_lag = min(lags, n_adjacency_matrices)

            for lag in range(1, max_available_lag + 1):
                adj_matrix = model.adjacency_matrices_[lag - 1]
                for i in range(n_vars):
                    for j in range(n_vars):
                        coef = adj_matrix[i, j]
                        if abs(coef) >= threshold:  # Only include strong enough links
                            graph[i, j, lag] = 1
                            val_matrix[i, j, lag] = coef
                        else:
                            graph[i, j, lag] = 0
                            val_matrix[i, j, lag] = 0.0

            return {'graph': graph, 'val_matrix': val_matrix}, var_names
        
        def prepare_for_var_lingam(df):
            df = df.copy()
            # 1. Drop constant columns
            const_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
            if const_cols:
                #print("Dropping constant columns:", const_cols)
                df = df.drop(columns=const_cols)
                info = f"Dropped constant columns for varligam: {const_cols}"
            # 2. Drop highly correlated columns (threshold at 0.9999, you can adjust)
            corr_matrix = df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.9999)]
            if to_drop:
                #print("Dropping highly collinear columns:", to_drop)
                df = df.drop(columns=to_drop)
                info = f"Dropped highly collinear columns for varligam: {to_drop}"
            # 3. Drop/infill NA or Inf
            else:
                info = "No constant or collinear columns found for varligam"
            df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
            return df, info
        
        def save_edited_graph(folder_lag, edition_graph):
            # Define the path to the JSON file
            graph_links_path = folder_lag  + "/graph.json"

            # Load the graph links from the JSON file
            with open(graph_links_path, "r") as file:
                graph_with_names = json.load(file)
        
        def edit_results(results, analysis_dict, var_names, deleting_other_links=False, alpha_threshold=0.05):
            """
            Update results by modifying val_matrix, p_matrix, and graph
            depending on deleting_other_links flag.

            Parameters
            ----------
            results : dict
                Dictionary with 'val_matrix', 'p_matrix', 'graph'
            analysis_dict : dict
                Dictionary containing target -> causes mapping
            var_names : list
                Variable names for plotting
            deleting_other_links : bool
                If True, use the second logic (replace others with blank, keep arrows).
                If False, use the first logic (directly assign '-->').
            alpha_threshold : float
                P-value threshold for selecting lags. Select all lags below this threshold.
                If none, select the lag with the lowest p-value.
            """

            # Always work on a deepcopy so we don't overwrite the original results
            edited_results = copy.deepcopy(results)

            for target, causes in analysis_dict.items():
                to_number = target
                for cause in causes:
                    from_number = cause
                    ##print(f"Target: {target}, Cause: {cause}")

                    # Check if there's already a link from causal discovery for this pair
                    row = edited_results['graph'][from_number][to_number]
                    has_existing = np.any(row != '')

                    if not deleting_other_links and has_existing:
                        # For keep_others, skip adding if there's already a link
                        continue

                    if deleting_other_links and has_existing:
                        # For delete_others, preserve existing links
                        row[row != ''] = 'arr'
                    else:
                        # For both cases, if no existing link, select lags and add
                        # Get p-values and absolute t-statistics for this pair
                        p_arr = edited_results['p_matrix'][from_number][to_number]
                        abs_arr = np.abs(edited_results['val_matrix'][from_number][to_number])

                        # Find significant lags (p < alpha_threshold)
                        significant_cols = np.where(p_arr < alpha_threshold)[0]

                        if len(significant_cols) > 0:
                            # Select all significant lags
                            selected_cols = significant_cols
                        else:
                            # No significant lags: select the one with the lowest p-value
                            selected_cols = [np.argmin(p_arr)]

                        # Update matrices for selected lags
                        for col in selected_cols:
                            if deleting_other_links:
                                # For deleting_other_links, use 'arr' (will be cleaned up later)
                                edited_results['val_matrix'][from_number][to_number][col] = 1
                                edited_results['p_matrix'][from_number][to_number][col] = 0
                                edited_results['graph'][from_number][to_number][col] = 'arr'
                            else:
                                # For keeping others, directly assign '-->'
                                edited_results['val_matrix'][from_number][to_number][col] = 1
                                edited_results['p_matrix'][from_number][to_number][col] = 0
                                edited_results['graph'][from_number][to_number][col] = '-->'

            if deleting_other_links:
                # Cleanup step: remove non-'arr' links, convert 'arr' to '-->'
                edited_results['graph'][edited_results['graph'] != 'arr'] = ''
                edited_results['graph'][edited_results['graph'] == 'arr'] = '-->'

            return edited_results

        def extract_info_from_dict(d):
            # Pattern to extract (number) and variable name
            pattern = r"\((\d+)\)\s*(.+)\s*"
            var_name_to_num = {}
            max_num = -1
            for key in d:
                match = re.match(pattern, key)
                if match:
                    num = int(match.group(1))
                    name = match.group(2).strip()  # Remove leading/trailing spaces
                    var_name_to_num[name] = num
                    if num > max_num:
                        max_num = num
            max_variables = max_num
            return var_name_to_num

        def prepare_editing_dicts(edition_graph, replaced_dict, mapping):

            # # Build dict with editing
            # result_dict_with_editing = {}
            # for key, value in edition_graph.items():
            #     key_name = key.strip()
            #     key_num = mapping[key_name]
            #     result_dict_with_editing[key_num] = [mapping[v[0]] for v in value]

            # # Build dict from causal discovery
            # causal_discovery = {}
            # for key, value in replaced_dict.items():
            #     key_stripped = key.split(')', 1)[-1].strip()
            #     key_num = mapping[key_stripped]
            #     causal_discovery[key_num] = [mapping[v[0]] for v in value]

            # # Find differences (editing but not in causal discovery)
            # difference_editing_not_in_causal = {}
            # for key in result_dict_with_editing:
            #     editing_set = set(result_dict_with_editing[key])
            #     causal_set = set(causal_discovery.get(key, []))  # handle missing keys
            #     diff = editing_set - causal_set
            #     if diff:
            #         difference_editing_not_in_causal[key] = list(diff)

            # Build dict with editing
            result_dict_with_editing = {}
            for key, value in edition_graph.items():
                key_name = key.strip()
                key_num = mapping.get(key_name)   # <-- safely get mapping
                if key_num is None:
                    continue  # skip if not in mapping
                mapped_values = [mapping[v[0]] for v in value if v[0] in mapping]
                if mapped_values:  # only add if non-empty
                    result_dict_with_editing[key_num] = mapped_values

            # Build dict from causal discovery
            causal_discovery = {}
            for key, value in replaced_dict.items():
                key_stripped = key.split(')', 1)[-1].strip()
                key_num = mapping.get(key_stripped)  # <-- safely get mapping
                if key_num is None:
                    continue  # skip if not in mapping
                mapped_values = [mapping[v[0]] for v in value if v[0] in mapping]
                if mapped_values:
                    causal_discovery[key_num] = mapped_values

            # Find differences (editing but not in causal discovery)
            difference_editing_not_in_causal = {}
            for key, editing_vals in result_dict_with_editing.items():
                editing_set = set(editing_vals)
                causal_set = set(causal_discovery.get(key, []))  # handle missing keys
                diff = editing_set - causal_set
                if diff:
                    difference_editing_not_in_causal[key] = list(diff)

            return result_dict_with_editing, difference_editing_not_in_causal
        

        def save_edited_graph(folder_lag, edition_graph):
            # Define the path to the JSON file
            graph_links_path = folder_lag  + "/graph.json"

            # Load the graph links from the JSON file
            with open(graph_links_path, "r") as file:
                graph_with_names = json.load(file)
  
            # Convert dictionary to string
            edition_graph = str(edition_graph)

            # Replace all occurrences of "df_extrainfo" with the actual variable value
            if df_extra_info != "nn":
                edition_graph = edition_graph.replace("df_extrainfo", df_extra_info)

            # Convert string back to dictionary
            edition_graph = ast.literal_eval(edition_graph)


            cleaned_dict = {}
            for k in graph_with_names:
                cleaned_dict[k] = []

            new_graph = cleaned_dict.copy()

            # Match keys from edition_graph with keys in cleaned_dict and add values
            for edition_key, edition_value in edition_graph.items():
                # Search for cleaned_dict keys that contain the edition_key text
                for cleaned_key in cleaned_dict.keys():
                    # Check if the edition_key text is contained in the cleaned_key
                    # Remove the numbering prefix from edition_key for comparison
                    edition_key_clean = edition_key.split(') ', 1)[-1].strip() if ') ' in edition_key else edition_key.strip()
                    if edition_key_clean in cleaned_key:
                        # Add the edition_graph values to the corresponding cleaned_dict key
                        new_graph[cleaned_key].extend(edition_value)
                        #print(f"Matched '{edition_key_clean}' with '{cleaned_key}' and added values: {edition_value}")
                        break  # Found a match, move to next edition_graph key

            # Define the path to the JSON file
            folder_to_save = folder_lag.split(f'lag_{horizon}_')[0] + f'lag_{horizon}_edited_all'

            os.makedirs(folder_to_save, exist_ok=True)


            with open(folder_to_save + "/graph.json", "w") as file:
                json.dump(new_graph, file, indent=4)

            json_info = {
                'target': predict,
                'tau': tau,
                'setpoint_change': setpoint_change,
                'other_info': 'edited',
                'lagged_already': lagged_already,
                "changepoint_removal": changepoint_removal,
                #'models_to_eval': models_to_eval
            }

            save_to_json(json_info, folder=folder_to_save, name='info.json')

################### Block of functions for the causal discovery and graph edition ####################
        # %%
        df_stationary.columns

        # %%
        predict = predict_columns

        # %%
        df_stationary.shape

        # %%
        df_stationary   

        # %%
        var_names = df.columns
        var_names

        # %% loop to generate the graphs

        setpoint_change = 'intervention_id'
        changepoint_removal = True
        lagged_already = False
        horizon = 1


        def main_analysis_loop(method_alpha_combinations, df_stationary, tau, folder, predict, horizon, setpoint_change,  changepoint_removal, lagged_already, extra_info):
            # Dictionary to store the results
            graph_pc_for_assumptions = {}
            store_results_for_edition = None
            for method, alphas in sorted(method_alpha_combinations.items(),key=lambda x: int(x[0].split("_")[0])):
                    method = method.split("_", 1)[1]
                    print("Causal discovery with:", method, alphas)
                    for alpha in alphas:

                        #print(f"dataset: {extra_info}")
                        extra_info = str(extra_info)
                        # Get the head of the dataframe
                        df = df_stationary#.iloc[0:2000]

                        # This finds constant columns (of any type, including strings)
                        constant_columns = [col for col in df.columns if df[col].nunique(dropna=False) == 1]

                        if constant_columns:
                            #print(f"Dropping constant columns: {constant_columns}")
                            df = df.drop(columns=constant_columns)
                            #print(f"DataFrame shape after dropping constant columns: {df.shape}")
                        #else:
                            #print("No constant columns found")
                        
                        # Start timing
                        for horizon in range(1, horizon + 1):
                            start_time = time.time()
                            #print(f'Generating graph for horizon: {horizon}, method: {method}, alpha: {alpha}')
                            tau_min = horizon
                            tau_max = tau_min + tau - 1
                            var_names = df.columns
                            if method == 'DYNOTEARS':
                                # Run DYNOTEARS analysis
                                results = run_dynotears_analysis(
                                    df, var_names, tau_min, tau_max, w_threshold=alpha
                                )
                            elif method == 'VARLINGAM':
                                # Run VARLiNGAM analysis
                                df, info = prepare_for_var_lingam(df)
                                extra_info += f"\n{info}"
                                results, var_names = lingam_to_tigramite_graph(df, lags=tau_max, threshold=alpha)

                            else:
                                # Original PCMCI/PC code
                                df_tri = pp.DataFrame(df.to_numpy(), var_names=df.columns)
                                parcorr = ParCorr()
                                pcmci = PCMCI(
                                    dataframe=df_tri,
                                    cond_ind_test=parcorr,
                                    verbosity=0
                                )
                                if method == 'PC':
                                    results = pcmci.run_pcalg(tau_min=tau_min, tau_max=tau_max, pc_alpha=alpha)
                                    
                                    p_matrix = results['p_matrix']
                                    # pcmci.print_significant_links(
                                    #     p_matrix=p_matrix,
                                    #     val_matrix=results['val_matrix'],
                                    #     alpha_level=alpha
                                    # )
                                    graph = pcmci.get_graph_from_pmatrix(p_matrix=p_matrix, alpha_level=alpha, tau_min=tau_min, tau_max=tau_max, link_assumptions=None)
                                    results['graph'] = graph

                                    store_results_for_edition = results
                                    method_for_edition = f"_{method}_{alpha}"

                                elif method == 'PCMCI':
                                    results = pcmci.run_pcmci(tau_min=tau_min, tau_max=tau_max, pc_alpha=alpha)
                                    
                                    p_matrix = results['p_matrix']
                                    # pcmci.print_significant_links(
                                    #     p_matrix=p_matrix,
                                    #     val_matrix=results['val_matrix'],
                                    #     alpha_level=alpha
                                    # )
                                    graph = pcmci.get_graph_from_pmatrix(p_matrix=p_matrix, alpha_level=alpha, tau_min=tau_min, tau_max=tau_max, link_assumptions=None)
                                    results['graph'] = graph

                                    store_results_for_edition = results
                                    method_for_edition = f"_{method}_{alpha}"

                                elif method == 'edit_graph_keep_others':

                                    # Edit the graph PC or PCMCI to add the missing links that were given in the manual json. The link is going to be the link between the variables with the 
                                    # Highest t-statisti

                                    mapping = extract_info_from_dict(store_replace_dict_for_edition)
                                    result_dict_with_editing, difference_editing_not_in_causal = prepare_editing_dicts(edition_graph, store_replace_dict_for_edition, mapping)
                                    results = edit_results(store_results_for_edition, difference_editing_not_in_causal, var_names, deleting_other_links=False)
                                    method = f"{method}__{method_for_edition}"

                                    
                                elif method == 'edit_graph_delete_others':

                                    # Same as before but if a connection does not appear in the manual graph, the it is going to be removed

                                    #mapping = extract_info_from_dict(store_replace_dict_for_edition)
                                    #result_dict_with_editing, difference_editing_not_in_causal = prepare_editing_dicts(edition_graph, store_replace_dict_for_edition, mapping)
                                    results = edit_results(store_results_for_edition, result_dict_with_editing, var_names, deleting_other_links=True)
                                    method = f"{method}__{method_for_edition}"

                                else:
                                    raise ValueError(f"Invalid method {method}. Choose 'PC', 'PCMCI', 'DYNOTEARS', 'edit_graph_keep_others', 'edit_graph_delete_others'")
                                # For PCMCI/PC, get graph from p_matrix

                            # Plot the graph
                            tp.plot_graph(
                                val_matrix=results['val_matrix'],
                                graph=results['graph'],
                                var_names=var_names,
                                show_autodependency_lags=True,
                                figsize=(15, 12)
                            )
                            # Create parents dictionary
                            if method == 'DYNOTEARS':
                                data = create_parents_dict_from_graph(results['graph'], results['val_matrix'])
                            elif method == 'VARLINGAM':
                                data = create_parents_dict_from_graph(results['graph'], results['val_matrix'])
                            else:
                                data = pcmci.return_parents_dict(results['graph'], results['val_matrix'])
                                
                
                            # Convert indices to variable names
                            variable_names = var_names
                            index_to_var_name = {i: variable_names[i] for i in range(len(variable_names))}
                            replaced_dict = {}
                            for key, tuples_list in data.items():
                                new_key = f"({key}) {index_to_var_name.get(key, key)} "
                                new_tuples_list = [(index_to_var_name.get(t[0], t[0]), t[1]) for t in tuples_list]
                                replaced_dict[new_key] = new_tuples_list
                            if method == 'PC' or method == 'PCMCI':
                                store_replace_dict_for_edition = copy.deepcopy(replaced_dict)
                                
                            #print(replaced_dict)
                            # Save results
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            folder_lag = f'{folder}/lag_' + str(horizon) + f'_{method}_alpha_{alpha}' + '_all'
                            save_to_json(replaced_dict, folder=folder_lag, name='graph.json')
                            save_to_json(data, folder=folder_lag, name='graph_links.json')
                            json_info = {
                                'target': predict,
                                'tau': tau,
                                'setpoint_change': setpoint_change,
                                'other_info': f'{method}_alpha_{alpha}',
                                'lagged_already': lagged_already,
                                'changepoint_removal': changepoint_removal,
                                'time_graph_generation': elapsed_time,
                                #'models_to_eval': models_to_eval
                            }
                            save_to_json(json_info, folder=folder_lag, name='info.json')
                            file_name = os.path.join(folder_lag, 'array_graph.npy')
                            np.save(file_name, results['graph'])
                            file_path = os.path.join(folder_lag, 'graph_plot.png')
                            plt.savefig(file_path)
                            plt.close()

                            if edition_graph is not None:
                                save_edited_graph(folder_lag, edition_graph)

                            info_file_path = os.path.join(folder, 'info.txt')
                            with open(info_file_path, 'w') as file:
                                file.write(f"tau = {tau}\n")
                                file.write(f"shape_graph_generator = {df_to_graph_generation.shape}\n")
                                file.write(f"shape_train = {df_train.shape}\n")
                                file.write(f"shape_test = {df_test.shape}\n")
                                file.write(f"extra_info = {extra_info}\n")
                                file.write(f"errors:\n")

                                if constant_columns:
                                    file.write(f"  - constant columns were dropped: {constant_columns}\n")


                # Print the timing results
            return folder_lag, results, replaced_dict, store_results_for_edition

        # Call the main function
        folder_lag, results, replaced_dict, store_results_for_edition = main_analysis_loop(method_alpha_combinations, df_stationary, tau, folder, predict, horizon, setpoint_change, changepoint_removal, lagged_already, extra_info)
