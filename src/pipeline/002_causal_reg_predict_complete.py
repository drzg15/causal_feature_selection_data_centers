

# %%
import json
import os
import sys
import datetime
import joblib
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
import xgboost as xgb
from lightgbm import LGBMRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import re
import time
import argparse
from pathlib import Path
from sklearn.neural_network import MLPRegressor
import os
import ast
import warnings
from utils import calculate_metrics

sys.path.append('../')
warnings.filterwarnings('ignore')


np.random.seed(123)
random.seed(123)
random_state = 123

# %%

################# FEATURE SELECTION TIME EVALUATION FLAG ####################
# If True, only measure feature selection execution times without training models
only_fs_time_evaluation = False
################# FEATURE SELECTION TIME EVALUATION FLAG ####################

# %%

################# CODE FOR RUNNING STANDALONE, WITHOUT THE COMPLETE PIPELINE####################

current_path = Path(__file__).resolve().parent if '__file__' in globals() else Path().resolve()
print(current_path)

experiment_name =  1

add_lag_all_to_feature_selection = True


parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str, default=None)
parser.add_argument("--models_to_eval", type=str, default=None)

def list_of_strings(arg):
    return arg.split(',')

args, _ = parser.parse_known_args()

data_directory = Path(args.datadir)
models_to_eval = ast.literal_eval(args.models_to_eval)

print("models to eval:", models_to_eval)

################# CODE FOR RUNNING STANDALONE, WITHOUT THE COMPLETE PIPELINE####################

# %%
def load_from_json(file_path):
    
    # Load the JSON data into a dictionary
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data

# Initialize a dictionary to hold all graphs, ke  yed by lag number
dict_graphs_by_lag = {}

# Loop through all folders with the causal graph
for folder_path in [f for f in data_directory.glob("*") if f.is_dir() and f.name.startswith("lag")]:

    foldername = str(folder_path).rsplit("/", 1)[-1]

    # Get all data from the fol
    with open(folder_path / 'info.json', 'r') as f:
        data = json.load(f)

    columns_target = data.get("target", None)
    column_change_point = data.get("setpoint_change", None) 
    changepoint_removal = data.get("changepoint_removal", None) 
    input_size = data.get("tau", None)
    
    # Initialize an empty dictionary for this lag if not already present
    if foldername not in dict_graphs_by_lag:
        dict_graphs_by_lag[foldername] = {}
    
    filename = folder_path / "graph.json"
    # Loop through all files in the folder
    try:
        # Load the JSON data from the file
        graph_dict = load_from_json(filename)
        
        # Add the loaded data to the dictionary of graphs for this lag
        dict_graphs_by_lag[foldername] = graph_dict
        
        # Print the contents of the JSON file
        print(f"Contents of {filename}:")
        for key, value in graph_dict.items():
            print(f"{key}: {value}")
        print("\n")
    except FileNotFoundError as e:
        print(e)


# Extracting the dataframes from the parquet files
df_train = pd.read_parquet(data_directory / 'train.parquet')
df_test = pd.read_parquet(data_directory / 'test.parquet')


# %%
def create_lagged_features(df, column_lags=None, tau_min=None, tau_max=None, target_column=None, select_columns=None):
    """
    Create lagged features for a pandas DataFrame using flexible lagging options.

    This function allows for three main lagging strategies:
      1. Lag specific columns by specific amounts (via `column_lags`).
      2. Lag selected columns by a specified range (`tau_min` to `tau_max`).
      3. Lag all columns (except the target) across a specified range if no columns or pairs provided.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    column_lags : list of tuple, optional
        List of (column_name, lag) tuples specifying which columns to lag and by how much.
        If provided, creates lagged columns for each (column, lag) pair. Negative lags only.
    tau_min : int, optional
        Minimum lag value (inclusive). Used if `column_lags` or `select_columns` is None.
    tau_max : int, optional
        Maximum lag value (inclusive). Used if `column_lags` or `select_columns` is None.
    target_column : str, optional
        The column to keep unlagged and aligned with lagged features.
    select_columns : list of str, optional
        List of column names to apply a range of lags (`tau_min` to `tau_max`).

    Returns
    -------
    pd.DataFrame
        A DataFrame with lagged feature columns added and the target column retained. 
        Lagged column names are formatted as 'column_name(lag)' 
        where lag is negative to indicate a lag (e.g., 'x(-1)' for x lagged by 1).

    Raises
    ------
    ValueError
        If required arguments are missing for a selected mode.

    Notes
    -----
    - Lags are represented as negative numbers in output column names.
    - At least one lagging specification method (`column_lags`, `select_columns`, or `tau_min`/`tau_max`) must be used.
    """

    lagged_dfs = []

    if column_lags:
        # Process specific column-lag pairs
        for column_name, lag in column_lags:
            if lag < 0:
                df_lagged = df[[column_name]].shift(-lag)
                df_lagged.columns = [f'{column_name}({lag})']
                lagged_dfs.append(df_lagged)
        
        # Combine all lagged features and then add the target column
        if lagged_dfs:
            df_lagged_combined = pd.concat(lagged_dfs, axis=1)
            df_result = pd.concat([df[[target_column]], df_lagged_combined], axis=1)

    elif select_columns:
        # Lag selected columns with the range of lags from tau_min to tau_max
        if tau_min is None or tau_max is None or target_column is None:
            print('tau_min: ', tau_min)
            print('tau_max: ', tau_max)
            print('target_column: ', target_column)
            raise ValueError("tau_min, tau_max, and target_column must be specified if select_columns is used.")

        
        lagged_features = []

        for column_name in select_columns:
            for tau in range(tau_min, tau_max + 1):
                df_lagged = df[[column_name]].shift(tau)
                df_lagged.columns = [f'{column_name}({-tau})']
                lagged_features.append(df_lagged)

        # Combine all lagged features and then add the target column
        if lagged_features:
            df_lagged_combined = pd.concat(lagged_features, axis=1)
            df_result = pd.concat([df[[target_column]], df_lagged_combined], axis=1)
    
    else:
        if tau_min is None or tau_max is None or target_column is None:
            print('tau_min: ', tau_min)
            print('tau_max: ', tau_max)
            print('target_column: ', target_column)
            raise ValueError("tau_min, tau_max, and target_column must be specified if column_lags is None and select_columns is not used.")
        
        # Initialize a DataFrame to store lagged features
        lagged_features = []

        for tau in range(tau_min, tau_max + 1):
            #df_lagged = df.drop(columns=[target_column]).shift(tau)
            df_lagged = df.shift(tau)
            df_lagged.columns = [f'{col}({-tau})' for col in df_lagged.columns]
            lagged_features.append(df_lagged)

        # Combine all lagged features and then add the target column
        if lagged_features:
            df_lagged_combined = pd.concat(lagged_features, axis=1)
            df_result = pd.concat([df[[target_column]], df_lagged_combined], axis=1)
    
    
    return df_result

# %%
################### Block of functions for traditional feature selection and prediction ####################
def find_values_with_substring(d, substring):
    """
    Returns the value associated with the first key in the dictionary `d` that contains the given `substring`.

    Args:
        d (dict): The dictionary to search through.
        substring (str): The substring to search for within the keys of the dictionary.

    Returns:
        The value corresponding to the first key that contains `substring`, or raises an IndexError if no such key exists.
    """
    # Initialize an empty list to store the results
    results = []

    
    # Iterate over the dictionary's items
    for key, value in d.items():
        # Check if the substring is in the key
        if substring in key:
            # Append the value to the results list
            results.append(value)
    
    # Return the list of values
    return results[0]


def select_features(X_train, y_train, method, measure_time=False):
    """
    Selects important features from the training data using the specified feature selection method.

    Parameters
    ----------
    X_train : pandas.DataFrame
        The input features for training.
    y_train : array-like or pandas.Series
        The target variable for regression.
    method : str
        The feature selection strategy to use.
        Supported options are:
            - 'RFE': Recursive Feature Elimination with Linear Regression.
            - 'Lasso': Lasso regression (L1 regularization).
            - 'PCA': Principal Component Analysis (selects features with high loadings in components explaining up to 85% variance).
            - 'Tree-based': Random Forest regressor feature importances.
    measure_time : bool, optional
        If True, measure and return the execution time along with features.

    Returns
    -------
    selected_features : list
        List of selected feature names.
    fs_duration : float (only if measure_time=True)
        Execution time in seconds for the feature selection method.

    Raises
    ------
    ValueError
        If the specified method is not one of 'RFE', 'Lasso', 'PCA', 'Tree-based'.

    Notes
    -----
    - 'RFE' uses LinearRegression to recursively eliminate features.
    - 'Lasso' selects features with non-zero coefficients when fitting with alpha=0.01.
    - 'PCA' chooses features contributing most strongly to the first components cumulatively explaining at least 85% variance.
    - 'Tree-based' selects features with importance score greater than 0.01 in a RandomForestRegressor.
    - The function prints the selected features before returning.

    """

    # Start timing if requested
    fs_start_time = time.time() if measure_time else None
    
    selected_features = []

    # print("HEREEEE!!: ", method)
    lag_method = method.split("__")[-1].split("/")[0]
    # print("HEREEEE!!: ", lag_method)
    tau = int(method.split("/")[-1])
    method = method.split("__")[-0]

    if method == "RFE":
        # Use a regression model, e.g., LinearRegression, for RFE
        estimator = LinearRegression()
        # Initialize RFE with cross-validation to find the optimal number of features
        selector = RFE(estimator, n_features_to_select=None, step=1)
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.get_support(indices=True)]
        
    elif method == "Lasso":
        # Lasso performs L1 regularization and can be used for feature selection in regression
        lasso = Lasso(alpha=0.01, random_state=random_state)
        lasso.fit(X_train, y_train)
        # Select features with non-zero coefficients
        selected_features = X_train.columns[np.abs(lasso.coef_) > 1e-5]
        
    elif method == "PCA":
        # Initialize PCA with all components
        pca = PCA(n_components=None)
        pca.fit(X_train)

        # Calculate the cumulative variance explained by each component
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        # Determine the number of components required to explain 85% of the variance
        n_components = np.argmax(cumulative_variance >= 0.85) + 1

        # Get the loadings (which indicate the importance of each feature in each component)
        loadings = pca.components_[:n_components]

        # Sum the absolute values of the loadings across the selected components
        feature_importance = np.sum(np.abs(loadings), axis=0)

        # Sort the features by importance (descending) and select the top ones
        sorted_features_idx = np.argsort(feature_importance)[::-1]
        selected_features = X_train.columns[sorted_features_idx[:n_components]]

        # Output the selected feature names
        selected_features = selected_features.tolist()
    
    elif method == "Tree-based":
        # Use a tree-based regressor like RandomForestRegressor
        model = RandomForestRegressor(max_depth=20, random_state=random_state)
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        # Automatically select features with importance greater than a threshold (e.g., 0.01)
        selected_features = X_train.columns[importances > 0.01]
    
    else:
        raise ValueError("Unknown method! Please choose from 'RFE', 'Lasso', 'PCA' or 'Tree-based'")
    
    if lag_method == "all":
        # extract base feature names
        base_features = set(re.sub(r"\(-\d+\)", "", f) for f in selected_features)

        final_features = []
        for bf in sorted(base_features):
            for t in range(1, tau + 1):
                final_features.append(f"{bf}(-{t})")

        # convert to sorted list if needed
        final_features = sorted(final_features)

        selected_features = final_features
    
    print('selected features: ', selected_features)
    
    # Calculate duration and return with features if measure_time is True
    if measure_time:
        fs_duration = time.time() - fs_start_time
        return list(selected_features), fs_duration
    else:
        return list(selected_features)

def train_and_evaluate_model(X_train, y_train, X_test, y_test, models, feature_selection=None, model_save_path=None, label = None, target_scaling=False, plot=False, save_model=True, only_timing=False):
    """
    Train and evaluate multiple models with cross-validation, optional feature selection, scaling, and target normalization.
    
    Parameters:
    - X_train: pd.DataFrame - Training features.
    - y_train: pd.Series - Training target.
    - X_test: pd.DataFrame - Testing features.
    - y_test: pd.Series - Testing target.
    - feature_selection: bool - Whether to apply feature selection.
    - target_scaling: bool - Whether to normalize the target variable.
    - save_model: bool - Whether to save the model to disk.
    - model_save_path: str - Directory path where models will be saved.
    - only_timing: bool - If True, only measure feature selection time and skip model training.
    
    Returns:
    - dict - {model_name: (ID, Mean Squared Error, Mean Absolute Error, R^2 Score, Training Time, DateTime, Model Characteristics)}
           - If only_timing=True, returns dict with feature selection timing info
    """
    column_features = X_train.columns.to_list()
    fs_duration = None
    
    if feature_selection is not None:
        print("feature selection", feature_selection)
        
        # Measure time if only_timing mode is active
        if only_timing:
            column_features, fs_duration = select_features(X_train, y_train, feature_selection, measure_time=True)
        else:
            column_features = select_features(X_train, y_train, feature_selection, measure_time=False)

        if len(column_features) == 0:
            info_file = os.path.join(model_save_path, "info.txt")
            with open(info_file, "a") as f:
                print(f"{feature_selection} found 0 features, skipping.\n")
                f.write(f"{feature_selection} found 0 features, skipping.\n")
                #model_results[name] = [mse, mae, r2, duration, date_time, model_charact, column_features, str(y_pred_path), mape, rmse]
            if only_timing:
                return {'timing_info': {'feature_selection_method': feature_selection, 'duration': fs_duration if fs_duration else 0, 'num_features': 0}}
            return {'no_model':[99, 99, 99, 99, 99, [], [], 99, 99, 99]}
    
    # If only_timing mode and we have timing info, return early
    if only_timing and feature_selection is not None:
        return {'timing_info': {'feature_selection_method': feature_selection, 'duration': fs_duration, 'num_features': len(column_features)}}
    
    # If only_timing but no feature selection, return empty
    if only_timing:
        return {'timing_info': {'feature_selection_method': 'none', 'duration': 0, 'num_features': len(column_features)}},

    print(f"Features: {len(column_features)}")
    
    # Initialize the scaler for features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[column_features])
    X_test_scaled = scaler.transform(X_test[column_features])

    # Initialize the scaler for the target variable if target scaling is enabled
    if target_scaling:
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
    else:
        y_train_scaled = y_train
        y_test_scaled = y_test

    model_results = {}

    for name, model_info in models.items():

        y_pred_path = model_save_path / label / f"y_pred_{name.replace(' ', '_').lower()}.parquet"

        if y_pred_path.exists():
            print(f"File {y_pred_path} already exists. Skipping {name} model.")
            continue  # Skip the current model and move to the next one

        print(f"Evaluating {name}...")
        
        start_time = time.time()
        
        model = model_info['model']
        params = model_info['params']
        
        if name == 'Polynomial Regression':
            poly = model_info['model']
            X_train_poly = poly.fit_transform(X_train_scaled)
            X_test_poly = poly.transform(X_test_scaled)
            model = LinearRegression()
            model.fit(X_train_poly, y_train_scaled)
            y_pred_scaled = model.predict(X_test_poly)
        else:
            if params:
                #grid_search = RandomizedSearchCV(model, params, cv=3, scoring='neg_mean_squared_error', error_score=np.nan, n_jobs=1,)
                grid_search = GridSearchCV(model, params, cv=4, scoring='neg_mean_squared_error', error_score=np.nan, n_jobs=1,)
                
                grid_search.fit(X_train_scaled, y_train_scaled)
                best_model_for_current = grid_search.best_estimator_
            else:
                best_model_for_current = model
                cross_val_scores = cross_val_score(best_model_for_current, X_train_scaled, y_train_scaled, cv=3, scoring='neg_mean_squared_error')
                mean_cv_score = np.mean(cross_val_scores)
                print(f"{name} - Cross-Validated MSE: {-mean_cv_score}")

                best_model_for_current.fit(X_train_scaled, y_train_scaled)

            y_pred_scaled = best_model_for_current.predict(X_test_scaled)

        if target_scaling:
            y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        else:
            y_pred = y_pred_scaled
        
        y_true = y_test

        metrics = calculate_metrics(y_true, y_pred)

        # Calculate MAPE
        mape = metrics['mape']
        mse = metrics['mse']
        rmse = metrics['rmse']
        mae = metrics['mae']
        r2 = metrics['r2']
        smape = metrics['smape']
        wape = metrics['wape']
        cvrmse = metrics['cvrmse']

        end_time = time.time()
        duration = end_time - start_time
        
        model_charact = {
            'params': getattr(best_model_for_current, 'get_params', lambda: None)(),
            'feature_importances': getattr(best_model_for_current, 'feature_importances_', np.array([])).tolist(),
        }

        date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        (model_save_path / label).mkdir(parents=True,exist_ok=True)
        y_pred_path = model_save_path / label / f"y_pred_{name.replace(' ', '_').lower()}.parquet"
        path_to_store = str(y_pred_path)[str(y_pred_path).find('data'):]
        pd.DataFrame({target_column + "_pred": y_pred}).to_parquet(y_pred_path)

        print(path_to_store)
        print(y_pred_path)
        # Store the results with the model ID
        model_results[name] = [mse, mae, r2, duration, date_time, model_charact, column_features, str(path_to_store ), mape, rmse, smape, wape, cvrmse]
        with open(model_save_path / label / f"results_{name.replace(' ', '_').lower()}.json","w") as f:
            json.dump(str(model_results[name]), f, indent=4)
        # Save the model to disk if requested
        if save_model:
            path = model_save_path / label
            path.mkdir(exist_ok=True, parents=True)
            model_path = path / f"{name.replace(' ', '_').lower()}.joblib"
            joblib.dump(str(best_model_for_current), model_path)
            # Define the file path
            file_path = f'{path}/X_test.parquet'
            # Check if the file already exists
            if os.path.exists(file_path):
                print(f"File {file_path} already exists. Skipping the save operation.")
            else:
                # If the file does not exist, save X_test as a CSV
                X_test_df = pd.DataFrame(X_test_scaled, columns=column_features)
                X_test_df.to_parquet(file_path)
                print(f"File saved to {file_path}.")


         # Increment the model ID
    with open(model_save_path / label / f"results_all.json","w") as f:
        
        json.dump(str(model_results), f, indent=4)
    if plot:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, color='blue', edgecolor='k', alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs. True Values with Reference Line')
        plt.grid(True)
        plt.show()

    print('DICT RESULTS', model_results)
    return model_results


# %%
# Models to evaluate with hyperparameters
models = {
        'LR': {
            'model': LinearRegression(),
            'params': {}},
        'XGB': {
            'model': xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, enable_categorical=False, random_state=random_state),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6, 9],
                'subsample': [0.8, 1.0],  
            }
        },
        'LGBM': {
            'model': LGBMRegressor(n_jobs=-1, verbosity=-1, random_state=random_state),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6, 9],
                'num_leaves': [31, 50],  
            }
        },
        "MLP": {
            "model":MLPRegressor(max_iter=1000, early_stopping=True, n_iter_no_change=10, random_state=random_state),
                'params':{
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'learning_rate': ['adaptive'],
        }
        }
    }

models = {k: v for k, v in models.items() if k in models_to_eval}

print("models to eval:", models)

# %%
# loop for training and testing
df = df_train.copy()

if changepoint_removal is True:
    df = df.drop(columns=[column_change_point])
    var_names = df.columns
else:
    var_names = df.columns


for target_column in columns_target:
    results_list = []

    # Define the path to the results file for the current target_column
    results_file_path = data_directory / '' / f'results_ml_v4_{target_column}.parquet'

    # Check if the results file already exists
    if results_file_path.exists():
        print(f"File {results_file_path} already exists. Skipping {target_column}...")
        continue  # Skip the current iteration if the file exists

    # Your model training/evaluation code here, which will run only if the file doesn't exist
    print(f"Processing {target_column}...")

    single_activation_labels = ['all_all', 'RFE_all', 'Lasso_all', 'PCA_all', 'Tree-based_all', 'RFE_lag', 'Lasso_lag', 'PCA_lag', 'Tree-based_lag']
    label_activation_status = {lbl: False for lbl in single_activation_labels}

    for graph_name, graph_dict in dict_graphs_by_lag.items():
        print(f"Processing graph: {graph_name} for target: {target_column}")

        horizon = int(graph_name.split("_", 2)[1])


        column_lags = find_values_with_substring(graph_dict, target_column)
        if not column_lags:
            print(f"No lag values for target {target_column} in graph {graph_name}. Recording info and stopping loop.")
            info_file = data_directory / "info.txt"
            with open(info_file, "a") as f:
                f.write(f"No lag values for target {target_column} in graph {graph_name}\n")
            continue  # Stops the loop over dict_graphs_by_lag

        # Just selected columns
        unique_columns = list(set(column for column, _ in column_lags))

        #print('lagged columns: ', column_lags) 

        tau_min = int(horizon)
        tau_max = tau_min + input_size -1

        ################################################ create the dataframes  ################################################
        # For specific column-lag pairs
        df_train_selec_lag = create_lagged_features(df_train[var_names], column_lags=column_lags, target_column=target_column).dropna()

        # For all columns with lags ranging from tau_min to tau_max
        df_train_all_all= create_lagged_features(df_train[var_names], tau_min=tau_min, tau_max=tau_max, target_column=target_column).dropna()

        # For selected columns with all lags ranging from tau_min to tau_max
        df_train_selec_all = create_lagged_features(df_train[var_names], tau_min=tau_min, tau_max=tau_max, target_column=target_column,select_columns=unique_columns).dropna()

        df_test_reset = df_test[[column_change_point]].reset_index()

        # For specific column-lag pairs in test set
        df_test_selec_lag = create_lagged_features(df_test[var_names], column_lags=column_lags, target_column=target_column).reset_index()
        df_test_selec_lag = pd.merge(df_test_selec_lag, df_test_reset, on='index').drop(columns='index').dropna()

        # For all columns with lags ranging from tau_min to tau_max in test set
        df_test_all_all = create_lagged_features(df_test[var_names], tau_min=tau_min, tau_max=tau_max, target_column=target_column).reset_index()
        df_test_all_all = pd.merge(df_test_all_all, df_test_reset, on='index').drop(columns='index').dropna()

        # For selected columns with all lags ranging from tau_min to tau_max in test set
        df_test_selec_all = create_lagged_features(df_test[var_names], tau_min=tau_min, tau_max=tau_max, target_column=target_column,select_columns=unique_columns).reset_index()
        df_test_selec_all = pd.merge(df_test_selec_all, df_test_reset, on='index').drop(columns='index').dropna()
        ########################################################################################################################

        using_methods = graph_name.split("_")[-1]

        # Dont use the all_all approach for now

        if using_methods == 'all':

            # Prepare the datasets to evaluate with the CD and the feature selection methods
            datasets = [
                
                (f"{graph_name}_selec_feat_selec_lag", df_train_selec_lag, df_test_selec_lag , None),

                ("all_all", df_train_all_all,  df_test_all_all, None),

                ("feature_sel_RFE", df_train_all_all,  df_test_all_all, "RFE"),

                ("feature_sel_Lasso", df_train_all_all,  df_test_all_all, "Lasso"),

                ("feature_sel_PCA", df_train_all_all,  df_test_all_all, "PCA"),

                ("feature_sel_Tree-based", df_train_all_all,  df_test_all_all, "Tree-based")

                
            ]

            if only_fs_time_evaluation is True:  
                add_lag_all_to_feature_selection = False

            if add_lag_all_to_feature_selection and "delete_others" not in graph_name:
                datasets.append((f"{graph_name}_selec_feat_all_lags", df_train_selec_all, df_test_selec_all, None))

            # Dont do predictions for edited without lag selection
            datasets = [item for item in datasets if "edited_all_selec_feat_selec_lag" not in item[0]]

        else:
            datasets = [
                (f"{graph_name}_selec_feat_selec_lag", df_train_selec_lag, df_test_selec_lag , None),
                (f"{graph_name}_selec_feat_all_lags", df_train_selec_all, df_test_selec_all, None)
            ]
        
        new_entries = []
        updated_datasets = []
    
        # Add to the traditional feature selection methods the all lagging approaches
        for name, train, test, method in datasets:
            if name.startswith("feature_sel_"):
                if add_lag_all_to_feature_selection:
                    new_entries.append((name + "_all", train, test, method + "__all" + "/" + str(input_size)))
                    # old one with "_lag"
                updated_datasets.append((name + "_lag", train, test, method+ "__lag" + "/" + str(input_size)))
                    # new one with "_all"
            else:
                updated_datasets.append((name, train, test, method))

            # add the new ones
        datasets = updated_datasets + new_entries

        # Main loop for the training and evaluation
        for label, df_train_in, df_test_with_change_point, feature_selection in datasets:
            print(label)
            should_skip = False
            for special_label in single_activation_labels:
                if special_label in label:
                    if label_activation_status[special_label]:
                        print(f'SKIPPING THIS LABEL ({special_label})!\n')
                        should_skip = True
                        break
                    label_activation_status[special_label] = True
                    print(f'Processing {special_label} for the first time!\n')
                    # You might want to break or continue here depending on your logic.
                    break
            if should_skip:
                continue

            print(df_test_with_change_point.columns)

            df_train_in.columns = df_train_in.columns.str.replace('_x$', '', regex=True)
            df_test_with_change_point.columns = df_test_with_change_point.columns.str.replace('_x$', '', regex=True)
            df_test_in = df_test_with_change_point

            print(f"\nEvaluating Model for {label} for horizon {horizon} and target {target_column}, datetime: {datetime.datetime.now()}...\n")
            print(target_column)

            # Prepare training data
            X_train = df_train_in.drop(columns=[target_column])
            y_train = df_train_in[target_column]
            
            # Prepare testing data
            X_test = df_test_in.drop(columns=[target_column])
            y_test = df_test_in[target_column]

            # Train and evaluate the model (or just measure feature selection time)
            dict_results = train_and_evaluate_model(X_train, y_train, X_test, y_test, models, feature_selection, model_save_path=data_directory / target_column, label = label, target_scaling=True, only_timing=only_fs_time_evaluation)

            # Handle timing-only mode
            if only_fs_time_evaluation:
                if 'timing_info' in dict_results:
                    timing_info = dict_results['timing_info']
                    results_list.append({
                        'graph_name': str(graph_name),
                        'horizon': horizon,
                        'label': label,
                        'feature_selection_method': timing_info['feature_selection_method'],
                        'duration': timing_info['duration'],
                        'num_features': timing_info['num_features'],
                        'train_size': len(X_train),
                        'total_original_features': len(X_train.columns),
                        'target': str(target_column),
                        'date_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                continue  # Skip model training and evaluation

            # Get the results and store them in the results list
            for model, results in dict_results.items(): 
                mse = results[0]
                mae = results[1]
                r2 = results[2]
                duration = results[3]
                date_time = results[4]
                model_param = results[5]
                column_features = results[6]
                model_id = "_".join([model, label, target_column])
                mape = results[8]
                rmse = results[9]
                smape = results[10]
                wape = results[11]
                cvrmse = results[12]

                # Store everything in a dataframe
                results_list.append({
                    'graph_name': str(graph_name),
                    'horizon': horizon,
                    'label': label,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape,
                    'smape': smape,
                    'wape': wape,
                    'cvrmse': cvrmse,
                    'r2': r2,
                    'model': str(model),
                    'duration': duration,
                    'date_time': str(date_time),
                    'target': str(target_column),
                    'features': column_features,
                    'n features': len(column_features),
                    'model_param': str(model_param),
                    'target': str(target_column),
                    'df': str(results[7]),
                    'model_id': str(model_id)
                })

    try:
        results_df = pd.DataFrame(results_list)
        #display(results_df)
        
        # Save results based on mode
        if only_fs_time_evaluation:
            # Add causal discovery method timing from lag folders
            if len(results_df) > 0:
                # Get the last row values for common columns
                last_row = results_df.iloc[-1]
                train_size = last_row['train_size']
                total_original_features = last_row['total_original_features']
                target = last_row['target']
                
                # Add timing info for each causal discovery method (lag folders)
                for folder_path in [f for f in data_directory.glob("*") if f.is_dir() and f.name.startswith("lag")]:
                    try:
                        folder_name = folder_path.name
                        info_path = folder_path / 'info.json'
                        
                        if info_path.exists():
                            with open(info_path, 'r') as f:
                                info_data = json.load(f)
                            
                            # Extract time_graph_generation
                            time_graph_generation = info_data.get("time_graph_generation", None)
                            
                            if time_graph_generation is not None:
                                # Extract label: everything after the second "_"
                                parts = folder_name.split("_")
                                if len(parts) >= 3:
                                    label = "_".join(parts[2:])
                                else:
                                    label = folder_name
                                
                                # Add row for causal discovery method
                                results_list.append({
                                    'graph_name': folder_name,
                                    'horizon': None,
                                    'label': label,
                                    'feature_selection_method': label,
                                    'duration': time_graph_generation,
                                    'num_features': None,
                                    'train_size': train_size,
                                    'total_original_features': total_original_features,
                                    'target': target,
                                    'date_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                })
                    except Exception as e:
                        print(f"Error processing causal discovery timing for {folder_path.name}: {e}")
                
                # Recreate dataframe with all rows including causal discovery methods
                results_df = pd.DataFrame(results_list)
            
            # Save feature selection timing results
            print(f"Saving feature selection timing results for {target_column}")
            print(results_df.info())
            timing_path = data_directory / target_column / 'feature_selection_timing.parquet'
            timing_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_parquet(timing_path)
            print(f"Timing results saved to {timing_path}")
        else:
            # Save regular model evaluation results
            results_df['horizon'] = results_df['horizon'].astype(int)
            results_df['model_param'] = results_df['model_param'].astype(str)
            print(results_df.info())
            results_df.to_parquet(data_directory / f'results_ml_v4_{target_column}.parquet')
    except Exception as e:
        print(f"Error saving results for {target_column}: {e}")
        info_file = data_directory / "info.txt"
        with open(info_file, "a") as f:
            f.write(f"{target_column} error in processing: {e}\n")


