# %%
import json
import os
import pandas as pd
import sys
import pandas as pd
import numpy as np
import random
from scipy import stats
from pathlib import Path
import argparse
import ast
from utils import calculate_metrics

sys.path.append('../')

np.random.seed(123)
random.seed(123)
random_state = 123



# %%

pd.set_option('mode.chained_assignment', None)

current_path = Path(__file__).resolve().parent if '__file__' in globals() else Path().resolve()
path_to_data_folder = current_path.parent.parent

# %%

parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str, default=None)
args, _ = parser.parse_known_args()
data_directory = Path(args.datadir)
# Get all data from the fol
list_of_graphs = []
for folder_path in [f for f in data_directory.glob("*") if f.is_dir() and f.name.startswith("lag")]:
    list_of_graphs.append(folder_path)

# get only the first folder to extract the targets and the change points
folder_path = list_of_graphs[0]

# Get all data from the fol
with open(folder_path  / 'info.json', 'r') as f:
    data = json.load(f)

columns_target = data.get("target", None)
column_change_point = data.get("setpoint_change", None)  

print(f"column_change_point: {column_change_point}")
print(f"columns_target: {columns_target}")


# %%
def extract_around_change_points(df, change_point_column, before_window=0, after_window=1, drop_point_column=True):
    """
    Extracts single data points occurring exactly 'after_window' steps after each change point in a DataFrame.

    A change point is detected whenever the value in the specified `change_point_column` changes from the previous row.
    For each change point, this function extracts the single row that is exactly 'after_window' steps after the change point,
    concatenates all such rows, and returns the combined DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    change_point_column : str
        Name of the column used to identify change points (value change triggers extraction).
    before_window : int, default=0
        Number of rows before each change point to include in the window. (Currently unused in logic.)
    after_window : int, default=1
        Number of rows after each change point to extract the single point.
    drop_point_column : bool, default=True
        Whether to drop the change point column from the output DataFrame.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of single rows extracted exactly 'after_window' steps after each change point.
        Rows may include duplicates if multiple change points lead to the same index.

    Notes
    -----
    - Change points are detected as any row where the `change_point_column` value is different from the previous row.
    - For each change point at index 'point', the row at index 'point + after_window' is extracted, if it exists.
    - Rows are combined and duplicates dropped.
    - Only the `after_window` parameter is used for extraction. If before_window > 0, logic may need to be updated.
    - If `drop_point_column` is True, the resulting DataFrame does not include the column used for change point detection.
    """
    # Step 1: Identify change points
    df_temp = df.copy().reset_index(drop=True)  # Create a temporary copy to identify change points
    df_temp['Change_Point'] = df_temp[change_point_column] != df_temp[change_point_column].shift(1)

    # Find indices where changes occur
    change_points = df_temp.index[df_temp['Change_Point']].tolist()

    # Initialize list to collect data
    extracted_data = []

    # Step 2: Extract the single point after each change point
    for point in change_points:
        target_index = point + after_window
        if target_index < len(df):
            point_row = df.iloc[[target_index]].copy()  # Extract single row as DataFrame
            extracted_data.append(point_row)

    # Step 3: Combine all extracted data into a single dataframe
    if extracted_data:
        df_change_point = pd.concat(extracted_data).drop_duplicates()
    else:
        df_change_point = pd.DataFrame()  # Return empty DataFrame if no points extracted

    if drop_point_column and not df_change_point.empty:
        df_change_point = df_change_point.drop(columns=[change_point_column])

    return df_change_point


# %%
pd.set_option('mode.chained_assignment', None)


def get_points_after_changepoint(df, changepoint_column, timesteps_after_changepoint):
    """
    Returns rows in the DataFrame occurring at a fixed lag after change points in the specified column.

    For each detected change in the values of `changepoint_column`, this function returns the row that is 
    `timesteps_after_changepoint` steps after the change occurs.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    changepoint_column : str
        The name of the column in which to detect change points (rows where column value differs from the previous row).
    timesteps_after_changepoint : int
        The number of time steps to count after a change before extracting the row.
    
    Returns
    -------
    pd.DataFrame
        Subset of rows which occur exactly `timesteps_after_changepoint` steps after each change in `changepoint_column`.
    """
    return df[(df[changepoint_column].shift(timesteps_after_changepoint+1) != df[changepoint_column].shift(timesteps_after_changepoint)) & (~df[changepoint_column].shift(timesteps_after_changepoint+1).isna())]


for target_column in columns_target:
    file_path = data_directory / f'z1_results_evaluated_{target_column}.parquet'

    try:
        results_df = pd.read_parquet(data_directory / f'results_ml_v4_{target_column}.parquet')
    except Exception as e:
        print(f"Skipping {target_column} due to error: {e}")
        continue

    change_point_column = column_change_point
    list_dataframes = []
    after_changepoint_eval_dict = {}
    after_changepoint_eval_dict_mse = {}
    after_changepoint_eval_dict_mape = {}
    after_changepoint_eval_dict_diff = {}
    for idx, df_row in results_df.iterrows():
        
        if df_row["df"] == "0":
            print(f"Skipping {target_column} because the DataFrame is None.")
            continue
        df_intern_pred =  pd.read_parquet(path_to_data_folder/Path(df_row["df"]))
        df_x_test = pd.read_parquet(data_directory / "test.parquet").reset_index(drop=True)
        df_x_test = df_x_test.iloc[len(df_x_test)- len(df_intern_pred):].reset_index(drop=True)
        print("x test", len(df_x_test), "pred len", len(df_intern_pred))
        df_intern = pd.concat([df_intern_pred,df_x_test],axis=1)
        df_row = df_row.to_dict()

        # Calculate metrics one step after the intervention
        for after_window in range(1,2):
            df_row_window = df_row.copy()
            df_row_window["features"] = list(df_row_window["features"])
            df_row_window.pop("model_param")
            relevant_values_df = extract_around_change_points(df_intern, change_point_column, before_window=0, after_window=after_window, drop_point_column= False)

            y_true_complete = df_intern[target_column]
            relevant_values_df = relevant_values_df.dropna()

            y_true = relevant_values_df[target_column]
            y_pred = relevant_values_df[target_column + '_pred']

            metrics = calculate_metrics(y_true, y_pred, y_true_complete, normalize=True)

            df_row_window['mse_window'] = metrics['mse']
            df_row_window['mae_window'] = metrics['mae']
            df_row_window['r2_window'] = metrics['r2']
            df_row_window['mape_window'] = metrics['mape']
            df_row_window['rmse_window'] = metrics['rmse']
            df_row_window['smape_window'] = metrics['smape']
            df_row_window['wape_window'] = metrics['wape']
            df_row_window['cvrmse_window'] = metrics['cvrmse']

            print("mae window", metrics['mae'])

            try:
                diff = (relevant_values_df[target_column] - relevant_values_df[target_column + '_pred']) 
            except KeyError:
                print(f"Skipping {target_column} due to missing predicted column.")
                continue

            df_row_window['after_window'] = after_window
            df_row_window['outliers_num'] = 0
            df_row_window = {key: [value] for key, value in df_row_window.items()}
            list_dataframes.append(pd.DataFrame.from_dict(df_row_window))


    print("target saving:", data_directory / f'z1_results_evaluated_{target_column}.parquet')
    dataframes = pd.concat(list_dataframes).reset_index(drop=True)
    dataframes.to_parquet(data_directory / f'z1_results_evaluated_{target_column}.parquet')

    print(f'done with {target_column}')


