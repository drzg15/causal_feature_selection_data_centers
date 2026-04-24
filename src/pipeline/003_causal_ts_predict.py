# %%
# %%
import json
import sys
sys.path.append('../')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from pathlib import Path
import itertools
import argparse
from utils import calculate_metrics
import time
import re
import os
import glob
import requests
import datetime
import tqdm
import random
np.random.seed(123)
random.seed(123)
# %% # stablish paths and parameters for the case the notebook is run as script
current_path = Path(__file__).resolve().parent if '__file__' in globals() else Path().resolve()
print(current_path)

# This file only works if the container is first started as explained in the README.md of the container


################# CODE FOR RUNNING STANDALONE, WITHOUT THE COMPLETE PIPELINE####################
experiment =  1
parquet_folder = current_path.parent.parent / 'data' / 'gold' / f'experiment_{experiment}' / "exp_8_minidc__raw_data_exp_3__f1_g1"
models_to_eval = ["TSMixerx", "NHITS"]

################# CODE FOR RUNNING STANDALONE, WITHOUT THE COMPLETE PIPELINE####################

# %% # load parameters if the notebook is run in the pipeline
parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str, default=parquet_folder)
parser.add_argument("--models_to_eval", type=str, default=models_to_eval)

def list_of_strings(arg):
    return arg.split(',')

args, _ = parser.parse_known_args()

data_directory = Path(args.datadir)
models_to_eval = args.models_to_eval

# %% # load data
df_train = pd.read_parquet(data_directory / 'train.parquet')
df_test = pd.read_parquet(data_directory / 'test.parquet')
tau_file = data_directory / "info.txt"

# Get tau from the info.txt file
tau = None
with open(tau_file, "r") as f:
    for line in f:
        if line.strip().startswith("tau"):
            # Extract number after "tau = ..."
            match = re.search(r"tau\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
            if match:
                tau = int(match.group(1))
                break
if tau is None:
    raise ValueError("tau not found in info.txt")
print("Extracted tau:", tau)

results_ml_files = glob.glob(os.path.join(data_directory, "results_ml*.parquet"))

# %% Set models and its endpoints

models = {
    "NHITS": {
        "endpoint": "http://localhost:8020",
    },
    "TSMixerx": {
        "endpoint": "http://localhost:8021",
    },
    "LSTM": {
        "endpoint": "http://localhost:8025",
    },

    "TiDE": {
        "endpoint": "http://localhost:8024",
    },
    "TFT": {
        "endpoint": "http://localhost:8023",
    },
}

# %%
models = {k: v for k, v in models.items() if k in models_to_eval}

# %%
def normalize_features(train_df, test_df, target, features):
    features = [f for f in features if f != target]

    # Separate normalization for train and test to avoid data leakage
    scaler = StandardScaler()
    train_scaled, test_scaled = train_df.copy(), test_df.copy()
    train_scaled[features] = scaler.fit_transform(train_df[features])

    scaler = StandardScaler()
    test_scaled[features]  = scaler.fit_transform(test_df[features])
    return train_scaled, test_scaled, features

def prepare_data(df_raw, target, features, start_date="2000-01-01", freq="h"):
    df_proc = df_raw[[target] + features].copy()
    df_proc["unique_id"] = 1
    df_proc["ds"] = pd.date_range(start=start_date, periods=len(df_proc), freq=freq)
    df_proc["ds"] = pd.to_datetime(df_proc["ds"])
    df_proc = df_proc.rename(columns={target: "y"})
    df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]
    return df_proc

def prepare_api_payload(df, features, freq="h", input_size=4, num_samples=5, use_auto=False):
    """
    Prepare the training data for external API call
    
    Args:
        df: DataFrame with columns ['unique_id', 'ds', 'y'] + features
        features: List of exogenous features (will be used as hist_exog)
        freq: Frequency string (e.g., 'h', 'D', 'M')
        num_samples: Number of samples for prediction intervals
        use_auto: Whether to use automatic model selection
    
    Returns:
        dict: API payload in the required format
    """
    # Prepare data list - exog stays empty as per API structure
    data_list = []
    for _, row in df.iterrows():
        data_entry = {
            "unique_id": str(row["unique_id"]),
            "ds": row["ds"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(row["ds"], pd.Timestamp) else str(row["ds"]),
            "y": float(row["y"]) if pd.notna(row["y"]) else None,
            "exog": {feat: row[feat] for feat in features}
        }
        data_list.append(data_entry)
    
    # Build the complete payload
    payload = {
        "data": data_list,
        "freq": freq,
        "h": 1,  # forecast horizon
        "input_size": input_size,  # input size (tau)
        "futr_exog": [],  # List of future exogenous variable names (empty for now)
        "hist_exog": features,  # Features are historical exogenous variables
        "stat_exog": [],  # Static exogenous (empty for now)
        "use_auto": use_auto,
        "num_samples": num_samples
    }
    
    return payload

def prepare_forecast_payload(df, features):
    """
    Prepare the forecast data for external API call
    
    Args:
        df: DataFrame with columns ['unique_id', 'ds', 'y'] + features
        features: List of exogenous features
    
    Returns:
        dict: API payload in the required format for forecast
    """
    # Prepare data list
    data_list = []
    for _, row in df.iterrows():
        data_entry = {
            "unique_id": str(row["unique_id"]),
            "ds": row["ds"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(row["ds"], pd.Timestamp) else str(row["ds"]),
            "y": float(row["y"]) if pd.notna(row["y"]) else None,
            "exog": {feat: row[feat] for feat in features}
        }
        data_list.append(data_entry)
    
    # Build the payload for forecast
    payload = {
        "data": data_list
    }
    
    return payload

def train_and_forecast_model(model_name, model_info, train_relevant, test_relevant, features, tau, label, target, idx, file):
    """
    Train the model and perform rolling forecast using API calls.
    
    Returns:
        df_merged: DataFrame with predictions
        metrics: dict of metrics
        duration: time taken
    """
    start_time = time.time()
    
    # Prepare API payload for external training
    api_payload = prepare_api_payload(
        df=train_relevant, 
        features=features, 
        freq="h", 
        input_size=tau,
        num_samples=5, 
        use_auto=False
    )

    model_id = model_name + '_' + Path(file).stem + "_" + str(idx)
    # get request to check if the model was already trained
    model_trained = False

    if not model_trained:
        response = requests.post(
            f"{model_info['endpoint']}/train",
            params={"model_name": model_id},
            json=api_payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            print("API call successful.")
            api_result = response.json()
            print(api_result)
        else:
            raise Exception(f"API call failed with status code {response.status_code}: {response.text}")
        status_url = f"{model_info['endpoint']}/train/status"
        job_id = api_result.get("job_id")
        if job_id:
            max_wait = 600  # steps
            waited = 0
            while waited < max_wait:
                status_response = requests.get(
                    status_url,
                    params={"job_id": job_id},
                    headers={"Content-Type": "application/json"}
                )
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if status_data.get("status") == "completed":
                        print(f"Job {job_id} completed.")
                        break
                    else:
                        print(f"Job {job_id} status: {status_data.get('status')}")
                else:
                    print(f"Status check failed: {status_response.status_code} {status_response.text}")
                time.sleep(2)
                waited += 1
            else:
                print(f"Job {job_id} did not complete within {max_wait * 2} seconds.")
        else:
            print("No job_id found in API result.")
    
    # rolling forecast - prepare test payload
    train_relevant_up_to_test = pd.concat([train_relevant.sort_values("ds").iloc[-tau:], test_relevant], ignore_index=True)
    # collect input dataframes
    inputs = [train_relevant_up_to_test.iloc[i:i+tau] for i in range(len(train_relevant_up_to_test) - tau + 1)]
    forecasts = []
    for input_df in tqdm.tqdm(inputs, desc="Forecasting"):
        api_test_payload = prepare_forecast_payload(
            df=input_df,
            features=features
        )
        response = requests.post(
            f"{model_info['endpoint']}/forecast",
            params={"model_name": model_id},
            json=api_test_payload,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            # print("Prediction API call successful.")
            forecast_df = pd.DataFrame.from_dict(response.json())
            forecasts.append(forecast_df)
        else:
            raise Exception(f"Prediction API call failed with status code {response.status_code}: {response.text}")

    # Concatenate forecasts and filter required columns
    median_col = [col for col in forecasts[0].columns if col.endswith("-median")][0]
    preds_df = pd.concat([f[["unique_id", "ds", median_col]] for f in forecasts], ignore_index=True)
    preds_df["model"] = median_col.replace("-median", "")
    preds_df = preds_df.rename(columns={median_col: "y_pred"})
    test_relevant["ds"] = pd.to_datetime(test_relevant["ds"])
    preds_df["ds"] = pd.to_datetime(preds_df["ds"])
    df_merged = test_relevant[["ds", "y"]].merge(preds_df, on="ds", how="left")
    
    y_true = df_merged["y"].values
    y_pred = df_merged["y_pred"].values
    metrics = calculate_metrics(y_true, y_pred)
    end_time = time.time()
    duration = end_time - start_time
    
    return df_merged, metrics, duration

def prepare_cross_validation_payload(train_df, test_df, features, freq="h", input_size=4, num_samples=5, use_auto=False):
    """
    Prepare the cross validation payload for the new API endpoint.
    
    Args:
        train_df: DataFrame for training
        test_df: DataFrame for testing
        features: List of exogenous features
        freq: Frequency string
        input_size: Input size (tau)
        num_samples: Number of samples
        use_auto: Whether to use automatic model selection
    
    Returns:
        dict: Payload for cross_validation endpoint
    """
    def prepare_data_list(df):
        data_list = []
        for _, row in df.iterrows():
            data_entry = {
                "unique_id": str(row["unique_id"]),
                "ds": row["ds"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(row["ds"], pd.Timestamp) else str(row["ds"]),
                "y": float(row["y"]) if pd.notna(row["y"]) else None,
                "exog": {feat: row[feat] for feat in features}
            }
            data_list.append(data_entry)
        return data_list
    
    train_data = prepare_data_list(train_df)
    test_data = prepare_data_list(test_df)
    
    payload = {
        "train_data": train_data,
        "test_data": test_data,
        "freq": freq,
        "h": 1,
        "input_size": input_size,
        "futr_exog": [],
        "hist_exog": features,
        "stat_exog": [],
        "use_auto": use_auto,
        "num_samples": num_samples,
        "val_size": None,
        "test_size": None,
        "step_size": 1,
        "n_windows": None
    }
    
    return payload

def train_and_forecast_model_cross_val(model_name, model_info, train_relevant, test_relevant, features, tau, label, target, idx, file):
    """
    Train the model and perform cross validation forecast using the new API endpoint.
    
    Returns:
        df_merged: DataFrame with predictions
        metrics: dict of metrics
        duration: time taken
    """
    start_time = time.time()
    
    # Prepare cross validation payload
    api_payload = prepare_cross_validation_payload(
        train_df=train_relevant,
        test_df=test_relevant,
        features=features,
        freq="h",
        input_size=tau,
        num_samples=5,
        use_auto=False
    )
    
    model_id = model_name + '_' + Path(file).stem + "_" + str(idx)
    
    # Call cross_validation endpoint
    response = requests.post(
        f"{model_info['endpoint']}/cross_validation",
        params={"model_name": model_id},
        json=api_payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        print("Cross validation API call successful.")
        forecast_data = response.json()
        # forecast_data is a list of dicts representing the DataFrame rows
    else:
        raise Exception(f"Cross validation API call failed with status code {response.status_code}: {response.text}")
    
    # Create DataFrame from the response
    preds_df = pd.DataFrame.from_dict(forecast_data)

    preds_df = preds_df.iloc[tau:]
    # Rename the model column (e.g., "NHITS") to "y_pred"
    if model_name in preds_df.columns:
        median_col = model_name
    else:
        median_col = [col for col in preds_df.columns if col.endswith("-median")][0]
    preds_df["model"] = median_col.replace("-median", "")
    preds_df = preds_df.rename(columns={median_col: "y_pred"})
    # Add model column
    preds_df["model"] = model_name
    # Ensure ds is datetime
    preds_df["ds"] = pd.to_datetime(preds_df["ds"])
    df_merged = preds_df
    
    y_true = df_merged["y"].values
    y_pred = df_merged["y_pred"].values
    metrics = calculate_metrics(y_true, y_pred)
    end_time = time.time()
    duration = end_time - start_time
    
    return df_merged, metrics, duration

# %%

# Main analysis loop
for file in results_ml_files:
    collected_rows = []
    all_results = []

    raw_df = pd.read_parquet(file)  # keep this pristine
    raw_df = raw_df.drop_duplicates(subset="label")
    raw_df = raw_df[~raw_df["label"].str.endswith("_lag")]
    raw_df["features"] = raw_df["features"].apply(lambda x: [re.sub(r"\(.*?\)", "", a) for a in x])

    raw_df["features_key"] = raw_df["features"].apply(lambda x: ','.join(set(sorted(x))))
    print(raw_df['label'].unique())
    print(raw_df['label'].nunique())
    deduplicated_df = raw_df.drop_duplicates(subset=["features_key", "target"])
    for idx, row in tqdm.tqdm(deduplicated_df.iterrows(), desc="Processing rows", total=len(deduplicated_df)):
        features: list[str] = list(set(re.sub(r"\(.*?\)", "", s) for s in row.features))

        target = row.target
        label = row.label
        print(f"\n Features: {features}")
        print(f" Target: {target}")

        # normalize train
        # df_train_scaled, df_test_scaled, features = normalize_features(df_train, df_test, target, features)
        features = [f for f in features if f != target]
        for model_name, model_info in models.items():

            # scaling in model service
            train_relevant = prepare_data(df_train, target, features,
                                    start_date="2000-01-01", freq="h")
            test_relevant  = prepare_data(df_test, target, features,
                                    start_date=train_relevant["ds"].max() + pd.Timedelta(hours=1),
                                    freq="h")
            df_merged, metrics, duration = train_and_forecast_model_cross_val(model_name, model_info, train_relevant, test_relevant, features, tau, label, target, idx, file)
            # save results
            all_results.append(df_merged[["ds", "y", "y_pred", "model"]])

            target_folder_model_result = data_directory / target / label

            predictions_df = df_merged[['y_pred']].rename(columns={"y_pred": f"{target}_pred"}).reset_index(drop=True)

            results_df_path = target_folder_model_result / f"y_pred_{model_name}.parquet"
            predictions_df.to_parquet(results_df_path)

            with open(target_folder_model_result / "metrics.json", "w") as f:
                json.dump(metrics, f)

            features_with_target = features.copy()
            features_with_target.append(target)

            mse = metrics['mse']
            mae = metrics['mae']
            r2 = metrics['r2']
            mape = metrics['mape']
            rmse = metrics['rmse']
            date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            column_features = features_with_target
            model_charact = model_info['endpoint']
            path_to_store = target_folder_model_result
            to_save_results = str([mse, mae, r2, duration, date_time, model_charact, column_features, str(path_to_store), mape, rmse])

            # Prepare JSON file path
            json_path = target_folder_model_result / f"results_{model_name}.json"

            # Save to JSON
            with open(json_path, "w") as f:
                json.dump(to_save_results, f, indent=4)

            updated_row = row.to_dict()  # start from original
            updated_row.update({
                "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "duration": duration,
                    "date_time": date_time,
                    "df": "data/" + str(results_df_path).split("data/", 1)[-1],
                    "mape": mape,
                    "rmse": rmse,
                    "model": model_name,
                    "model_param": str(model_charact),
                    "features": features_with_target,
                    "n features": len(features_with_target)*tau, # Sum also tau, the target variabl ei
                    "model_id": f"{model_name}_{row.label}_{target}"
                })
            collected_rows.append(updated_row)
            # additionally we have the same results for other labels with the same features and target
            similar_rows = raw_df[(raw_df["features_key"] == row["features_key"]) & (raw_df["target"] == row["target"])]
            for _, sim_row in similar_rows.iterrows():
                if sim_row["label"] != row["label"]:
                    updated_sim_row = sim_row.to_dict()
                    updated_sim_row.update({
                        "mse": mse,
                        "mae": mae,
                        "r2": r2,
                        "duration": duration,
                        "date_time": date_time,
                        "df": "data/" + str(results_df_path).split("data/", 1)[-1],
                        "mape": mape,
                        "rmse": rmse,
                        "model": model_name,
                        "model_param": str(model_charact),
                        "features": features_with_target,
                        "n features": len(features_with_target)*tau, # Sum also tau, the target variabl ei
                        "model_id": f"{model_name}_{sim_row['label']}_{target}"
                    })
                    collected_rows.append(updated_sim_row)
                    print(f" Also updated {sim_row['label']} with same results.")
    file_to_save = file.replace("ml", "ts")
    final_df = pd.DataFrame(collected_rows)
    final_df.to_parquet(file_to_save)

    df_ml = pd.read_parquet(file)

    df_ml = pd.concat([df_ml, final_df])

    df_ml.to_parquet(file)
