import subprocess
from pathlib import Path
import sys
import logging
import time
from datetime import datetime
import os


start_datetime = time.time()

# Set up paths and logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')
sys.stdout.reconfigure(line_buffering=True)
current_path = Path(__file__).resolve().parent if '__file__' in globals() else Path().resolve()
print(f"Current path: {current_path}", flush=True)

# Define the numebr of experiment
experiment_number  = 1

# Select the data fractions to evaluate (1 means the complete train set)
fractions = [1]

# Select the causal discovery methods to evaluate
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

suffix = "_prepared"

# Select the models to evaluate. To run the time-series algorithms (NHITS, TSMixerx, "LSTM"), make sure to start the corresponding container
models_to_eval = ["LR", "XGB", "LGBM", "MLP", "NHITS", "TSMixerx", "LSTM"]

# Select the max tau to use (max lag to find features)
tau_list = [5]

for tau in tau_list:
    
    if len(tau_list) > 1:
        experiment_name = 'experiment_' + str(experiment_number) + str(tau)
    else:
        experiment_name = 'experiment_' + str(experiment_number)
    data_root = current_path.parent.parent / 'data' / 'gold' / experiment_name

    print(data_root)

    data_root.mkdir(parents=True, exist_ok=True)


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

    def run_script(script_path, args):
        command = ["python", script_path] + args
        print(f"Running {script_path} with arguments {args}", flush=True)
        with subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr, text=True, bufsize=1) as proc:
            proc.wait()

    if True:
        run_script(
            str(current_path / "001_causal_graph_generator.py"), ["--experiment_name", str(experiment_name), "--config", str(config)]  # File to generate the graphs from CD
        )

        experiment_folders = sorted([
            f for f in data_root.iterdir()
            if f.is_dir() and f.name.startswith("exp_")
        ])
    else:
        #Use fallback; "xyz" should be interpreted as a subfolder in data_root
        experiment_folders = [data_root / "xyz"]

    print("Experiment folders found:", [f.name for f in experiment_folders])

    scripts = [
    (current_path / "002_causal_reg_predict_complete.py", ["--datadir", "{folder}", "--models_to_eval", str(models_to_eval)]), # File to run the regression models
    (current_path / "003_causal_ts_predict.py", ["--datadir", "{folder}", "--models_to_eval", str(models_to_eval)]), # File to run the time series models
    (current_path / "004_evaluation.py", ["--datadir", "{folder}"]), # File to run the final evaluation (specially the inervention)
    ]

    # run all scripts, but each script over all folders before moving to the next script
    for script, script_args_template in scripts:
        print(f"🚀 Running {script.name} for all folders...")
        start_script = time.time()

        for folder in experiment_folders:
            print(f"Processing {folder} with {script.name}", flush=True)
            start = time.time()

            # replace placeholders
            script_args = [arg.format(folder=folder) for arg in script_args_template]

            run_script(str(script), script_args)

            elapsed = time.time() - start
            print(f"✅ Completed {script.name} for {folder.name} in {elapsed:.2f}s", flush=True)

        elapsed_script = time.time() - start_script
        print(f"Finished {script.name} for all folders in {elapsed_script:.2f}s ({elapsed_script/60:.2f} min)\n")


    end_time = time.time()
    start_dt, end_dt = [datetime.fromtimestamp(s) for s in [start_datetime, end_time]]
    print(f"experiment started at: {start_dt}\nexperiment ended at: {end_dt}\ntotal duration in hours: {(end_dt - start_dt).total_seconds()/3600}")

    errors = []


    # Walk through all subdirectories to get a summary of errors
    for root, dirs, files in os.walk(data_root):
        if "info.txt" in files:
            file_path = os.path.join(root, "info.txt")
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("error:"):
                        # keep the whole line (including "error:")
                        errors.append(line)

    # Combine and print
    all_errors = "\n".join(errors)
    print(all_errors)