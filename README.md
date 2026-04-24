# Repository for the Paper

This repository contains the code and supplementary materials associated with the paper.  
The file **`online_appendix.html`** includes all tables and visualizations presented in the manuscript, as well as the complete set of results for every dataset, target variable, feature selection method, ML algorithm, and lag strategy.

---

## How to Run the Experiments

This guide describes how to set up and reproduce the experiments for the paper. Please ensure that all required dependencies are installed before proceeding.

### Prerequisites

1. **Clone the repository**

2. **Set up the environment**
   - Create a virtual environment with **Python 3.10.19**.
   - Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```

3. **Data availability**
   - Experimental data are located in the `data/silver` folder.
   - The raw data for **MiniDC** and **simulations** are provided.
   - Due to privacy constraints imposed by data center operators, the raw data for **DC1** and **DC2** are confidential and not provided.  
     However, complete results for all datasets are included.

4. **Running the main experiments**
   - The main experiment orchestrator is **`000_pipeline.py`**.  
     This script allows selection of:
       - ML algorithms for evaluation  
       - Datasets  
       - Causal feature selection methods  

   - Use the notebook **`006_partial_correlation.ipynb`** to evaluate the τ-lag selection for the experiments.

   - You can run this script directly to execute the full experimental pipeline.  
     It orchestrates the following steps in order:
       1. **`001_causal_graph_generator.py`** – generates causal graphs  
       2. **`002_causal_reg_predict_complete.py`** – evaluates regression models  
       3. **`003_causal_ts_predict.py`** – evaluates time-series models (see the next section)  
       4. **`004_evaluation.py`** – analyzes interventions and produces the final aggregated results dataset

### Running Time-Series Models

- To run experiments involving time-series models, you must first start the corresponding **Docker container** before running the pipeline.
- Navigate to the `container` folder and follow the instructions in its README.

### Running Additional Experiments

- **Experiment 2** reports runtime evaluations of the causal discovery graphs. These results are combined with those from Experiment 1, which contains the runtime measurements of the ML models.
- **Experiment 3** presents a study on the impact of degrading domain knowledge. The corresponding silver datasets used for this evaluation are located in `silver/dk_evaluation`.
- **Experiment 4** contains a sensitivity analysis examining variations in the alpha parameter of a causal discovery method.

5. **Results**
   - All experiment outputs are stored in the `data/gold` folder.
   - Each experiment is defined by a dataset and a training fraction. A fraction of **1** represents the full training set, while **0.8, 0.6, 0.4, and 0.2** correspond to progressively smaller subsets.
   - Primary evaluations are conducted using the full dataset (fraction 1). Reduced fractions are used to assess the impact of training size on performance.
   - For the **0.2 fraction**, it is necessary to set `training_02 = true` in the file `/container/main.py` so that the time-series models have sufficient training data.
   - Use the notebook **`005_evaluation_visual.ipynb`** to generate all visualizations and tables from the stored results.