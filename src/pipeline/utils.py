import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_test_not_normalized, y_pred_not_normalized, normalize_extra_predict=None, normalize=True):
    """Calculates various regression metrics for model evaluation.

    This function computes standard regression metrics such as MSE, MAE, R2, RMSE,
    MAPE, SMAPE, WAPE, and CVRMSE. Optionally, it normalizes the input data using
    Z-score standardization before computing the metrics, which can be useful for
    comparing performance across different scales.

    Args:
        y_test_not_normalized (array-like): The true target values (ground truth).
        y_pred_not_normalized (array-like): The predicted target values from the model.
        normalize_extra_predict (array-like, optional): Additional data to use for
            computing the normalization parameters (mean and std). If None, uses
            y_test_not_normalized. Defaults to None.
        normalize (bool, optional): Whether to apply Z-score normalization to the
            data before computing metrics. Defaults to True.

    Returns:
        dict: A dictionary containing the computed metrics with the following keys:
            - 'mse': Mean Squared Error.
            - 'mae': Mean Absolute Error.
            - 'r2': R-squared (coefficient of determination).
            - 'rmse': Root Mean Squared Error.
            - 'mape': Mean Absolute Percentage Error (as a percentage).
            - 'smape': Symmetric Mean Absolute Percentage Error (as a percentage).
            - 'wape': Weighted Absolute Percentage Error (as a percentage).
            - 'cvrmse': Coefficient of Variation of RMSE (as a percentage).
    """
    y_test, y_pred = y_test_not_normalized, y_pred_not_normalized

    if normalize:
        if normalize_extra_predict is not None:
            mean_y, std_y = normalize_extra_predict.mean(), normalize_extra_predict.std()
        else:
            mean_y, std_y = y_test.mean(), y_test.std()

        # --- Z-score standardization ---
        if std_y > 0:  # avoid division by zero
            y_test = (y_test - mean_y) / std_y
            y_pred = (y_pred - mean_y) / std_y
        else:
            # All values constant, degenerate case
            y_test = np.zeros_like(y_test)
            y_pred = np.zeros_like(y_pred)

    # --- Metrics definitions ---
    def mean_absolute_percentage_error(y_true, y_pred):
        nonzero = y_true != 0
        if np.any(nonzero):
            return np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        else:
            return np.nan

    def smape(y_true, y_pred):
        denominator = np.abs(y_true) + np.abs(y_pred)
        nonzero = denominator != 0
        if np.any(nonzero):
            return np.mean(200.0 * np.abs(y_pred[nonzero] - y_true[nonzero]) / denominator[nonzero])
        else:
            return np.nan

    def wape(y_true, y_pred):
        total_abs_actual = np.sum(np.abs(y_true))
        if total_abs_actual > 0:
            return (np.sum(np.abs(y_true - y_pred)) / total_abs_actual) * 100
        else:
            return np.nan

    def cvrmse(y_true, y_pred):
        """
        Calculate the Coefficient of Variation of RMSE (CVRMSE)
        
        Parameters:
        y_true : array-like
            Actual observed values
        y_pred : array-like
            Predicted values from the model
        
        Returns:
        float
            CVRMSE as a percentage
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mean_actual = np.mean(y_true)
        
        return (rmse / mean_actual) * 100


    # --- Metric computation ---
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    smape = smape(y_test, y_pred)
    wape = wape(y_test, y_pred)
    cvrmse = cvrmse(y_test, y_pred)

    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'rmse': rmse,
        'mape': mape,
        'smape': smape,
        'wape': wape,
        'cvrmse': cvrmse
    }
