import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, entropy
from typing import Dict, Any, List

def calculate_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) for a single column.
    """
    
    def scale_range(input, min_val, max_val):
        input += -(np.min(input))
        input /= np.max(input) / (max_val - min_val)
        input += min_val
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

    if expected.nunique() > buckets: # Continuous
        try:
            # Scale to handle different ranges if needed, but usually we bin based on expected distribution
            # Simple binning
            breakpoints = np.percentile(expected, breakpoints)
        except:
            return 0.0 # Fallback
    else: # Categorical/Discrete
        # Not handling categorical in this simple implementation for now, assuming numerical
        # Or using value counts
        pass

    # Simplified PSI for numerical data
    # Create bins based on expected data
    try:
        expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    except ValueError:
        # Fallback for unique value binning issues
        return 0.0

    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    psi_value = np.sum((expected_percents - actual_percents) * np.log(actual_percents / expected_percents))
    return float(psi_value)

def calculate_ks_test(reference_col: pd.Series, current_col: pd.Series) -> Dict[str, float]:
    """
    Kolmogorov-Smirnov Test
    """
    statistic, p_value = ks_2samp(reference_col, current_col)
    return {"statistic": float(statistic), "p_value": float(p_value)}

def calculate_kl_divergence(reference_col: pd.Series, current_col: pd.Series, buckets: int = 10) -> float:
    """
    KL Divergence
    """
    # Create histograms to get probability distributions
    # We need shared bins to make them comparable
    combined = np.concatenate([reference_col, current_col])
    # Check if data is constant
    if len(np.unique(combined)) < 2:
        return 0.0
        
    hist_range = (min(combined), max(combined))
    
    ref_hist, _ = np.histogram(reference_col, bins=buckets, range=hist_range, density=True)
    curr_hist, _ = np.histogram(current_col, bins=buckets, range=hist_range, density=True)
    
    # Avoid zero probability
    ref_hist = np.where(ref_hist == 0, 1e-10, ref_hist)
    curr_hist = np.where(curr_hist == 0, 1e-10, curr_hist)
    
    return float(entropy(ref_hist, curr_hist))

def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect drift between two dataframes column by column.
    """
    report = {}
    
    # Find common numerical columns
    ref_numerics = reference_df.select_dtypes(include=[np.number]).columns
    curr_numerics = current_df.select_dtypes(include=[np.number]).columns
    common_cols = list(set(ref_numerics) & set(curr_numerics))
    
    drift_summary = {
        "columns_analyzed": len(common_cols),
        "drifted_columns": 0,
        "details": {}
    }
    
    for col in common_cols:
        ref_data = reference_df[col].dropna()
        curr_data = current_df[col].dropna()
        
        if len(ref_data) == 0 or len(curr_data) == 0:
            continue
            
        psi = calculate_psi(ref_data, curr_data)
        ks = calculate_ks_test(ref_data, curr_data)
        kl = calculate_kl_divergence(ref_data, curr_data)
        
        # Drift logic
        is_drifted = False
        if psi > 0.25 or ks["p_value"] < 0.05:
             is_drifted = True
             drift_summary["drifted_columns"] += 1
             
        drift_summary["details"][col] = {
            "psi": psi,
            "ks_test": ks,
            "kl_divergence": kl,
            "drift_detected": is_drifted
        }
        
    return drift_summary
