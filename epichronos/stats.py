import polars as pl
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Union, Optional
from epichronos.core import MethylationDataset

def fdr_correction(p_values: np.ndarray) -> np.ndarray:
    """
    Perform Benjamini-Hochberg (BH) False Discovery Rate (FDR) correction.
    
    Args:
        p_values: NumPy array of raw p-values.
    Returns:
        Adjusted p-values (q-values).
    """
    n = len(p_values)
    if n == 0:
        return p_values
        
    # Find indices that would sort the p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Calculate BH adjusted values: p_adj = p * n / rank
    ranks = np.arange(1, n + 1)
    adj_p = sorted_p * (n / ranks)
    
    # Ensure monotonic increasing: adj_p[i] = min(adj_p[i], adj_p[i+1])
    adj_p = np.minimum.accumulate(adj_p[::-1])[::-1]
    
    # Clip to maximum value of 1.0
    adj_p = np.clip(adj_p, 0.0, 1.0)
    
    # Restore original sorting order
    q_values = np.empty_like(p_values)
    q_values[sorted_indices] = adj_p
    return q_values


def call_dmls(
    dataset: MethylationDataset,
    group_a_samples: Optional[List[str]] = None,
    group_b_samples: Optional[List[str]] = None,
    method: str = "t-test",
    design_matrix: Optional[np.ndarray] = None,
    coef_index: int = 1,
    samples_order: Optional[List[str]] = None
) -> pl.DataFrame:
    """
    Perform high-speed, vectorized calling of Differentially Methylated Loci (DMLs).
    Supports Welch's t-test or multiple linear regression with covariates (design matrix).
    
    Args:
        dataset: Unified MethylationDataset.
        group_a_samples: Optional list of sample names in Group A (required for t-test).
        group_b_samples: Optional list of sample names in Group B (required for t-test).
        method: Statistical method ('t-test' for Welch's t-test, or 'regression' when using design_matrix).
        design_matrix: Optional N x P numpy array representing the regression design matrix.
        coef_index: The index of the design matrix column representing the group effect to test.
        samples_order: The list of sample names corresponding to the rows of design_matrix.
        
    Returns:
        Polars DataFrame containing DML statistics and FDR-adjusted q-values.
    """
    # 1. Regression-based DML calling (with design matrix)
    if design_matrix is not None or method == "regression":
        if design_matrix is None:
            raise ValueError("design_matrix must be provided when method is 'regression'.")
        if samples_order is None:
            raise ValueError("samples_order must be specified when using a design_matrix.")
            
        n_samples_reg = len(samples_order)
        assert design_matrix.shape[0] == n_samples_reg, "Number of rows in design_matrix must equal the number of samples in samples_order"
        
        # Check that samples exist
        for s in samples_order:
            assert s in dataset.samples, f"Sample '{s}' not found in dataset"
            
        # Get beta values matrix (M sites, N samples)
        S = dataset.beta_df.select(samples_order).to_numpy().astype(np.float64)
        
        # Cohort-mean imputation per site (regression cannot have NaNs)
        row_means = np.nanmean(S, axis=1)
        row_means = np.nan_to_num(row_means, nan=0.5)
        inds = np.where(np.isnan(S))
        S[inds] = row_means[inds[0]]
        
        # Design matrix X: shape (N, P)
        X = design_matrix.astype(np.float64)
        N, P = X.shape
        assert N > P, f"Number of samples ({N}) must be greater than number of covariates ({P}) for statistical calling."
        
        # Solve OLS: beta = (X^T X)^{-1} X^T Y^T where Y_T shape (N, M)
        Y_T = S.T
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)
            
        beta = XtX_inv @ X.T @ Y_T  # Shape: (P, M)
        
        # Residuals E = Y_T - X * beta (N, M)
        E = Y_T - (X @ beta)
        
        # Residual Sum of Squares (RSS) shape: (M,)
        RSS = np.sum(E**2, axis=0)
        
        # Residual variance (sigma_sq) shape: (M,)
        df = N - P
        sigma_sq = RSS / df
        
        # Standard error of coefficient coef_index
        se = np.sqrt(sigma_sq * XtX_inv[coef_index, coef_index])
        
        # Avoid division by zero
        eps = 1e-9
        se = np.maximum(se, eps)
        
        # t-statistic shape: (M,)
        t_stat = beta[coef_index, :] / se
        
        # Two-sided p-values
        p_values = 2 * stats.t.sf(np.abs(t_stat), df=df)
        p_values = np.nan_to_num(p_values, nan=1.0)
        
        # Multiple testing FDR correction
        q_values = fdr_correction(p_values)
        
        # Construct results DataFrame
        coords = dataset.beta_df.select(["chrom", "pos"])
        res_df = pl.DataFrame({
            "chrom": coords["chrom"],
            "pos": coords["pos"],
            "beta_coefficient": beta[coef_index, :],
            "t_stat": t_stat,
            "p_value": p_values,
            "q_value": q_values
        })
        
        return res_df.sort(["chrom", "pos"])
        
    # 2. Welch's t-test default DML calling (Group A vs Group B)
    if group_a_samples is None or group_b_samples is None:
        raise ValueError("Either group_a_samples and group_b_samples OR design_matrix must be provided.")
        
    # Check that samples exist
    for s in group_a_samples + group_b_samples:
        assert s in dataset.samples, f"Sample '{s}' not found in dataset"
        
    n_a = len(group_a_samples)
    n_b = len(group_b_samples)
    assert n_a >= 2 and n_b >= 2, "Each group must have at least 2 samples for statistical calling"
    
    # Exclude coordinates and convert only sample columns to numpy matrices for high-speed computation
    data_a = dataset.beta_df.select(group_a_samples).to_numpy()
    data_b = dataset.beta_df.select(group_b_samples).to_numpy()
    
    # Retrieve coordinates
    coords = dataset.beta_df.select(["chrom", "pos"])
    
    # Calculate row-wise means
    mean_a = np.nanmean(data_a, axis=1)
    mean_b = np.nanmean(data_b, axis=1)
    mean_diff = mean_b - mean_a
    
    # Calculate row-wise sample variances (ddof=1)
    var_a = np.nanvar(data_a, axis=1, ddof=1)
    var_b = np.nanvar(data_b, axis=1, ddof=1)
    
    # Welch's t-test (unequal variances, unequal sample sizes)
    # Handle zero-variance cases by adding a tiny epsilon
    eps = 1e-9
    var_a = np.maximum(var_a, eps)
    var_b = np.maximum(var_b, eps)
    
    # Standard Error of Difference
    se = np.sqrt(var_a / n_a + var_b / n_b)
    t_stat = mean_diff / se
    
    # Satterthwaite-Welch degrees of freedom
    numerator = (var_a / n_a + var_b / n_b) ** 2
    denominator = ((var_a / n_a) ** 2) / (n_a - 1) + ((var_b / n_b) ** 2) / (n_b - 1)
    dof = numerator / denominator
    
    # Calculate two-sided p-values
    p_values = 2 * stats.t.sf(np.abs(t_stat), df=dof)
    
    # Handle possible NaN values in statistics
    p_values = np.nan_to_num(p_values, nan=1.0)
    
    # Multiple testing FDR correction
    q_values = fdr_correction(p_values)
    
    # Construct results DataFrame
    res_df = pl.DataFrame({
        "chrom": coords["chrom"],
        "pos": coords["pos"],
        "mean_A": mean_a,
        "mean_B": mean_b,
        "mean_diff": mean_diff,
        "t_stat": t_stat,
        "p_value": p_values,
        "q_value": q_values
    })
    
    return res_df.sort(["chrom", "pos"])


def call_dmrs(
    dml_df: pl.DataFrame,
    p_cutoff: float = 0.05,
    max_dist: int = 1000,
    min_sites: int = 3
) -> pl.DataFrame:
    """
    Cluster neighboring DMLs into Differentially Methylated Regions (DMRs) in linear time.
    
    Args:
        dml_df: Polars DataFrame output from call_dmls.
        p_cutoff: Maximum p-value (or q-value) for a site to be considered active.
        max_dist: Maximum physical distance in base pairs between consecutive CpGs in a region.
        min_sites: Minimum number of active CpG sites required to define a DMR.
        
    Returns:
        Polars DataFrame of called DMRs.
    """
    # Filter for active sites first
    active_df = dml_df.filter(pl.col("p_value") <= p_cutoff).sort(["chrom", "pos"])
    
    if active_df.height == 0:
        # Return empty schema-compliant DataFrame
        return pl.DataFrame(schema={
            "chrom": pl.String,
            "start": pl.Int64,
            "end": pl.Int64,
            "num_sites": pl.Int64,
            "mean_diff": pl.Float64,
            "min_p_value": pl.Float64,
            "area": pl.Float64
        })
        
    # Parse coordinates to lists/arrays for sequential grouping (fast in memory)
    chroms = active_df["chrom"].to_list()
    positions = active_df["pos"].to_list()
    diffs = active_df["mean_diff"].to_list()
    p_vals = active_df["p_value"].to_list()
    
    dmrs = []
    
    # Sliding window cluster assignment
    current_dmr = {
        "chrom": chroms[0],
        "start": positions[0],
        "end": positions[0],
        "sites": [positions[0]],
        "diffs": [diffs[0]],
        "p_vals": [p_vals[0]]
    }
    
    for i in range(1, len(positions)):
        same_chrom = chroms[i] == current_dmr["chrom"]
        dist = positions[i] - current_dmr["end"]
        
        if same_chrom and dist <= max_dist:
            # Expand current DMR
            current_dmr["end"] = positions[i]
            current_dmr["sites"].append(positions[i])
            current_dmr["diffs"].append(diffs[i])
            current_dmr["p_vals"].append(p_vals[i])
        else:
            # Save current DMR if it meets constraints
            if len(current_dmr["sites"]) >= min_sites:
                dmrs.append(current_dmr)
            # Initialize a new DMR
            current_dmr = {
                "chrom": chroms[i],
                "start": positions[i],
                "end": positions[i],
                "sites": [positions[i]],
                "diffs": [diffs[i]],
                "p_vals": [p_vals[i]]
            }
            
    # Add final DMR
    if len(current_dmr["sites"]) >= min_sites:
        dmrs.append(current_dmr)
        
    # Compile called DMRs into a Polars DataFrame
    if len(dmrs) == 0:
        return pl.DataFrame(schema={
            "chrom": pl.String,
            "start": pl.Int64,
            "end": pl.Int64,
            "num_sites": pl.Int64,
            "mean_diff": pl.Float64,
            "min_p_value": pl.Float64,
            "area": pl.Float64
        })
        
    dmr_chroms = [d["chrom"] for d in dmrs]
    dmr_starts = [d["start"] for d in dmrs]
    dmr_ends = [d["end"] for d in dmrs]
    dmr_sites = [len(d["sites"]) for d in dmrs]
    dmr_diffs = [float(np.mean(d["diffs"])) for d in dmrs]
    dmr_min_p = [float(np.min(d["p_vals"])) for d in dmrs]
    
    # Area = length * average absolute difference (common metric in epigenetics)
    dmr_area = [float(len(d["sites"]) * np.mean(np.abs(d["diffs"]))) for d in dmrs]
    
    res_df = pl.DataFrame({
        "chrom": dmr_chroms,
        "start": dmr_starts,
        "end": dmr_ends,
        "num_sites": dmr_sites,
        "mean_diff": dmr_diffs,
        "min_p_value": dmr_min_p,
        "area": dmr_area
    })
    
    return res_df.sort(["chrom", "start"])
