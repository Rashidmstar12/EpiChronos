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
    group_a_samples: List[str],
    group_b_samples: List[str],
    method: str = "t-test"
) -> pl.DataFrame:
    """
    Perform high-speed, vectorized calling of Differentially Methylated Loci (DMLs).
    Compares methylation beta values between Group A and Group B.
    
    Args:
        dataset: Unified MethylationDataset.
        group_a_samples: List of sample names in Group A.
        group_b_samples: List of sample names in Group B.
        method: Statistical method ('t-test' for Welch's t-test).
        
    Returns:
        Polars DataFrame containing DML statistics and FDR-adjusted q-values.
    """
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
