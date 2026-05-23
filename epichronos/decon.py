import polars as pl
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Union, Optional
from epichronos.core import MethylationDataset

def _load_decon_model() -> dict:
    """Helper to load deconvolution model JSON from resources."""
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "resources", "blood_decon_model.json")
    with open(model_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load the model dynamically
try:
    _decon_data = _load_decon_model()
except Exception as e:
    # Fail-safe empty defaults in case of loading issues
    _decon_data = {"features": [], "pseudo_inv": [], "cell_types": [], "manifest": {}}

DECON_MANIFEST = {k: tuple(v) for k, v in _decon_data.get("manifest", {}).items()}


def project_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project a vector v onto the probability simplex (non-negative, sum to 1.0).
    Using an efficient pure NumPy translation of the projection algorithm.
    
    Args:
        v: A 1D NumPy array representing raw cell-type proportions.
        
    Returns:
        A projected 1D NumPy array of the same shape.
    """
    sorted_v = np.sort(v)[::-1]
    cssv = np.cumsum(sorted_v)
    inds = np.arange(1, len(v) + 1)
    cond = sorted_v - (cssv - 1.0) / inds > 0
    rho = np.sum(cond) - 1
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    return w


def estimate_cell_proportions(
    dataset: MethylationDataset,
    impute_missing_val: float = 0.5
) -> pl.DataFrame:
    """
    Estimate the proportions of 6 major immune cell types (Neutrophils, NK, B-cell,
    CD4+ T, CD8+ T, Monocytes) in blood samples.
    
    Args:
        dataset: The target MethylationDataset containing blood sample beta values.
        impute_missing_val: Default beta value placeholder if a site is missing and cohort average is unavailable.
        
    Returns:
        Polars DataFrame containing Sample ID and estimated proportions for each cell type.
    """
    # Dynamic reload to ensure we capture the fetched coordinates manifest
    global _decon_data, DECON_MANIFEST
    if not DECON_MANIFEST or len(DECON_MANIFEST) < 10:
        try:
            _decon_data = _load_decon_model()
            DECON_MANIFEST = {k: tuple(v) for k, v in _decon_data.get("manifest", {}).items()}
        except Exception:
            pass

    features = _decon_data.get("features", [])
    pseudo_inv = np.array(_decon_data.get("pseudo_inv", [])) # Shape: (6, 600)
    cell_types = _decon_data.get("cell_types", ["Neutrophils", "NK", "Bcell", "CD4T", "CD8T", "Monocytes"])
    
    if not features or pseudo_inv.size == 0:
        raise ValueError("Deconvolution model parameters are empty or failed to load.")
        
    sample_names = dataset.samples
    n_samples = len(sample_names)
    n_features = len(features)
    
    # 1. Align input dataset coordinates with the deconvolution panel
    df = dataset.beta_df
    
    # Pre-allocate a matrix of beta values (600 features x n_samples)
    # Defaulting to None to allow cohort-mean imputation later
    beta_matrix = np.full((n_features, n_samples), None, dtype=object)
    
    # High-performance lookup mapping of coordinates in the dataset
    coord_map = {}
    for idx, (c, p) in enumerate(zip(df["chrom"].to_list(), df["pos"].to_list())):
        coord_map[(c, p)] = idx
        
    # Map probe coordinates
    for i, probe in enumerate(features):
        if probe in DECON_MANIFEST:
            chrom, pos = DECON_MANIFEST[probe]
            
            # Find the row index using the Watson/Crick strand-aware jitter tolerance (exact, +1 bp, -1 bp)
            row_idx = None
            if (chrom, pos) in coord_map:
                row_idx = coord_map[(chrom, pos)]
            elif (chrom, pos + 1) in coord_map:
                row_idx = coord_map[(chrom, pos + 1)]
            elif (chrom, pos - 1) in coord_map:
                row_idx = coord_map[(chrom, pos - 1)]
                
            if row_idx is not None:
                for j, sample in enumerate(sample_names):
                    val = df[sample][row_idx]
                    if val is not None and not np.isnan(val):
                        beta_matrix[i, j] = float(val)
                        
    # 2. Impute missing coordinates
    for i in range(n_features):
        # Calculate cohort average for this CpG site
        valid_vals = [beta_matrix[i, j] for j in range(n_samples) if beta_matrix[i, j] is not None]
        cohort_mean = float(np.mean(valid_vals)) if len(valid_vals) > 0 else impute_missing_val
        
        for j in range(n_samples):
            if beta_matrix[i, j] is None:
                beta_matrix[i, j] = cohort_mean
                
    # Convert beta matrix to a float array
    beta_array = beta_matrix.astype(np.float64) # Shape: (600, n_samples)
    
    # 3. Compute raw proportions
    # pseudo_inv shape: (6, 600), beta_array shape: (600, n_samples)
    # raw_props shape: (6, n_samples)
    raw_props = np.dot(pseudo_inv, beta_array)
    
    # 4. Project each sample onto the probability simplex
    final_props = np.zeros_like(raw_props)
    for j in range(n_samples):
        final_props[:, j] = project_simplex(raw_props[:, j])
        
    # 5. Build output DataFrame
    out_dict = {"sample": sample_names}
    for idx, cell_name in enumerate(cell_types):
        out_dict[cell_name] = final_props[idx, :].tolist()
        
    return pl.DataFrame(out_dict).sort("sample")
