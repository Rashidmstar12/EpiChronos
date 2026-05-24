import polars as pl
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Union, Optional
from epichronos.core import MethylationDataset

# Enforce pyliftover as a hard dependency for coordinate-level liftovers
try:
    from pyliftover import LiftOver
except ImportError as e:
    raise ImportError("EpiChronos requires the 'pyliftover' package. Please install it using: pip install pyliftover") from e

def _load_model(name: str) -> dict:
    """Helper to load model JSON from resources."""
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "resources", f"{name}_model.json")
    with open(model_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load the models dynamically
try:
    _horvath_data = _load_model("horvath")
    _hannum_data = _load_model("hannum")
except Exception as e:
    # Fail-safe defaults in case of any package environment loading issues
    _horvath_data = {"weights": {}, "reference_means": {}, "manifest": {}, "intercept": 0.0}
    _hannum_data = {"weights": {}, "reference_means": {}, "manifest": {}, "intercept": 0.0}

# 1. Coordinate mapping manifest between Array Probe ID and GRCh37 Coordinates
# Consolidate all coordinates dynamically from the compiled manifests!
CLOCK_MANIFEST = {}
for k, v in _horvath_data.get("manifest", {}).items():
    CLOCK_MANIFEST[k] = tuple(v)
for k, v in _hannum_data.get("manifest", {}).items():
    CLOCK_MANIFEST[k] = tuple(v)

# Inverse lookup: (chrom, pos) -> probe_id
_COORD_TO_PROBE = {coords: probe for probe, coords in CLOCK_MANIFEST.items()}


def horvath_inverse_calibrate(y: float) -> float:
    """
    Apply Horvath's inverse calibration function to calculate chronological equivalent age.
    F^-1(y) = exp(y) - 1 if y <= log(21) else 21 * (y - log(21)) + 20
    """
    threshold = np.log(21)
    if y <= threshold:
        return float(np.exp(y) - 1.0)
    else:
        return float(21.0 * (y - threshold) + 20.0)


def list_available_clocks() -> List[str]:
    """Returns a list of pre-coded biological clocks."""
    return ["horvath", "hannum", "pacemaker"]


def calculate_biological_age(
    dataset: MethylationDataset,
    clock_name: str = "horvath",
    chronological_ages: Optional[Dict[str, float]] = None
) -> pl.DataFrame:
    """
    Calculate biological age for all samples in a MethylationDataset.
    Automatically maps sequencing coordinates to clock array probes.
    
    Args:
        dataset: The target MethylationDataset.
        clock_name: Name of the aging clock ('horvath', 'hannum', or 'pacemaker').
        chronological_ages: Optional dict mapping sample names to their true chronological ages.
                            If provided, calculates Epigenetic Age Acceleration (residuals).
                            For 'pacemaker', chronological_ages is required for on-the-fly training.
                            
    Returns:
        Polars DataFrame containing Sample ID, Predicted Biological Age,
        and optionally Age Acceleration.
    """
    clock_name = clock_name.lower()
    if clock_name == "pacemaker":
        if not chronological_ages:
            raise ValueError("Epigenetic Pacemaker requires chronological_ages to train on-the-fly.")
        from epichronos.pacemaker import EpigeneticPacemaker
        model = EpigeneticPacemaker()
        model.fit(dataset, chronological_ages)
        res_df = model.predict(dataset)
        
        # Save model reference on dataset for visualization metrics if needed
        dataset._last_epm_model = model
        
        samples_list = res_df["sample"].to_list()
        bio_ages = res_df["biological_age"].to_list()
        
        chron_ages_list = [chronological_ages.get(s) for s in samples_list]
        accel_list = [bio_ages[idx] - chron_ages_list[idx] if chron_ages_list[idx] is not None else None for idx in range(len(samples_list))]
        
        out_dict = {
            "sample": samples_list,
            "biological_age": bio_ages,
            "chronological_age": chron_ages_list,
            "age_acceleration": accel_list
        }
        return pl.DataFrame(out_dict).sort("sample")
        
    if clock_name == "horvath":
        clock = _horvath_data
    elif clock_name == "hannum":
        clock = _hannum_data
    else:
        raise ValueError(f"Clock '{clock_name}' is not recognized. Choose from: {list_available_clocks()}")
        
    intercept = clock["intercept"]
    weights = clock["weights"]
    ref_means = clock["reference_means"]
    
    # 1. Map dataset to probe IDs to extract relevant clock sites
    sample_names = dataset.samples
    probe_values = {probe: {s: None for s in sample_names} for probe in weights}
    
    # Identify representation format in input dataset
    # We will search the dataset.beta_df for matching coordinates
    df = dataset.beta_df
    
    # High-performance lookup mapping of coordinates in the dataset
    coord_map = {}
    for idx, (c, p) in enumerate(zip(df["chrom"].to_list(), df["pos"].to_list())):
        coord_map[(c, p)] = idx
        
    # Check if input coordinates appear to be GRCh38/hg38 instead of GRCh37/hg19
    grch37_matches = 0
    total_checks = 0
    for probe in weights:
        if probe in CLOCK_MANIFEST:
            chrom, pos = CLOCK_MANIFEST[probe]
            total_checks += 1
            if (chrom, pos) in coord_map or (chrom, pos + 1) in coord_map or (chrom, pos - 1) in coord_map:
                grch37_matches += 1
                
    is_hg38 = False
    lo_19_to_38 = None
    
    # If no matches in GRCh37/hg19, check if they align with the hg38 coordinates using pyliftover
    if grch37_matches == 0 and total_checks > 0:
        try:
            lo_19_to_38 = LiftOver('hg19', 'hg38')
            # Check if lifted positions of some manifest probes match coordinates in df
            lift_matches = 0
            for probe in list(weights.keys())[:20]:
                if probe in CLOCK_MANIFEST:
                    chrom, pos = CLOCK_MANIFEST[probe]
                    lifted = lo_19_to_38.convert_coordinate(chrom, pos)
                    if lifted:
                        lc, lp = lifted[0][0], lifted[0][1]
                        if (lc, lp) in coord_map or (lc, lp + 1) in coord_map or (lc, lp - 1) in coord_map:
                            lift_matches += 1
            if lift_matches > 0:
                is_hg38 = True
        except Exception:
            pass
            
    # Initialize LiftOver if detected hg38
    if is_hg38:
        print("Assembly Detection: Input dataset coordinates appear to be GRCh38/hg38. Initializing Liftover.")
        if lo_19_to_38 is None:
            try:
                lo_19_to_38 = LiftOver('hg19', 'hg38')
            except Exception as e:
                raise RuntimeError(
                    "Input dataset coordinates appear to be GRCh38/hg38, but the pyliftover hg19->hg38 chain file "
                    "could not be loaded. To avoid silent calculation errors, please check your network connection "
                    "or supply a GRCh37 coordinate-aligned dataset."
                ) from e
                
        if lo_19_to_38 is None:
            raise RuntimeError(
                "Input dataset coordinates appear to be GRCh38/hg38, but pyliftover is unavailable or failed to initialize."
            )

    for probe in weights:
        if probe in CLOCK_MANIFEST:
            chrom, pos = CLOCK_MANIFEST[probe]
            
            # Dynamically lift over hg19 coordinate to hg38 if needed
            if is_hg38:
                lifted = lo_19_to_38.convert_coordinate(chrom, pos)
                if lifted:
                    chrom, pos = lifted[0][0], lifted[0][1]
                else:
                    raise ValueError(
                        f"Failed to lift over Clock CpG '{probe}' coordinate ({chrom}:{pos}) to GRCh38. "
                        "To prevent incorrect biological age calculations, execution has been stopped."
                    )
            
            # Find the row index using the Watson/Crick strand-aware jitter tolerance (exact, +1 bp, -1 bp)
            row_idx = None
            if (chrom, pos) in coord_map:
                row_idx = coord_map[(chrom, pos)]
            elif (chrom, pos + 1) in coord_map:
                row_idx = coord_map[(chrom, pos + 1)]
            elif (chrom, pos - 1) in coord_map:
                row_idx = coord_map[(chrom, pos - 1)]
                
            if row_idx is not None:
                for sample in sample_names:
                    val = df[sample][row_idx]
                    if val is not None and not np.isnan(val):
                        probe_values[probe][sample] = float(val)
                        
    # 2. Perform advanced cohort-mean and reference-mean imputation for missing sites
    for probe in weights:
        for sample in sample_names:
            if probe_values[probe][sample] is None:
                # Cohort-mean imputation (average of all non-null samples in this batch)
                valid_vals = [probe_values[probe][s] for s in sample_names if probe_values[probe][s] is not None]
                if len(valid_vals) > 0:
                    probe_values[probe][sample] = float(np.mean(valid_vals))
                else:
                    # Cohort-wide missing: fallback to standard reference mean
                    probe_values[probe][sample] = ref_means[probe]
                    
    # 3. Calculate predicted age using coefficients
    predicted_ages = {}
    
    for sample in sample_names:
        # Linear sum = intercept + sum(weight * beta_value)
        linear_sum = intercept
        for probe, weight in weights.items():
            linear_sum += weight * probe_values[probe][sample]
            
        # Calibrate output based on clock type
        if clock_name == "horvath":
            predicted_ages[sample] = horvath_inverse_calibrate(linear_sum)
        else:
            # Hannum is a simple linear output
            predicted_ages[sample] = float(linear_sum)
            
    # Compile results
    samples_list = []
    bio_ages_list = []
    chron_ages_list = []
    accel_list = []
    
    for sample in sample_names:
        samples_list.append(sample)
        bio_ages_list.append(predicted_ages[sample])
        
        if chronological_ages and sample in chronological_ages:
            chron_age = chronological_ages[sample]
            chron_ages_list.append(chron_age)
            # Age Acceleration = Biological Age - Chronological Age (simple linear residual)
            accel_list.append(predicted_ages[sample] - chron_age)
        else:
            chron_ages_list.append(None)
            accel_list.append(None)
            
    # Build output DataFrame
    out_dict = {
        "sample": samples_list,
        "biological_age": bio_ages_list
    }
    
    if chronological_ages:
        out_dict["chronological_age"] = chron_ages_list
        out_dict["age_acceleration"] = accel_list
        
    return pl.DataFrame(out_dict).sort("sample")


def calculate_intrinsic_age_acceleration(
    age_df: pl.DataFrame,
    cell_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Calculate cellular-intrinsic Epigenetic Age Acceleration (IEAA) by regressing
    predicted biological age on chronological age AND cell-type proportions.
    
    This adjusts for shifts in blood cell-type composition, leaving a residual that
    represents the intrinsic intracellular aging rate.
    
    Args:
        age_df: DataFrame containing columns ['sample', 'biological_age', 'chronological_age'].
        cell_df: DataFrame containing cell proportions (e.g. from decon.estimate_cell_proportions).
        
    Returns:
        Polars DataFrame containing 'sample' and 'intrinsic_age_acceleration' (residuals).
    """
    # 1. Join age and cell proportion data
    joined = age_df.join(cell_df, on="sample", how="inner")
    
    # Filter out samples with missing chronological age or biological age
    joined = joined.filter(
        pl.col("chronological_age").is_not_null() & 
        pl.col("biological_age").is_not_null()
    )
    
    if joined.height < 3:
        raise ValueError(
            f"Insufficient samples ({joined.height}) to perform multi-variable linear regression. "
            "At least 3 samples with chronological age are required."
        )
        
    # Extract cell-type columns dynamically (excluding sample, biological_age, chronological_age, and age_acceleration if present)
    exclude_cols = {"sample", "biological_age", "chronological_age", "age_acceleration"}
    cell_cols = [c for c in cell_df.columns if c not in exclude_cols]
    
    # Multicollinearity check (dummy variable trap):
    # Since cell proportions sum to exactly 1.0, including all cell types plus an intercept
    # causes perfect multicollinearity (singular covariance matrix).
    # To resolve this, we omit the first cell type (reference cell type) from the regression.
    reg_cell_cols = cell_cols[1:]
    
    # 2. Construct regression feature matrix X and target vector y
    # y = biological_age
    y = joined["biological_age"].to_numpy()
    
    # X = [intercept, chronological_age, cell_proportion_1, cell_proportion_2, ...]
    n_samples = joined.height
    intercept = np.ones((n_samples, 1))
    chron_age = joined["chronological_age"].to_numpy().reshape(-1, 1)
    
    cell_matrix = joined.select(reg_cell_cols).to_numpy()
    
    X = np.hstack([intercept, chron_age, cell_matrix])
    
    # 3. Fit multi-variable linear regression using least-squares (lstsq)
    # y ~ X * beta
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    
    # 4. Calculate predicted values and residuals (IEAA)
    y_pred = np.dot(X, beta)
    ieaa = y - y_pred
    
    # Create output DataFrame
    out_df = pl.DataFrame({
        "sample": joined["sample"],
        "intrinsic_age_acceleration": ieaa.tolist()
    }).sort("sample")
    
    return out_df

