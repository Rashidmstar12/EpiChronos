import pytest
import numpy as np
import polars as pl
from epichronos.core import MethylationDataset
from epichronos.pacemaker import EpigeneticPacemaker
from epichronos.clocks import calculate_biological_age

def test_epigenetic_pacemaker_convergence():
    # Construct a synthetic methylation matrix with an aging gradient across samples
    np.random.seed(42)
    
    n_sites = 20
    n_samples = 10
    
    # Chronological ages: linear range
    chronological_ages = {f"Sample_{j}": float(30 + j * 5) for j in range(n_samples)}
    t_true = np.array([chronological_ages[f"Sample_{j}"] for j in range(n_samples)])
    
    # Generate CpG site properties:
    # Some sites gain methylation, others lose methylation with age
    rates_true = np.random.uniform(-0.01, 0.01, size=n_sites)
    intercepts_true = np.random.uniform(0.2, 0.8, size=n_sites)
    
    # Generate beta values matrix S = r * t + b + noise
    S = rates_true[:, np.newaxis] * t_true[np.newaxis, :] + intercepts_true[:, np.newaxis]
    S += np.random.normal(0, 0.01, size=S.shape)
    S = np.clip(S, 0.0, 1.0)
    
    # Pack into Polars beta_df
    columns = ["chrom", "pos"] + [f"Sample_{j}" for j in range(n_samples)]
    rows = []
    for i in range(n_sites):
        row = {
            "chrom": "chr1",
            "pos": 1000 + i * 100
        }
        for j in range(n_samples):
            row[f"Sample_{j}"] = float(S[i, j])
        rows.append(row)
        
    beta_df = pl.DataFrame(rows)
    dataset = MethylationDataset(beta_df)
    
    # Fit the EPM model
    model = EpigeneticPacemaker(max_iter=50, tol=1e-6)
    model.fit(dataset, chronological_ages)
    
    # Verify convergence and model fitting
    assert len(model.loss_history_) > 0
    assert model.loss_history_[-1] < model.loss_history_[0]  # Error decreased
    assert len(model.rates_) == n_sites
    assert len(model.intercepts_) == n_sites
    
    # Predict ages on the same dataset
    pred_df = model.predict(dataset)
    assert pred_df.height == n_samples
    assert "biological_age" in pred_df.columns
    
    # Verify that the predicted Pacemaker biological ages correlate highly with chronological ages
    pred_ages = pred_df.sort("sample")["biological_age"].to_numpy()
    chron_ages = np.array([chronological_ages[s] for s in sorted(list(chronological_ages.keys()))])
    
    correlation = np.corrcoef(pred_ages, chron_ages)[0, 1]
    assert correlation >= 0.85  # Strong correlation since we generated linear age data


def test_calculate_biological_age_routing_pacemaker():
    # Test clocks.py integrated routing
    np.random.seed(42)
    n_samples = 6
    chronological_ages = {f"S_{j}": float(20 + j * 8) for j in range(n_samples)}
    
    beta_df = pl.DataFrame({
        "chrom": ["chr1", "chr2", "chr3"],
        "pos": [1000, 2000, 3000],
        "S_0": [0.10, 0.85, 0.40],
        "S_1": [0.15, 0.80, 0.45],
        "S_2": [0.20, 0.70, 0.50],
        "S_3": [0.25, 0.65, 0.55],
        "S_4": [0.30, 0.60, 0.60],
        "S_5": [0.35, 0.50, 0.65]
    })
    
    dataset = MethylationDataset(beta_df)
    
    clock_res = calculate_biological_age(
        dataset, 
        clock_name="pacemaker", 
        chronological_ages=chronological_ages
    )
    
    assert clock_res.height == n_samples
    assert "biological_age" in clock_res.columns
    assert "chronological_age" in clock_res.columns
    assert "age_acceleration" in clock_res.columns
    
    # Confirm dataset has EPM model reference stored
    assert hasattr(dataset, "_last_epm_model")
    assert dataset._last_epm_model is not None
    assert len(dataset._last_epm_model.loss_history_) > 0
