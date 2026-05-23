import pytest
import polars as pl
import numpy as np
from epichronos.core import MethylationDataset
from epichronos.stats import call_dmls, call_dmrs, fdr_correction

def test_fdr_correction():
    p_values = np.array([0.01, 0.04, 0.03, 0.80])
    q_values = fdr_correction(p_values)
    
    # Check that q-values are ordered monotonically with p-values
    assert q_values[0] <= q_values[2]
    assert q_values[2] <= q_values[1]
    assert q_values[1] <= q_values[3]
    
    # Maximum adjusted p-value should be capped at 1.0
    assert np.all(q_values <= 1.0)


def test_call_dmls():
    # Make synthetic dataset with clear differential methylation at pos 200
    beta_df = pl.DataFrame({
        "chrom": ["chr1", "chr1"],
        "pos": [100, 200],
        "ctrl1": [0.5, 0.1],
        "ctrl2": [0.6, 0.2],
        "treat1": [0.55, 0.8],
        "treat2": [0.49, 0.9]
    })
    
    ds = MethylationDataset(beta_df)
    res = call_dmls(ds, ["ctrl1", "ctrl2"], ["treat1", "treat2"])
    
    assert res.height == 2
    
    # pos 100 has very similar values (~0.5), so mean diff should be near 0
    diff_100 = res.filter(pl.col("pos") == 100)["mean_diff"][0]
    assert abs(diff_100) < 0.1
    
    # pos 200 is highly differential (Ctrl is ~0.15, Treat is ~0.85), so mean diff should be high (~0.70)
    diff_200 = res.filter(pl.col("pos") == 200)["mean_diff"][0]
    assert abs(diff_200 - 0.70) < 0.05
    
    # Welch's t-test p-value at pos 200 should be highly significant
    p_200 = res.filter(pl.col("pos") == 200)["p_value"][0]
    assert p_200 < 0.05


def test_call_dmrs():
    # Mock DML output dataframe with clustered active sites
    # Inject clustered sites on chr1
    dml_df = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr1", "chr1", "chr2"],
        "pos": [1000, 1500, 1800, 2200, 10000],  # chr1 sites are close, chr2 is isolated
        "mean_diff": [0.30, 0.40, 0.35, 0.28, 0.50],
        "p_value": [0.01, 0.005, 0.008, 0.02, 0.01]  # All are statistically active
    })
    
    # Call DMRs with max distance of 1000 bp, requiring at least 3 sites
    dmrs = call_dmrs(dml_df, p_cutoff=0.05, max_dist=1000, min_sites=3)
    
    # The 4 sites on chr1 are all within 1000 bp of each other (1000 to 1500=500, 1500 to 1800=300, 1800 to 2200=400)
    # They should form a single DMR
    # The site on chr2 is isolated and cannot meet the min_sites=3 constraint alone, so it is excluded
    assert dmrs.height == 1
    assert dmrs["chrom"][0] == "chr1"
    assert dmrs["start"][0] == 1000
    assert dmrs["end"][0] == 2200
    assert dmrs["num_sites"][0] == 4
