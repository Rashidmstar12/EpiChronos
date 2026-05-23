import pytest
import polars as pl
import numpy as np
from epichronos.core import MethylationDataset

def test_dataset_creation():
    # Setup mock dataframes
    beta_df = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr2"],
        "pos": [100, 200, 100],
        "sample1": [0.1, 0.8, 0.5],
        "sample2": [0.2, 0.7, np.nan]
    })
    
    cov_df = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr2"],
        "pos": [100, 200, 100],
        "sample1": [30, 25, 40],
        "sample2": [12, 18, 5]
    })
    
    metadata = {"sample1": "Young", "sample2": "Old"}
    
    ds = MethylationDataset(beta_df, cov_df, metadata)
    
    assert ds.shape == (3, 4)
    assert len(ds.samples) == 2
    assert "sample1" in ds.samples
    assert ds.metadata["sample1"] == "Young"


def test_coverage_filtering():
    beta_df = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr2"],
        "pos": [100, 200, 100],
        "sample1": [0.1, 0.8, 0.5],
        "sample2": [0.2, 0.7, 0.4]
    })
    
    cov_df = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr2"],
        "pos": [100, 200, 100],
        "sample1": [30, 2, 40],  # pos 200 is low cov
        "sample2": [12, 1, 5]   # pos 200 is low cov
    })
    
    ds = MethylationDataset(beta_df, cov_df)
    filtered = ds.filter_by_coverage(min_cov=5)
    
    # Position 200 has cov < 5 in both samples, so it should be filtered out
    assert filtered.shape[0] == 2
    assert 200 not in filtered.beta_df["pos"].to_list()


def test_variance_filtering():
    beta_df = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr2"],
        "pos": [100, 200, 100],
        "sample1": [0.5, 0.8, 0.5],
        "sample2": [0.5, 0.79, 0.1]
    })
    
    ds = MethylationDataset(beta_df)
    filtered = ds.filter_by_variance(min_var=0.005)
    
    # pos 100 has 0 variance (0.5 and 0.5)
    # pos 200 has tiny variance (0.8 and 0.79)
    # pos 100 on chr2 has high variance (0.5 and 0.1)
    # So only the last site should remain
    assert filtered.shape[0] == 1
    assert filtered.beta_df["pos"].to_list() == [100]
    assert filtered.beta_df["chrom"].to_list() == ["chr2"]


def test_imputation():
    beta_df = pl.DataFrame({
        "chrom": ["chr1", "chr1"],
        "pos": [100, 200],
        "sample1": [0.1, np.nan],
        "sample2": [0.2, 0.8]
    })
    
    ds = MethylationDataset(beta_df)
    imputed = ds.impute_missing(method="mean")
    
    # NaN in sample1 pos 200 should be imputed with the mean of sample1 (which is 0.1)
    val = imputed.beta_df.filter((pl.col("chrom") == "chr1") & (pl.col("pos") == 200))["sample1"][0]
    assert abs(val - 0.1) < 1e-9
