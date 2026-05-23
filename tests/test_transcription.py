import pytest
import polars as pl
import numpy as np
from epichronos.core import MethylationDataset
from epichronos.transcription import integrate_expression_data

def test_transcription_integration():
    # Make a mock dataset with a called DMR inside chr2:460000-465000 (NFKB1 vicinity)
    samples = ["Ctrl_1", "Ctrl_2", "Ctrl_3", "Treat_1", "Treat_2", "Treat_3"]
    
    # 3 CpGs inside the DMR
    beta_df = pl.DataFrame({
        "chrom": ["chr2", "chr2", "chr2"],
        "pos": [461000, 462000, 463000],
        "Ctrl_1": [0.8, 0.9, 0.85],
        "Ctrl_2": [0.75, 0.8, 0.85],
        "Ctrl_3": [0.85, 0.85, 0.9],
        "Treat_1": [0.2, 0.15, 0.25],
        "Treat_2": [0.1, 0.2, 0.15],
        "Treat_3": [0.25, 0.1, 0.2]
    })
    
    ds = MethylationDataset(beta_df)
    
    # Define a mock called DMR
    dmr_df = pl.DataFrame({
        "chrom": ["chr2"],
        "start": [460500],
        "end": [464500],
        "num_sites": [3],
        "mean_diff": [-0.65],
        "area": [-2.6]
    })
    
    # Define a matching RNA-seq expression dataframe (strongly anti-correlated with methylation)
    # High methylation in Ctrl -> Low expression (e.g., ~15)
    # Low methylation in Treat -> High expression (e.g., ~85)
    expression_df = pl.DataFrame({
        "gene": ["NFKB1"],
        "Ctrl_1": [12.0],
        "Ctrl_2": [18.0],
        "Ctrl_3": [10.0],
        "Treat_1": [78.0],
        "Treat_2": [85.0],
        "Treat_3": [90.0]
    })
    
    # Run meQTL Transcription Integration
    meqtl_df = integrate_expression_data(ds, dmr_df, expression_df, max_dist_bp=100000)
    
    # Validate structure and values
    assert meqtl_df.height == 1
    assert meqtl_df["chrom"][0] == "chr2"
    assert meqtl_df["gene"][0] == "NFKB1"
    
    # Pearson correlation r must be highly negative
    r_val = meqtl_df["correlation_r"][0]
    assert r_val <= -0.8
    
    # Functional status must be classified as "Transcriptional Silencing"
    assert meqtl_df["functional_status"][0] == "Transcriptional Silencing"


def test_insufficient_samples_raises_error():
    # If matching samples are < 3, raise ValueError
    samples = ["s1", "s2"]
    beta_df = pl.DataFrame({
        "chrom": ["chr2"],
        "pos": [461000],
        "s1": [0.8],
        "s2": [0.2]
    })
    ds = MethylationDataset(beta_df)
    
    dmr_df = pl.DataFrame({
        "chrom": ["chr2"],
        "start": [460500],
        "end": [464500]
    })
    
    expression_df = pl.DataFrame({
        "gene": ["NFKB1"],
        "s1": [10.0],
        "s2": [90.0]
    })
    
    with pytest.raises(ValueError):
        integrate_expression_data(ds, dmr_df, expression_df)
