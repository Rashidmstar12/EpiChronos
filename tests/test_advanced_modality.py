import pytest
import numpy as np
import polars as pl
from epichronos.core import MethylationDataset
import epichronos.clocks as clocks
import epichronos.decon as decon

def test_strand_coordinates_collapsing():
    # 1. Watson (x) and Crick (x+1) coordinates on same chromosome
    beta_df = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr1", "chr2"],
        "pos": [100, 101, 200, 500],
        "S1": [0.2, 0.5, 0.8, 0.1],
        "S2": [0.3, 0.9, 0.4, 0.7]
    })
    cov_df = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr1", "chr2"],
        "pos": [100, 101, 200, 500],
        "S1": [10, 20, 30, 40],
        "S2": [15, 5, 25, 35]
    })
    
    dataset = MethylationDataset(beta_df, cov_df)
    
    # Run collapsing
    collapsed = dataset.collapse_strand_coordinates()
    
    # Assert Watson (100) and Crick (101) are collapsed into Watson (100)
    # Remaining coordinates: 100, 200, 500
    assert collapsed.shape[0] == 3
    
    # Position checking
    assert collapsed.beta_df["chrom"].to_list() == ["chr1", "chr1", "chr2"]
    assert collapsed.beta_df["pos"].to_list() == [100, 200, 500]
    
    # Weighted average math verification:
    # S1 at 100: (0.2 * 10 + 0.5 * 20) / (10 + 20) = (2 + 10) / 30 = 0.4
    # S2 at 100: (0.3 * 15 + 0.9 * 5) / (15 + 5) = (4.5 + 4.5) / 20 = 0.45
    assert pytest.approx(collapsed.beta_df.filter(pl.col("pos") == 100)["S1"][0]) == 0.4
    assert pytest.approx(collapsed.beta_df.filter(pl.col("pos") == 100)["S2"][0]) == 0.45
    
    # Total coverage depth checking
    assert collapsed.cov_df.filter(pl.col("pos") == 100)["S1"][0] == 30
    assert collapsed.cov_df.filter(pl.col("pos") == 100)["S2"][0] == 20
    
    # Non-collapsible sites should remain unchanged
    assert collapsed.beta_df.filter(pl.col("pos") == 200)["S1"][0] == 0.8
    assert collapsed.cov_df.filter(pl.col("pos") == 500)["S2"][0] == 35


def test_wilson_score_confidence_filtering():
    # 2. Vectorized 95% Wilson Score Interval filtering
    beta_df = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr1"],
        "pos": [100, 200, 300],
        "S1": [0.5, 0.5, 0.5]
    })
    # High coverage (pos 100) -> Narrow CI width
    # Low coverage (pos 200) -> Extremely wide CI width
    # Zero coverage (pos 300) -> Max CI width
    cov_df = pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr1"],
        "pos": [100, 200, 300],
        "S1": [1000, 3, 0]
    })
    
    dataset = MethylationDataset(beta_df, cov_df)
    
    # Run with a narrow threshold that only keeps high-confidence sites
    filtered = dataset.filter_by_binomial_confidence(min_ci_width=0.20, min_samples_ratio=1.0)
    
    # Should only keep chr1:100 (high-confidence)
    assert filtered.shape[0] == 1
    assert filtered.beta_df["pos"][0] == 100


def test_jitter_tolerant_clock_mapping():
    # Backup original state
    orig_manifest = clocks.CLOCK_MANIFEST.copy()
    orig_horvath = clocks._horvath_data
    
    try:
        # Setup mock CLOCK_MANIFEST coordinates
        clocks.CLOCK_MANIFEST = orig_manifest.copy()
        clocks.CLOCK_MANIFEST["cg_mock_1"] = ("chr1", 100)
        clocks.CLOCK_MANIFEST["cg_mock_2"] = ("chr2", 204)
        
        # Setup mock clock coefficients
        clocks._horvath_data = {
            "weights": {"cg_mock_1": 2.0, "cg_mock_2": -1.5},
            "reference_means": {"cg_mock_1": 0.5, "cg_mock_2": 0.5},
            "intercept": 10.0
        }
        
        beta_df = pl.DataFrame({
            "chrom": ["chr1", "chr2"],
            "pos": [101, 203],  # Crick shift: cg1 is mapped at 100, cg2 mapped at 204
            "S1": [0.45, 0.85]
        })
        dataset = MethylationDataset(beta_df)
        
        # Calculate predicted age
        clock_res = clocks.calculate_biological_age(dataset, clock_name="horvath")
        
        assert clock_res.height == 1
        assert "biological_age" in clock_res.columns
        
        expected_linear = 10.0 + 2.0 * 0.45 - 1.5 * 0.85
        expected_age = clocks.horvath_inverse_calibrate(expected_linear)
        assert pytest.approx(clock_res["biological_age"][0]) == expected_age
        
    finally:
        # Restore original state
        clocks.CLOCK_MANIFEST = orig_manifest
        clocks._horvath_data = orig_horvath


def test_jitter_tolerant_deconvolution_mapping():
    # Backup original state
    orig_manifest = decon.DECON_MANIFEST.copy()
    orig_decon = decon._decon_data
    
    try:
        # Setup mock deconvolution model
        decon.DECON_MANIFEST = orig_manifest.copy()
        decon.DECON_MANIFEST["cg_decon_1"] = ("chr3", 500)
        
        decon._decon_data = {
            "features": ["cg_decon_1"],
            "pseudo_inv": [[1.0]],  # 1 cell type, 1 feature
            "cell_types": ["Lymphocytes"],
            "manifest": {"cg_decon_1": ["chr3", 500]}
        }
        
        beta_df = pl.DataFrame({
            "chrom": ["chr3"],
            "pos": [501],  # Shifted from expected 500
            "S1": [0.65]
        })
        dataset = MethylationDataset(beta_df)
        
        # Run deconvolution estimation
        decon_res = decon.estimate_cell_proportions(dataset)
        
        assert decon_res.height == 1
        assert "Lymphocytes" in decon_res.columns
        assert decon_res["Lymphocytes"][0] == 1.0
        
    finally:
        # Restore original state
        decon.DECON_MANIFEST = orig_manifest
        decon._decon_data = orig_decon
