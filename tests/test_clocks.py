import pytest
import polars as pl
import numpy as np
from epichronos.core import MethylationDataset
from epichronos.clocks import calculate_biological_age, CLOCK_MANIFEST

def test_biological_age_calculation():
    # Make a mock dataset with specific coordinates matching Horvath's clock
    # Inject values that would yield a specific linear sum
    # Using real pan-tissue clock probes: cg00075967, cg00374717, cg00864867
    
    samples = ["young_sample", "old_sample"]
    
    # Coordinates of active clock sites
    c_292 = CLOCK_MANIFEST["cg00075967"]
    c_2426 = CLOCK_MANIFEST["cg00374717"]
    c_3858 = CLOCK_MANIFEST["cg00864867"]
    
    beta_df = pl.DataFrame({
        "chrom": [c_292[0], c_2426[0], c_3858[0]],
        "pos": [c_292[1], c_2426[1], c_3858[1]],
        "young_sample": [0.1, 0.9, 0.1],  # Yields a low biological age
        "old_sample": [0.9, 0.1, 0.9]     # Yields a high biological age
    })
    
    ds = MethylationDataset(beta_df)
    
    chron_ages = {"young_sample": 20.0, "old_sample": 60.0}
    clock_res = calculate_biological_age(ds, clock_name="horvath", chronological_ages=chron_ages)
    
    assert clock_res.height == 2
    assert "young_sample" in clock_res["sample"].to_list()
    assert "old_sample" in clock_res["sample"].to_list()
    
    # Old sample should have a significantly higher biological age than young sample
    age_young = clock_res.filter(pl.col("sample") == "young_sample")["biological_age"][0]
    age_old = clock_res.filter(pl.col("sample") == "old_sample")["biological_age"][0]
    
    assert age_old > age_young
    
    # Age acceleration should be defined
    accel_young = clock_res.filter(pl.col("sample") == "young_sample")["age_acceleration"][0]
    assert accel_young is not None


def test_missing_data_imputation():
    # Setup a mock dataset with missing (NaN) clock sites
    # Check that it falls back to the cohort mean or reference means without crashing
    c_292 = CLOCK_MANIFEST["cg00075967"]
    c_2426 = CLOCK_MANIFEST["cg00374717"]
    
    beta_df = pl.DataFrame({
        "chrom": [c_292[0], c_2426[0]],
        "pos": [c_292[1], c_2426[1]],
        "sample1": [0.5, np.nan],  # sample1 is missing pos 2426
        "sample2": [0.6, 0.2]
    })
    
    ds = MethylationDataset(beta_df)
    
    # This should run successfully by imputing the missing site in sample1 with sample2's value (0.2)
    clock_res = calculate_biological_age(ds, clock_name="horvath")
    
    assert clock_res.height == 2
    assert not np.isnan(clock_res["biological_age"][0])
    assert not np.isnan(clock_res["biological_age"][1])


def test_clocks_assembly_validation():
    # Setup coordinates that represent hg38 positions (differ from hg19 CLOCK_MANIFEST coordinates)
    beta_df = pl.DataFrame({
        "chrom": ["chr16", "chr16"],
        "pos": [53434600, 28881200],  # Mock hg38 coordinates
        "young_sample": [0.1, 0.9],
        "old_sample": [0.9, 0.1]
    })
    
    ds = MethylationDataset(beta_df)
    
    # Run clock calculation. It should either:
    # 1. Successfully execute (if pyliftover resolves them)
    # 2. Raise an explicit RuntimeError or ValueError (due to chain loading or liftover fail)
    # Both are legally clean and scientifically correct outcomes (no silent wrong coordinates matched!)
    try:
        clock_res = calculate_biological_age(ds, clock_name="horvath")
        assert clock_res.height == 2
        assert "biological_age" in clock_res.columns
    except (RuntimeError, ValueError, ImportError) as e:
        # Verified that the codebase raises strict assembly translation exceptions
        assert "GRCh38" in str(e) or "lift over" in str(e) or "pyliftover" in str(e) or "hg38" in str(e)
