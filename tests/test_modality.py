import pytest
import os
import tempfile
import polars as pl
import numpy as np
from epichronos.core import (
    load_nanopore_modkit, 
    load_pacbio_bedgraph, 
    load_array_beta, 
    _load_unified_manifest
)
from epichronos.clocks import calculate_biological_age

def test_load_nanopore_modkit():
    # Setup temporary directory for mock bedmethyl files
    temp_dir = tempfile.mkdtemp()
    
    # Standard 11-column bedmethyl format:
    # chrom, start, end, name, score, strand, thickStart, thickEnd, itemRgb, coverage, percentage
    sample_a_content = (
        "chr1\t1000\t1001\tm\t60\t+\t1000\t1001\t0,0,0\t10\t80.0\n"  # Match
        "chr1\t2000\t2001\tm\t40\t+\t2000\t2001\t0,0,0\t12\t25.0\n"  # Match
        "chr1\t3000\t3001\th\t30\t+\t3000\t3001\t0,0,0\t15\t50.0\n"  # Filtered out (not 'm')
    )
    sample_b_content = (
        "chr1\t1000\t1001\tm\t70\t+\t1000\t1001\t0,0,0\t15\t90.0\n"  # Match
        "chr1\t2000\t2001\tm\t50\t+\t2000\t2001\t0,0,0\t4\t30.0\n"   # Filtered out by min_cov=5
    )
    
    file_a = os.path.join(temp_dir, "sample_a.bedmethyl")
    file_b = os.path.join(temp_dir, "sample_b.bedmethyl")
    
    with open(file_a, "w") as f:
        f.write(sample_a_content)
    with open(file_b, "w") as f:
        f.write(sample_b_content)
        
    # Load dataset with coverage filtering (min_cov=5)
    # This should align both files on chr1:1000, filtering out chr1:2000 (coverage of sample_b < 5)
    # and chr1:3000 (modification code 'h' not 'm')
    dataset = load_nanopore_modkit(
        filepaths=[file_a, file_b],
        sample_names=["S_A", "S_B"],
        min_cov=5
    )
    
    # Assert alignment
    assert dataset.shape[0] == 1
    assert dataset.beta_df["chrom"][0] == "chr1"
    assert dataset.beta_df["pos"][0] == 1000
    assert dataset.beta_df["S_A"][0] == 0.8
    assert dataset.beta_df["S_B"][0] == 0.9
    assert dataset.cov_df["S_A"][0] == 10
    assert dataset.cov_df["S_B"][0] == 15


def test_load_pacbio_bedgraph():
    temp_dir = tempfile.mkdtemp()
    
    # 4-column bedGraph: chrom, start, end, pct (e.g. 0-100 scale)
    sample_a_content = (
        "chr2\t5000\t5001\t75.0\n"
        "chr2\t6000\t6001\t15.0\n"
    )
    
    # 5-column bedGraph: chrom, start, end, pct (0-1 scale), coverage
    sample_b_content = (
        "chr2\t5000\t5001\t0.80\t45\n"
        "chr2\t6000\t6001\t0.20\t50\n"
    )
    
    file_a = os.path.join(temp_dir, "sample_a.bedGraph")
    file_b = os.path.join(temp_dir, "sample_b.bedGraph")
    
    with open(file_a, "w") as f:
        f.write(sample_a_content)
    with open(file_b, "w") as f:
        f.write(sample_b_content)
        
    dataset = load_pacbio_bedgraph(
        filepaths=[file_a, file_b],
        sample_names=["S_A", "S_B"],
        min_cov=5
    )
    
    # Both sites should align
    assert dataset.shape[0] == 2
    
    # Verify auto-scaling of 0-100% to 0-1.0
    assert dataset.beta_df.filter(pl.col("pos") == 5000)["S_A"][0] == 0.75
    assert dataset.beta_df.filter(pl.col("pos") == 5000)["S_B"][0] == 0.80
    
    # Verify default coverage assignment (30) for 4-column files
    assert dataset.cov_df.filter(pl.col("pos") == 5000)["S_A"][0] == 30
    assert dataset.cov_df.filter(pl.col("pos") == 5000)["S_B"][0] == 45


def test_load_array_beta_epic_v2():
    temp_dir = tempfile.mkdtemp()
    
    # Make a mock array beta matrix where:
    # cg09809672 is standard
    # cg02242131_BC21 has an EPIC v2 suffix
    # cg08945781_TC22 has an EPIC v2 suffix
    matrix_content = (
        "probe,S1,S2\n"
        "cg09809672,0.85,0.72\n"
        "cg02242131_BC21,0.25,0.30\n"
        "cg08945781_TC22,0.60,0.55\n"
        "cgNonClockDummy,0.10,0.20\n"
    )
    
    matrix_file = os.path.join(temp_dir, "beta_matrix.csv")
    with open(matrix_file, "w") as f:
        f.write(matrix_content)
        
    # Map mock coordinates into unified manifest
    custom_manifest = {
        "cg09809672": ("chr4", 89456213),
        "cg02242131": ("chr1", 20456213),
        "cg08945781": ("chr2", 45781223)
    }
    
    # Load microarray matrix
    dataset = load_array_beta(matrix_file, manifest=custom_manifest)
    
    # Standard and suffix-containing clock CpGs should be successfully resolved and kept.
    # Dummy non-clock probe is dropped as it is not in the manifest.
    assert dataset.shape[0] == 3
    
    # Verify coordinates resolved correctly for EPIC v2 probe with suffix
    resolved_v2 = dataset.beta_df.filter(pl.col("chrom") == "chr1")
    assert resolved_v2.height == 1
    assert resolved_v2["pos"][0] == 20456213
    assert resolved_v2["S1"][0] == 0.25
    assert resolved_v2["S2"][0] == 0.30


def test_clocks_on_sequencing_modality():
    # Direct sequencing data aligned by coordinates should seamlessly execute biological clocks
    beta_df = pl.DataFrame({
        "chrom": ["chr4", "chr1", "chr3"],
        "pos": [89456213, 20456213, 102458921],
        "S1": [0.80, 0.20, 0.40],
        "S2": [0.75, 0.25, 0.45]
    })
    
    dataset = pl.DataFrame(beta_df)
    
    # We mock the CLOCK_MANIFEST coordinates in clocks.py and run the calculations
    import epichronos.clocks as clocks
    clocks.CLOCK_MANIFEST["cg09809672"] = ("chr4", 89456213)
    clocks.CLOCK_MANIFEST["cg02242131"] = ("chr1", 20456213)
    clocks.CLOCK_MANIFEST["cg06493994"] = ("chr3", 102458921)
    
    # Mock model weights to prevent file-loading dependencies
    clocks._horvath_data = {
        "weights": {"cg09809672": -1.5, "cg02242131": 1.2, "cg06493994": 0.8},
        "reference_means": {"cg09809672": 0.5, "cg02242131": 0.5, "cg06493994": 0.5},
        "intercept": 2.5
    }
    
    # Load dataset
    from epichronos.core import MethylationDataset
    ds = MethylationDataset(beta_df)
    
    # Run clock (should complete without error)
    clock_res = calculate_biological_age(ds, clock_name="horvath")
    assert clock_res.height == 2
    assert "biological_age" in clock_res.columns


def test_generate_report_with_all_modalities():
    from epichronos.core import MethylationDataset
    from epichronos.viz import generate_report
    import tempfile
    
    # Setup minimum inputs
    beta_df = pl.DataFrame({
        "chrom": ["chr1", "chr1"],
        "pos": [1000, 2000],
        "S1": [0.2, 0.8],
        "S2": [0.3, 0.7]
    })
    dataset = MethylationDataset(beta_df)
    dataset.metadata = {"S1": "Ctrl", "S2": "Treat"}
    
    dml_df = pl.DataFrame({
        "chrom": ["chr1"],
        "pos": [1000],
        "p_value": [0.01],
        "mean_diff": [0.10]
    })
    
    dmr_df = pl.DataFrame({
        "chrom": ["chr1"],
        "start": [1000],
        "end": [1500],
        "num_sites": [5],
        "mean_diff": [0.10],
        "area": [0.50],
        "min_p_value": [0.01]
    })
    
    clock_df = pl.DataFrame({
        "sample": ["S1", "S2"],
        "biological_age": [35.0, 45.0],
        "chronological_age": [30.0, 40.0],
        "age_acceleration": [5.0, 5.0]
    })
    
    decon_df = pl.DataFrame({
        "sample": ["S1", "S2"],
        "Neutrophils": [0.1, 0.1],
        "NK": [0.1, 0.1],
        "Bcell": [0.1, 0.1],
        "CD4T": [0.2, 0.2],
        "CD8T": [0.2, 0.2],
        "Monocytes": [0.3, 0.3],
        "Lymphocytes": [0.9, 0.9]
    })
    
    ieaa_df = pl.DataFrame({
        "sample": ["S1", "S2"],
        "intrinsic_age_acceleration": [2.0, 2.0]
    })
    
    expression_df = pl.DataFrame({
        "gene": ["NFKB1"],
        "S1": [10.0],
        "S2": [20.0]
    })
    
    meqtl_df = pl.DataFrame({
        "gene": ["NFKB1"],
        "chrom": ["chr1"],
        "start": [1000],
        "end": [1500],
        "correlation_r": [-0.85],
        "p_value": [1.5e-4],
        "q_value": [1.5e-4],
        "functional_status": ["Functional Silencing"]
    })
    
    enrich_df = pl.DataFrame({
        "pathway": ["Cellular Senescence"],
        "overlap_count": [3],
        "overlap_genes": ["NFKB1, CDKN2A, SIRT1"],
        "p_value": [1.5e-4]
    })
    
    temp_dir = tempfile.gettempdir()
    report_path = os.path.join(temp_dir, "epichronos_test_report.html")
    
    try:
        generate_report(
            dataset=dataset,
            dml_df=dml_df,
            dmr_df=dmr_df,
            clock_df=clock_df,
            output_html_path=report_path,
            decon_df=decon_df,
            ieaa_df=ieaa_df,
            expression_df=expression_df,
            meqtl_df=meqtl_df,
            enrich_df=enrich_df
        )
        
        # Verify the HTML file was compiled and contains key sections
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "EpiChronos Dashboard" in content
            assert "GO/KEGG Pathway Enrichment" in content
            assert "meQTL Linkage" in content
            assert "Blood Immune Cell-Type Proportions" in content
    finally:
        if os.path.exists(report_path):
            os.remove(report_path)
