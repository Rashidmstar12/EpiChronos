import pytest
import os
import sys
import polars as pl
from epichronos.core import MethylationDataset
from epichronos.ai import EpigeneticCopilot

def _setup_mock_data():
    beta_df = pl.DataFrame({
        "chrom": ["chr1", "chr2"],
        "pos": [1000, 2000],
        "S1": [0.35, 0.72],
        "S2": [0.45, 0.65]
    })
    dataset = MethylationDataset(beta_df)
    
    dmr_df = pl.DataFrame({
        "chrom": ["chr1"],
        "start": [1000],
        "end": [1500],
        "num_sites": [5],
        "mean_diff": [0.10],
        "area": [0.50],
        "gene": ["NFKB1"]
    })
    
    enrich_df = pl.DataFrame({
        "pathway": ["Cellular Senescence"],
        "overlap_count": [3],
        "overlap_genes": ["NFKB1, CDKN2A, SIRT1"],
        "p_value": [1.5e-4]
    })
    
    clock_df = pl.DataFrame({
        "sample": ["S1", "S2"],
        "biological_age": [35.0, 45.0],
        "chronological_age": [30.0, 40.0],
        "age_acceleration": [5.0, 5.0]
    })
    
    decon_df = pl.DataFrame({
        "sample": ["S1", "S2"],
        "NK": [0.15, 0.20],
        "Bcell": [0.10, 0.08],
        "Monocytes": [0.75, 0.72]
    })
    return dataset, dmr_df, enrich_df, clock_df, decon_df


def test_ai_copilot_mock_report_generation():
    dataset, dmr_df, enrich_df, clock_df, decon_df = _setup_mock_data()
    
    # Run in offline mock mode (api_key = None)
    report = EpigeneticCopilot.generate_discussion_draft(
        dataset=dataset,
        dmr_df=dmr_df,
        enrich_df=enrich_df,
        clock_df=clock_df,
        decon_df=decon_df,
        focus_area="Aging and Senescence",
        api_key=None
    )
    
    assert isinstance(report, str)
    assert len(report) > 100
    assert "NFKB1" in report
    assert "Cellular Senescence" in report
    assert "Aging and Senescence" in report
    assert "40.00 years" in report
    assert "35.00 years" in report
    assert "5.00 years" in report
    assert "NK" in report
    assert "Bcell" in report
    assert "https://pubmed.ncbi.nlm.nih.gov/?term=NFKB1" in report


def test_ai_copilot_thinking_mode_fallback():
    dataset, dmr_df, enrich_df, clock_df, decon_df = _setup_mock_data()
    
    # When an API key is provided but base_url is invalid, it falls back to mock report cleanly
    report = EpigeneticCopilot.generate_discussion_draft(
        dataset=dataset,
        dmr_df=dmr_df,
        enrich_df=enrich_df,
        clock_df=clock_df,
        decon_df=decon_df,
        focus_area="Immunology and Inflammation",
        api_key="sk-invalid-test-key",
        base_url="https://api.invalid-endpoint-domain.xyz/v1",
        model_name="o3-mini",
        is_thinking_model=True
    )
    
    assert isinstance(report, str)
    # Notice message prepended
    assert "Custom AI Endpoint request (o3-mini) encountered an error" in report
    assert "NFKB1" in report
    assert "Immunology and Inflammation" in report


def test_ai_copilot_cli_entry_hook(tmp_path):
    # Verify that the CLI logic operates cleanly when calling files
    dataset, dmr_df, enrich_df, clock_df, decon_df = _setup_mock_data()
    
    # Save datasets locally in temp directory
    beta_path = os.path.join(tmp_path, "beta.csv")
    dmr_path = os.path.join(tmp_path, "dmr.csv")
    enrich_path = os.path.join(tmp_path, "enrich.csv")
    clock_path = os.path.join(tmp_path, "clock.csv")
    decon_path = os.path.join(tmp_path, "decon.csv")
    output_path = os.path.join(tmp_path, "report.md")
    
    dataset.beta_df.write_csv(beta_path)
    dmr_df.write_csv(dmr_path)
    enrich_df.write_csv(enrich_path)
    clock_df.write_csv(clock_path)
    decon_df.write_csv(decon_path)
    
    # Run CLI command via subprocess or direct execution simulation
    # Simulate arguments
    import subprocess
    cmd = [
        sys.executable,
        "-m", "epichronos.ai",
        "--beta-path", beta_path,
        "--dmr-path", dmr_path,
        "--enrich-path", enrich_path,
        "--clock-path", clock_path,
        "--decon-path", decon_path,
        "--focus", "Cancer and Proliferation",
        "--output", output_path
    ]
    
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Execute CLI script
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=current_dir)
    
    # Assert successful CLI execution
    assert result.returncode == 0
    assert os.path.exists(output_path)
    
    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "EpiChronos Autonomous AI Research Copilot Report" in content
        assert "Cancer and Proliferation" in content
        assert "NFKB1" in content
