import streamlit as st
import polars as pl
import numpy as np
import os
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple

# Import core EpiChronos libraries
from epichronos.core import (
    load_bismark_coverage, 
    MethylationDataset, 
    load_array_beta,
    load_nanopore_modkit,
    load_pacbio_bedgraph
)
from epichronos.stats import call_dmls, call_dmrs
from epichronos.clocks import calculate_biological_age, CLOCK_MANIFEST, calculate_intrinsic_age_acceleration
from epichronos.decon import estimate_cell_proportions
from epichronos.transcription import integrate_expression_data
from epichronos.viz import generate_report, _compute_pca
from typing import Union

# Page Configurations
st.set_page_config(
    page_title="EpiChronos Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Premium font styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
    }
    /* Sleek glassmorphic card metrics */
    .glass-metric {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8 !important;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8 !important;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


def generate_synthetic_cohort_data(mode: str) -> Tuple[Union[List[str], str], Dict[str, float], Dict[str, str]]:
    """Generate synthetic Bismark, Nanopore, PacBio, or Microarray files for cohort testing."""
    temp_dir = tempfile.mkdtemp()
    
    samples = ["Ctrl_1", "Ctrl_2", "Ctrl_3", "Treat_1", "Treat_2", "Treat_3"]
    ages = {"Ctrl_1": 25.0, "Ctrl_2": 32.0, "Ctrl_3": 38.0, "Treat_1": 56.0, "Treat_2": 62.0, "Treat_3": 68.0}
    groups = {"Ctrl_1": "Young", "Ctrl_2": "Young", "Ctrl_3": "Young", "Treat_1": "Old", "Treat_2": "Old", "Treat_3": "Old"}
    
    np.random.seed(42)
    num_cpgs = 800
    
    # Coordinates
    chroms = ["chr1"] * 250 + ["chr2"] * 250 + ["chr3"] * 200 + ["chr16"] * 100
    positions = []
    for chrom in ["chr1", "chr2", "chr3", "chr16"]:
        c_count = chroms.count(chrom)
        pos_list = sorted(np.random.randint(1000, 1000000, size=c_count))
        positions.extend(pos_list)
        
    # Inject clock CpG coordinates
    clock_coords = list(CLOCK_MANIFEST.values())
    for idx, (chrom, pos) in enumerate(clock_coords):
        target_idx = (idx * 40) % num_cpgs
        chroms[target_idx] = chrom
        positions[target_idx] = pos

    # Sort genomic coordinates
    sorted_idx = sorted(range(num_cpgs), key=lambda k: (chroms[k], positions[k]))
    chroms = [chroms[i] for i in sorted_idx]
    positions = [positions[i] for i in sorted_idx]

    # For Microarray mode, we create a single CSV beta matrix file
    if "Microarray" in mode:
        matrix_path = os.path.join(temp_dir, "synthetic_epic_v2_matrix.csv")
        rows = []
        for idx in range(num_cpgs):
            chrom = chroms[idx]
            pos = positions[idx]
            
            probe_id = f"cg{idx:08d}"
            for probe, coords in CLOCK_MANIFEST.items():
                if coords == (chrom, pos):
                    # Test EPIC v2 manifest resolution by appending suffix
                    probe_id = f"{probe}_BC21"
                    break
                    
            row = {"probe": probe_id}
            
            for sample in samples:
                age = ages[sample]
                group = groups[sample]
                
                if "_BC21" in probe_id:
                    base_probe = probe_id.split("_")[0]
                    if base_probe == "cg02242131":
                        beta = 0.2 + 0.007 * age + np.random.normal(0, 0.02)
                    elif base_probe == "cg09809672":
                        beta = 0.8 - 0.008 * age + np.random.normal(0, 0.02)
                    elif base_probe == "cg00000292":
                        beta = 0.3 + 0.005 * age + np.random.normal(0, 0.02)
                    elif base_probe == "cg00002426":
                        beta = 0.6 - 0.006 * age + np.random.normal(0, 0.02)
                    elif base_probe == "cg06493994":
                        beta = 0.4 + 0.005 * age + np.random.normal(0, 0.02)
                    else:
                        beta = 0.5 + np.random.normal(0, 0.05)
                elif chrom == "chr2" and 450000 <= pos <= 500000:
                    beta = 0.85 + np.random.normal(0, 0.04) if group == "Old" else 0.25 + np.random.normal(0, 0.04)
                else:
                    beta = 0.15 + np.random.normal(0, 0.05) if idx % 2 == 0 else 0.85 + np.random.normal(0, 0.05)
                    
                row[sample] = float(np.clip(beta, 0.0, 1.0))
            rows.append(row)
            
        pl.DataFrame(rows).write_csv(matrix_path)
        return matrix_path, ages, groups

    # For direct sequencing modes (Bismark, Nanopore, PacBio)
    filepaths = []
    for sample in samples:
        age = ages[sample]
        group = groups[sample]
        
        if "Bismark" in mode:
            filepath = os.path.join(temp_dir, f"{sample}.cov")
        elif "Nanopore" in mode:
            filepath = os.path.join(temp_dir, f"{sample}.bedmethyl")
        else: # PacBio
            filepath = os.path.join(temp_dir, f"{sample}.bedGraph")
            
        filepaths.append(filepath)
        
        with open(filepath, "w") as f:
            for idx in range(num_cpgs):
                chrom = chroms[idx]
                pos = positions[idx]
                
                # Check for clock match
                matched_probe = None
                for probe, coords in CLOCK_MANIFEST.items():
                    if coords == (chrom, pos):
                        matched_probe = probe
                        break
                        
                if matched_probe:
                    if matched_probe == "cg02242131":
                        beta = 0.2 + 0.007 * age + np.random.normal(0, 0.02)
                    elif matched_probe == "cg09809672":
                        beta = 0.8 - 0.008 * age + np.random.normal(0, 0.02)
                    elif matched_probe == "cg00000292":
                        beta = 0.3 + 0.005 * age + np.random.normal(0, 0.02)
                    elif matched_probe == "cg00002426":
                        beta = 0.6 - 0.006 * age + np.random.normal(0, 0.02)
                    elif matched_probe == "cg06493994":
                        beta = 0.4 + 0.005 * age + np.random.normal(0, 0.02)
                    else:
                        beta = 0.5 + np.random.normal(0, 0.05)
                elif chrom == "chr2" and 450000 <= pos <= 500000:
                    beta = 0.85 + np.random.normal(0, 0.04) if group == "Old" else 0.25 + np.random.normal(0, 0.04)
                else:
                    beta = 0.15 + np.random.normal(0, 0.05) if idx % 2 == 0 else 0.85 + np.random.normal(0, 0.05)
                        
                beta = np.clip(beta, 0.0, 1.0)
                depth = int(np.random.poisson(35))
                depth = max(depth, 5)
                
                pct = beta * 100.0
                
                if "Bismark" in mode:
                    methylated = int(round(beta * depth))
                    unmethylated = depth - methylated
                    f.write(f"{chrom}\t{pos}\t{pos}\t{pct:.4f}\t{methylated}\t{unmethylated}\n")
                elif "Nanopore" in mode:
                    # chrom, start, end, name, score, strand, thickStart, thickEnd, itemRgb, coverage, percentage
                    f.write(f"{chrom}\t{pos}\t{pos+1}\tm\t60\t+\t{pos}\t{pos+1}\t0,0,0\t{depth}\t{pct:.4f}\n")
                else: # PacBio 5-column bedGraph
                    f.write(f"{chrom}\t{pos}\t{pos+1}\t{pct:.4f}\t{depth}\n")
                    
    return filepaths, ages, groups


# Header
st.title("🧬 EpiChronos Studio")
st.caption("A High-Performance Unified Downstream DNA Methylation & Biological Aging Analysis Suite")

# Initialize Session State
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "dml_df" not in st.session_state:
    st.session_state.dml_df = None
if "dmr_df" not in st.session_state:
    st.session_state.dmr_df = None
if "clock_df" not in st.session_state:
    st.session_state.clock_df = None
if "decon_df" not in st.session_state:
    st.session_state.decon_df = None
if "ieaa_df" not in st.session_state:
    st.session_state.ieaa_df = None
if "expression_df" not in st.session_state:
    st.session_state.expression_df = None
if "meqtl_df" not in st.session_state:
    st.session_state.meqtl_df = None
if "enrich_df" not in st.session_state:
    st.session_state.enrich_df = None

# Sidebar parameters
st.sidebar.header("⚙️ Pipeline Configuration")

# Sidebar - Mode choice
input_mode = st.sidebar.selectbox(
    "Data Modality Selection",
    [
        "WGBS / RRBS (Bismark .cov)",
        "Oxford Nanopore direct 5mC (Modkit .bedmethyl)",
        "PacBio direct 5mC (bedGraph)",
        "Microarray Beta-Value Matrix (CSV/TSV)"
    ]
)

# Sidebar - Filters
st.sidebar.subheader("🔍 Quality Control Filters")
min_cov = st.sidebar.slider("Min Coverage per CpG", 1, 50, 5)
min_var = st.sidebar.slider("Min variance across cohort", 0.0, 0.1, 0.005, step=0.001)
min_ci_width = st.sidebar.slider("Max Binomial 95% CI Width", 0.05, 0.50, 0.30, step=0.01)
collapse_strands = st.sidebar.checkbox("Watson/Crick strand collapse", value=True)

# Sidebar - Statistical parameters
st.sidebar.subheader("📊 Differential Analysis (DMR)")
p_cutoff = st.sidebar.slider("DML Significance Threshold (P-value)", 0.001, 0.10, 0.05, step=0.005)
max_dist = st.sidebar.number_input("Max Coordinate Distance between CpGs (bp)", 100, 10000, 1500, step=100)
min_sites = st.sidebar.number_input("Min coordinate sites in a DMR", 2, 20, 4)

# Sidebar - Biological clock parameters
st.sidebar.subheader("⏰ Epigenetic Clock Selection")
clock_selection = st.sidebar.selectbox("Model", ["Horvath's Multi-tissue", "Hannum's Blood", "Epigenetic Pacemaker (EPM)"])

# Execution triggers
st.sidebar.markdown("---")
run_analysis = st.sidebar.button("🚀 Run EpiChronos Analytics", use_container_width=True)
use_demo = st.sidebar.button("🧪 Load Synthetic Aging Cohort", use_container_width=True)

# Main Workspace Logic
if use_demo:
    with st.spinner("Generating synthetic 6-sample biological aging cohort..."):
        demo_out, ages, groups = generate_synthetic_cohort_data(input_mode)
        
        # Load and align depending on selected modality
        if "Bismark" in input_mode:
            dataset = load_bismark_coverage(demo_out, ["Ctrl_1", "Ctrl_2", "Ctrl_3", "Treat_1", "Treat_2", "Treat_3"], min_cov=min_cov)
        elif "Nanopore" in input_mode:
            dataset = load_nanopore_modkit(demo_out, ["Ctrl_1", "Ctrl_2", "Ctrl_3", "Treat_1", "Treat_2", "Treat_3"], min_cov=min_cov)
        elif "PacBio" in input_mode:
            dataset = load_pacbio_bedgraph(demo_out, ["Ctrl_1", "Ctrl_2", "Ctrl_3", "Treat_1", "Treat_2", "Treat_3"], min_cov=min_cov)
        else: # Microarray Beta Matrix
            dataset = load_array_beta(demo_out, sample_metadata=groups)
            
        # Apply Watson/Crick strand-aware collapsing and binomial quality control filtering
        if collapse_strands and dataset.cov_df is not None:
            dataset = dataset.collapse_strand_coordinates()
        if dataset.cov_df is not None:
            dataset = dataset.filter_by_binomial_confidence(min_ci_width=min_ci_width)
            
        dataset.metadata = groups
        
        # Stats
        ctrl_samples = [s for s, g in groups.items() if g == "Young"]
        treat_samples = [s for s, g in groups.items() if g == "Old"]
        dml_df = call_dmls(dataset, ctrl_samples, treat_samples)
        dmr_df = call_dmrs(dml_df, p_cutoff=p_cutoff, max_dist=max_dist, min_sites=min_sites)
        
        # Clock calculations
        if "Horvath" in clock_selection:
            clock_name = "horvath"
        elif "Hannum" in clock_selection:
            clock_name = "hannum"
        else:
            clock_name = "pacemaker"
        clock_df = calculate_biological_age(dataset, clock_name=clock_name, chronological_ages=ages)
        
        # Deconvolution & Intrinsic Residuals
        decon_df = estimate_cell_proportions(dataset)
        ieaa_df = calculate_intrinsic_age_acceleration(clock_df, decon_df)
        
        # Multi-Omics Expression Integration (NFKB1 silenced by hypermethylation in controls)
        expression_df = pl.DataFrame({
            "gene": ["NFKB1", "CDKN2A", "IL6", "TNF", "SIRT1"],
            "Ctrl_1": [12.0, 5.0, 10.0, 8.0, 80.0],
            "Ctrl_2": [18.0, 4.0, 15.0, 10.0, 85.0],
            "Ctrl_3": [10.0, 6.0, 12.0, 6.0, 75.0],
            "Treat_1": [78.0, 52.0, 65.0, 40.0, 25.0],
            "Treat_2": [85.0, 60.0, 72.0, 48.0, 18.0],
            "Treat_3": [90.0, 58.0, 80.0, 50.0, 20.0]
        })
        meqtl_df = integrate_expression_data(dataset, dmr_df, expression_df)
        
        # Enrichment ORA Analysis
        from epichronos.enrich import annotate_dmrs_to_genes, perform_pathway_enrichment
        annotated_df = annotate_dmrs_to_genes(dmr_df)
        gene_list = annotated_df["gene"].to_list() if annotated_df.height > 0 else []
        enrich_df = perform_pathway_enrichment(gene_list)
        
        # Save to state
        st.session_state.dataset = dataset
        st.session_state.dml_df = dml_df
        st.session_state.dmr_df = dmr_df
        st.session_state.clock_df = clock_df
        st.session_state.decon_df = decon_df
        st.session_state.ieaa_df = ieaa_df
        st.session_state.expression_df = expression_df
        st.session_state.meqtl_df = meqtl_df
        st.session_state.enrich_df = enrich_df
        st.session_state.analyzed = True
        st.success(f"Synthetic {input_mode} cohort successfully compiled and loaded!")

elif run_analysis:
    st.info("Please upload your cohort data files below to execute the live pipeline.")

# File Uploader Section
if not st.session_state.analyzed:
    st.subheader("📁 Upload Cohort Data")
    
    if "Microarray" not in input_mode:
        file_exts = {
            "WGBS / RRBS (Bismark .cov)": ["cov", "txt"],
            "Oxford Nanopore direct 5mC (Modkit .bedmethyl)": ["bedmethyl", "bed", "tsv", "txt"],
            "PacBio direct 5mC (bedGraph)": ["bedGraph", "bed", "tsv", "txt"]
        }
        curr_exts = file_exts.get(input_mode, ["txt"])
        
        st.write(f"Upload individual direct sequencing files ({input_mode}):")
        uploaded_files = st.file_uploader(
            "Select direct sequencing files", 
            accept_multiple_files=True,
            type=curr_exts
        )
        
        # Optional RNA-seq uploader
        uploaded_expr = st.file_uploader(
            "Upload matching RNA-seq Expression Matrix (CSV/TSV) [Optional]", 
            type=["csv", "tsv", "txt"],
            key="expr_seq"
        )
        
        if uploaded_files:
            st.write(f"Loaded {len(uploaded_files)} raw files.")
            
            # Interactive metadata editor
            st.markdown("### 🏷️ Sample Information Mapping")
            st.write("Assign names, phenotypic groups (e.g. Control vs Treatment) and true chronological ages (optional):")
            
            meta_cols = st.columns(3)
            sample_names = []
            sample_groups = {}
            sample_ages = {}
            
            # Save uploaded files to temp directory
            temp_dir = tempfile.mkdtemp()
            temp_paths = []
            
            for idx, uf in enumerate(uploaded_files):
                default_name = uf.name.split(".")[0]
                
                with meta_cols[0]:
                    s_name = st.text_input(f"File {idx+1} Name", value=default_name, key=f"sname_{idx}")
                    sample_names.append(s_name)
                with meta_cols[1]:
                    s_group = st.text_input(f"File {idx+1} Cohort Group", value="Group_A" if idx % 2 == 0 else "Group_B", key=f"sgrp_{idx}")
                    sample_groups[s_name] = s_group
                with meta_cols[2]:
                    s_age = st.number_input(f"File {idx+1} Chronological Age", value=30.0 + idx*5, key=f"sage_{idx}")
                    sample_ages[s_name] = s_age
                    
                t_path = os.path.join(temp_dir, uf.name)
                with open(t_path, "wb") as f:
                    f.write(uf.getbuffer())
                temp_paths.append(t_path)
                
            trigger_run = st.button("🚀 Process Uploaded Cohort")
            
            if trigger_run:
                with st.spinner("Processing aligned genomic coordinates and running pipeline..."):
                    if "Bismark" in input_mode:
                        dataset = load_bismark_coverage(temp_paths, sample_names, min_cov=min_cov)
                    elif "Nanopore" in input_mode:
                        dataset = load_nanopore_modkit(temp_paths, sample_names, min_cov=min_cov)
                    else: # PacBio
                        dataset = load_pacbio_bedgraph(temp_paths, sample_names, min_cov=min_cov)
                        
                    # Apply Watson/Crick strand-aware collapsing and binomial quality control filtering
                    if collapse_strands and dataset.cov_df is not None:
                        dataset = dataset.collapse_strand_coordinates()
                    if dataset.cov_df is not None:
                        dataset = dataset.filter_by_binomial_confidence(min_ci_width=min_ci_width)
                        
                    dataset.metadata = sample_groups
                    
                    # Group sample lists
                    groups_dict = dataset.get_groups()
                    grps = list(groups_dict.keys())
                    if len(grps) >= 2:
                        group_a = groups_dict[grps[0]]
                        group_b = groups_dict[grps[1]]
                        dml_df = call_dmls(dataset, group_a, group_b)
                        dmr_df = call_dmrs(dml_df, p_cutoff=p_cutoff, max_dist=max_dist, min_sites=min_sites)
                    else:
                        st.warning("Only one phenotypic group found. Statistical DML/DMR calling is skipped.")
                        dml_df = pl.DataFrame(schema={"chrom": pl.String, "pos": pl.Int64, "p_value": pl.Float64, "mean_diff": pl.Float64})
                        dmr_df = pl.DataFrame(schema={"chrom": pl.String, "start": pl.Int64, "end": pl.Int64, "num_sites": pl.Int64, "mean_diff": pl.Float64, "area": pl.Float64})
                        
                    # Clocks
                    if "Horvath" in clock_selection:
                        clock_name = "horvath"
                    elif "Hannum" in clock_selection:
                        clock_name = "hannum"
                    else:
                        clock_name = "pacemaker"
                    clock_df = calculate_biological_age(dataset, clock_name=clock_name, chronological_ages=sample_ages)
                    
                    # Deconvolution & Intrinsic Acceleration
                    decon_df = estimate_cell_proportions(dataset)
                    try:
                        ieaa_df = calculate_intrinsic_age_acceleration(clock_df, decon_df)
                    except Exception:
                        ieaa_df = None
                        
                    # Process RNA-seq if provided
                    expression_df = None
                    meqtl_df = None
                    if uploaded_expr:
                        t_expr_path = os.path.join(temp_dir, uploaded_expr.name)
                        with open(t_expr_path, "wb") as f:
                            f.write(uploaded_expr.getbuffer())
                        sep = "\t" if uploaded_expr.name.endswith((".tsv", ".txt")) else ","
                        expression_df = pl.read_csv(t_expr_path, separator=sep)
                        try:
                            meqtl_df = integrate_expression_data(dataset, dmr_df, expression_df)
                        except Exception as e:
                            st.warning(f"meQTL Correlation failed: {e}")
                    
                    # Enrichment ORA Analysis
                    from epichronos.enrich import annotate_dmrs_to_genes, perform_pathway_enrichment
                    annotated_df = annotate_dmrs_to_genes(dmr_df)
                    gene_list = annotated_df["gene"].to_list() if annotated_df.height > 0 else []
                    enrich_df = perform_pathway_enrichment(gene_list)

                    # Store state
                    st.session_state.dataset = dataset
                    st.session_state.dml_df = dml_df
                    st.session_state.dmr_df = dmr_df
                    st.session_state.clock_df = clock_df
                    st.session_state.decon_df = decon_df
                    st.session_state.ieaa_df = ieaa_df
                    st.session_state.expression_df = expression_df
                    st.session_state.meqtl_df = meqtl_df
                    st.session_state.enrich_df = enrich_df
                    st.session_state.analyzed = True
                    st.rerun()

    else:
        st.write("Upload a single `.csv` or `.tsv` matrix of beta values where the first column contains CpG Probe IDs:")
        uploaded_matrix = st.file_uploader("Select Beta Matrix", type=["csv", "tsv", "txt"])
        
        # Optional RNA-seq uploader
        uploaded_expr = st.file_uploader(
            "Upload matching RNA-seq Expression Matrix (CSV/TSV) [Optional]", 
            type=["csv", "tsv", "txt"],
            key="expr_array"
        )
        
        if uploaded_matrix:
            temp_dir = tempfile.mkdtemp()
            t_path = os.path.join(temp_dir, uploaded_matrix.name)
            with open(t_path, "wb") as f:
                f.write(uploaded_matrix.getbuffer())
                
            preview_df = pl.read_csv(t_path, n_rows=5)
            st.write("Matrix Preview:")
            st.dataframe(preview_df.to_pandas())
            
            sample_names = preview_df.columns[1:]
            
            st.markdown("### 🏷️ Sample Mapping Configuration")
            meta_cols = st.columns(2)
            sample_groups = {}
            sample_ages = {}
            
            for idx, s_name in enumerate(sample_names):
                with meta_cols[0]:
                    s_group = st.text_input(f"Sample '{s_name}' Group", value="Group_A" if idx % 2 == 0 else "Group_B", key=f"sgrp_arr_{idx}")
                    sample_groups[s_name] = s_group
                with meta_cols[1]:
                    s_age = st.number_input(f"Sample '{s_name}' Chronological Age", value=30.0 + idx*5, key=f"sage_arr_{idx}")
                    sample_ages[s_name] = s_age
                    
            trigger_run_array = st.button("🚀 Process Array Matrix")
            
            if trigger_run_array:
                with st.spinner("Aligning array coordinates and running calculations..."):
                    dataset = load_array_beta(t_path, sample_metadata=sample_groups)
                    
                    groups_dict = dataset.get_groups()
                    grps = list(groups_dict.keys())
                    if len(grps) >= 2:
                        group_a = groups_dict[grps[0]]
                        group_b = groups_dict[grps[1]]
                        dml_df = call_dmls(dataset, group_a, group_b)
                        dmr_df = call_dmrs(dml_df, p_cutoff=p_cutoff, max_dist=max_dist, min_sites=min_sites)
                    else:
                        dml_df = pl.DataFrame(schema={"chrom": pl.String, "pos": pl.Int64, "p_value": pl.Float64, "mean_diff": pl.Float64})
                        dmr_df = pl.DataFrame(schema={"chrom": pl.String, "start": pl.Int64, "end": pl.Int64, "num_sites": pl.Int64, "mean_diff": pl.Float64, "area": pl.Float64})
                        
                    if "Horvath" in clock_selection:
                        clock_name = "horvath"
                    elif "Hannum" in clock_selection:
                        clock_name = "hannum"
                    else:
                        clock_name = "pacemaker"
                    clock_df = calculate_biological_age(dataset, clock_name=clock_name, chronological_ages=sample_ages)
                    
                    # Deconvolution & Intrinsic Acceleration
                    decon_df = estimate_cell_proportions(dataset)
                    try:
                        ieaa_df = calculate_intrinsic_age_acceleration(clock_df, decon_df)
                    except Exception:
                        ieaa_df = None
                        
                    # Process RNA-seq if provided
                    expression_df = None
                    meqtl_df = None
                    if uploaded_expr:
                        t_expr_path = os.path.join(temp_dir, uploaded_expr.name)
                        with open(t_expr_path, "wb") as f:
                            f.write(uploaded_expr.getbuffer())
                        sep = "\t" if uploaded_expr.name.endswith((".tsv", ".txt")) else ","
                        expression_df = pl.read_csv(t_expr_path, separator=sep)
                        try:
                            meqtl_df = integrate_expression_data(dataset, dmr_df, expression_df)
                        except Exception as e:
                            st.warning(f"meQTL Correlation failed: {e}")
                    
                    # Enrichment ORA Analysis
                    from epichronos.enrich import annotate_dmrs_to_genes, perform_pathway_enrichment
                    annotated_df = annotate_dmrs_to_genes(dmr_df)
                    gene_list = annotated_df["gene"].to_list() if annotated_df.height > 0 else []
                    enrich_df = perform_pathway_enrichment(gene_list)

                    st.session_state.dataset = dataset
                    st.session_state.dml_df = dml_df
                    st.session_state.dmr_df = dmr_df
                    st.session_state.clock_df = clock_df
                    st.session_state.decon_df = decon_df
                    st.session_state.ieaa_df = ieaa_df
                    st.session_state.expression_df = expression_df
                    st.session_state.meqtl_df = meqtl_df
                    st.session_state.enrich_df = enrich_df
                    st.session_state.analyzed = True
                    st.rerun()

# ----------------- ANALYZED VIEW -----------------
if st.session_state.analyzed:
    dataset = st.session_state.dataset
    dml_df = st.session_state.dml_df
    dmr_df = st.session_state.dmr_df
    clock_df = st.session_state.clock_df
    decon_df = st.session_state.decon_df
    ieaa_df = st.session_state.ieaa_df
    expression_df = st.session_state.expression_df
    meqtl_df = st.session_state.meqtl_df
    
    # Top stats display grid
    st.markdown("### 📊 Cohort Analytics Summary")
    
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.markdown(f"""
        <div class="glass-metric">
            <span class="metric-label">Aligned CpGs</span>
            <div class="metric-val">{dataset.shape[0]:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with metric_cols[1]:
        st.markdown(f"""
        <div class="glass-metric">
            <span class="metric-label">Sample Count</span>
            <div class="metric-val">{len(dataset.samples)}</div>
        </div>
        """, unsafe_allow_html=True)
    with metric_cols[2]:
        sig_count = len(dml_df.filter(pl.col("p_value") <= 0.05)) if dml_df.height > 0 else 0
        st.markdown(f"""
        <div class="glass-metric">
            <span class="metric-label">Active DMLs</span>
            <div class="metric-val">{sig_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with metric_cols[3]:
        st.markdown(f"""
        <div class="glass-metric">
            <span class="metric-label">Called DMRs</span>
            <div class="metric-val">{len(dmr_df)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")

    # Tabs
    tab_overview, tab_pca, tab_dml, tab_enrich, tab_decon, tab_meqtl, tab_clocks, tab_figures, tab_ai, tab_export = st.tabs([
        "📁 Aligned Data Preview", 
        "🌀 Dimensionality Reduction (PCA)", 
        "⚡ Differential Methylation (DML/DMR)", 
        "🧬 GO Pathway Enrichment",
        "🩸 Cell-Type Deconvolution",
        "🔗 Transcription meQTL Linker",
        "⏰ Epigenetic Clocks & Aging",
        "📊 Publication Figures",
        "🤖 AI Research Assistant",
        "📥 Export Report"
    ])
    
    with tab_overview:
        st.subheader("Aligned Genomic Beta Matrix")
        st.write("Preview of your cohort aligned CpG coordinates (chromosome, base position, and beta values):")
        st.dataframe(dataset.beta_df.head(50).to_pandas(), use_container_width=True)
        
    with tab_pca:
        st.subheader("Global Cohort Structuring")
        pca_df = _compute_pca(dataset)
        fig_pca = px.scatter(
            pca_df.to_pandas(), 
            x="PC1", 
            y="PC2", 
            color="group",
            hover_name="sample",
            title="Unsupervised Centering SVD PCA",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.G10
        )
        st.plotly_chart(fig_pca, use_container_width=True)
        
    with tab_dml:
        st.subheader("Differential Analysis Results")
        
        dml_cols = st.columns([3, 2])
        
        with dml_cols[0]:
            if dml_df.height > 0:
                volc_data = dml_df.to_pandas()
                volc_data["neg_log_p"] = -np.log10(volc_data["p_value"] + 1e-300)
                volc_data["status"] = "Not Significant"
                volc_data.loc[(volc_data["p_value"] <= p_cutoff) & (volc_data["mean_diff"].abs() >= 0.1), "status"] = "Significant"
                
                fig_volc = px.scatter(
                    volc_data, 
                    x="mean_diff", 
                    y="neg_log_p", 
                    color="status",
                    hover_name="pos",
                    title="Effect Size vs. Significance (Volcano)",
                    labels={"mean_diff": "Methylation Difference", "neg_log_p": "-log10(P-Value)"},
                    template="plotly_dark",
                    color_discrete_map={"Significant": "#f43f5e", "Not Significant": "#64748b"}
                )
                st.plotly_chart(fig_volc, use_container_width=True)
            else:
                st.info("DML calling is only active when at least two cohort comparison groups are loaded.")
                
        with dml_cols[1]:
            st.markdown("### Top Called DMR Coordinates")
            if dmr_df.height > 0:
                st.dataframe(dmr_df.to_pandas(), use_container_width=True)
            else:
                st.info("No significant contiguous regions meeting your min CpG / max distance parameters were called.")

    with tab_enrich:
        st.subheader("Functional Overrepresentation Analysis (ORA)")
        st.write("Identifies statistically enriched biological pathways based on genes adjacent to called DMRs:")
        
        enrich_df = st.session_state.enrich_df
        
        if enrich_df is not None and enrich_df.height > 0:
            enrich_cols = st.columns([3, 2])
            
            with enrich_cols[0]:
                enrich_data = enrich_df.to_pandas()
                enrich_data["neg_log_p"] = -np.log10(enrich_data["p_value"] + 1e-300)
                enrich_data = enrich_data.sort_values(by="p_value", ascending=False)
                
                fig_enrich = px.bar(
                    enrich_data,
                    x="neg_log_p",
                    y="pathway",
                    orientation="h",
                    color="overlap_count",
                    title="Significantly Enriched Biological Pathways",
                    labels={"neg_log_p": "-log10(P-Value)", "pathway": "Pathway Name", "overlap_count": "Overlapping Genes"},
                    template="plotly_dark",
                    color_continuous_scale=px.colors.sequential.Teal
                )
                st.plotly_chart(fig_enrich, use_container_width=True)
                
            with enrich_cols[1]:
                st.markdown("### Pathway Enrichment Database Scores")
                filtered_enrich = enrich_df.filter(pl.col("overlap_count") > 0).to_pandas()
                if len(filtered_enrich) > 0:
                    st.dataframe(filtered_enrich[["pathway", "overlap_count", "overlap_genes", "p_value"]], use_container_width=True)
                else:
                    st.info("No mapped genes overlap with database biological pathways.")
        else:
            st.info("Pathway enrichment calculation is active after DMR regional coordinate mapping.")

    with tab_decon:
        st.subheader("🩸 Immune Cell-Type Deconvolution")
        st.write(
            "Estimates the relative proportions of 6 major immune cell types using a high-accuracy, "
            "simplex-projected IDOL blood reference panel:"
        )
        
        if decon_df is not None:
            decon_cols = st.columns([3, 2])
            
            with decon_cols[0]:
                cell_types = ["Neutrophils", "NK", "Bcell", "CD4T", "CD8T", "Monocytes"]
                decon_pd = decon_df.to_pandas()
                
                fig_decon = px.bar(
                    decon_pd,
                    x="sample",
                    y=cell_types,
                    title="Immune Cell-Type Proportions per Sample",
                    labels={"value": "Proportion", "sample": "Sample ID", "variable": "Cell Type"},
                    template="plotly_dark",
                    barmode="stack",
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                fig_decon.update_layout(
                    yaxis=dict(range=[0, 1.0]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0)
                )
                st.plotly_chart(fig_decon, use_container_width=True)
                
                if ieaa_df is not None:
                    comp_df = clock_df.join(ieaa_df, on="sample", how="inner")
                    comp_pd = comp_df.to_pandas()
                    
                    if "age_acceleration" in comp_pd.columns:
                        melted = comp_pd.melt(
                            id_vars=["sample"],
                            value_vars=["age_acceleration", "intrinsic_age_acceleration"],
                            var_name="Metric",
                            value_name="Acceleration (Years)"
                        )
                        melted["Metric"] = melted["Metric"].map({
                            "age_acceleration": "EEAA (Unadjusted)",
                            "intrinsic_age_acceleration": "IEAA (Cell-Type Adjusted)"
                        })
                        
                        fig_comp = px.bar(
                            melted,
                            x="sample",
                            y="Acceleration (Years)",
                            color="Metric",
                            barmode="group",
                            title="Epigenetic Age Acceleration comparison (EEAA vs. Intrinsic IEAA)",
                            template="plotly_dark",
                            color_discrete_map={"EEAA (Unadjusted)": "#ef4444", "IEAA (Cell-Type Adjusted)": "#10b981"}
                        )
                        st.plotly_chart(fig_comp, use_container_width=True)
                        
            with decon_cols[1]:
                st.markdown("### Deconvolution Proportions Table")
                st.dataframe(decon_df.to_pandas(), use_container_width=True)
                
                if ieaa_df is not None:
                    st.markdown("### Intrinsic Epigenetic Age Acceleration (IEAA)")
                    st.write(
                        "IEAA represents the cellular-intrinsic aging rate independent of immune system "
                        "composition shifts. Positive residuals indicate accelerated intrinsic aging."
                    )
                    st.dataframe(ieaa_df.to_pandas(), use_container_width=True)
        else:
            st.info("Deconvolution results are available after processing.")

    with tab_meqtl:
        st.subheader("🔗 Transcription meQTL Linker")
        st.write(
            "Integrates matching RNA-seq gene expression matrices with called DMR regions, "
            "identifying which epigenetic alterations physically regulate transcription:"
        )
        
        if meqtl_df is not None and meqtl_df.height > 0:
            meqtl_cols = st.columns([3, 2])
            
            with meqtl_cols[0]:
                # Dynamic gene selector
                genes_with_meqtls = sorted(meqtl_df["gene"].unique().to_list())
                selected_gene = st.selectbox("Select Target Gene for meQTL plot", genes_with_meqtls)
                
                # Filter meQTL record
                record = meqtl_df.filter(pl.col("gene") == selected_gene).to_dicts()[0]
                chrom = record["chrom"]
                start = record["start"]
                end = record["end"]
                r_val = record["correlation_r"]
                q_val = record["q_value"]
                status = record["functional_status"]
                
                # Compute sample-specific DMR beta value
                dmr_sites = dataset.beta_df.filter(
                    (pl.col("chrom") == chrom) & 
                    (pl.col("pos") >= start) & 
                    (pl.col("pos") <= end)
                )
                
                samples_list = dataset.samples
                beta_vals = []
                expr_vals = []
                
                # Fetch expression
                gene_row = expression_df.filter(pl.col("gene") == selected_gene).to_dicts()[0]
                
                for s in samples_list:
                    cpg_betas = dmr_sites[s].drop_nans().drop_nulls().to_list()
                    beta_vals.append(np.mean(cpg_betas) if len(cpg_betas) > 0 else 0.5)
                    expr_vals.append(gene_row[s])
                    
                beta_vals = np.array(beta_vals)
                expr_vals = np.array(expr_vals)
                
                # Interactive Scatter Plot
                fig_scatter = px.scatter(
                    x=beta_vals,
                    y=expr_vals,
                    text=samples_list,
                    title=f"DMR {chrom}:{start}-{end} vs. {selected_gene} Expression",
                    labels={"x": "DMR Average Methylation (Beta)", "y": "Gene Expression Level"},
                    template="plotly_dark",
                    color_discrete_sequence=["#10b981"]
                )
                fig_scatter.update_traces(textposition='top center', marker=dict(size=12))
                
                # Fit linear trendline using pure NumPy
                if len(beta_vals) >= 2:
                    slope, intercept = np.polyfit(beta_vals, expr_vals, 1)
                    x_line = np.linspace(beta_vals.min() - 0.05, beta_vals.max() + 0.05, 100)
                    y_line = slope * x_line + intercept
                    
                    fig_scatter.add_trace(go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name="Linear Fit",
                        line=dict(color="#38bdf8", dash="dash")
                    ))
                    
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                st.markdown(f"""
                **meQTL Correlation Statistics**:
                * **Genomic Coordinates**: `{chrom}:{start}-{end}`
                * **Pearson Correlation ($r$)**: `{r_val:.4f}`
                * **FDR-Adjusted P-Value ($q$)**: `{q_val:.4e}`
                * **Regulatory Status**: <span style="color:{'#f43f5e' if 'Silencing' in status else '#10b981'};font-weight:700;">{status}</span>
                """, unsafe_allow_html=True)
                
            with meqtl_cols[1]:
                st.markdown("### Mapped meQTL Correlations")
                st.dataframe(meqtl_df.to_pandas(), use_container_width=True)
        else:
            st.info(
                "meQTL analysis is inactive. To run, upload a matching RNA-seq count matrix "
                "or load the Synthetic Aging Cohort demo."
            )

    with tab_clocks:
        st.subheader("Biological Aging Analysis")
        
        is_epm = "Pacemaker" in clock_selection
        
        if is_epm and hasattr(dataset, "_last_epm_model") and dataset._last_epm_model is not None:
            # Render EPM specific metrics and plots
            epm_cols = st.columns([2, 2])
            
            with epm_cols[0]:
                st.markdown("### 📈 Pacemaker Convergence & Rates")
                
                # Convergence Loss
                loss_history = dataset._last_epm_model.loss_history_
                fig_loss = px.line(
                    x=list(range(1, len(loss_history) + 1)),
                    y=loss_history,
                    title="EPM Alternating Coordinate-Descent Convergence Loss",
                    labels={"x": "Iteration", "y": "Mean Squared Error (MSE)"},
                    template="plotly_dark",
                    markers=True
                )
                fig_loss.update_traces(line_color="#38bdf8", marker=dict(size=6))
                st.plotly_chart(fig_loss, use_container_width=True)
                
                # Rates Histogram
                rates = dataset._last_epm_model.rates_
                fig_rates = px.histogram(
                    x=rates,
                    title="Distribution of CpG Aging Rates (r_i)",
                    labels={"x": "Methylation Rate (r_i) per unit Biological Age", "y": "CpG Site Count"},
                    template="plotly_dark",
                    nbins=40,
                    color_discrete_sequence=["#10b981"]
                )
                fig_rates.update_layout(
                    shapes=[
                        dict(type="line", x0=0, x1=0, y0=0, y1=1, yref="paper", line=dict(color="#ef4444", dash="dash"))
                    ]
                )
                st.plotly_chart(fig_rates, use_container_width=True)
                
            with epm_cols[1]:
                st.markdown("### ⏰ Biological Pacemaker Age Predictions")
                
                clock_data = clock_df.to_pandas()
                has_chron = "chronological_age" in clock_data.columns
                
                if has_chron:
                    fig_scatter = px.scatter(
                        clock_data,
                        x="chronological_age",
                        y="biological_age",
                        text="sample",
                        title="Predicted Pacemaker Age vs. Chronological Age",
                        labels={"chronological_age": "True Chronological Age (Years)", "biological_age": "Predicted Pacemaker Age (Years)"},
                        template="plotly_dark",
                        color_discrete_sequence=["#38bdf8"]
                    )
                    fig_scatter.update_traces(textposition='top center', marker=dict(size=12))
                    
                    if len(clock_data) >= 2:
                        x_vals = clock_data["chronological_age"].values
                        y_vals = clock_data["biological_age"].values
                        slope, intercept = np.polyfit(x_vals, y_vals, 1)
                        x_line = np.linspace(x_vals.min() - 5, x_vals.max() + 5, 100)
                        y_line = slope * x_line + intercept
                        fig_scatter.add_trace(go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            name="Linear Fit",
                            line=dict(color="#10b981", dash="dash")
                        ))
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    fig_age = px.bar(
                        clock_data,
                        x="sample",
                        y="biological_age",
                        title="Calculated Pacemaker Ages",
                        labels={"biological_age": "Biological Age (Years)", "sample": "Sample"},
                        template="plotly_dark",
                        color_discrete_sequence=["#10b981"]
                    )
                    st.plotly_chart(fig_age, use_container_width=True)
                    
                st.dataframe(clock_df.to_pandas(), use_container_width=True)
                
        else:
            clock_cols = st.columns([3, 2])
            
            with clock_cols[0]:
                has_chron = "chronological_age" in clock_df.columns
                clock_data = clock_df.to_pandas()
                
                if has_chron:
                    fig_age = px.bar(
                        clock_data,
                        x="sample",
                        y="age_acceleration",
                        color="age_acceleration",
                        title="Cohort Epigenetic Age Acceleration (Years)",
                        labels={"age_acceleration": "Residual (Years)", "sample": "Sample"},
                        template="plotly_dark",
                        color_continuous_scale=px.colors.sequential.RdBu_r
                    )
                    st.plotly_chart(fig_age, use_container_width=True)
                else:
                    fig_age = px.bar(
                        clock_data,
                        x="sample",
                        y="biological_age",
                        title="Calculated Biological Ages",
                        labels={"biological_age": "Biological Age (Years)", "sample": "Sample"},
                        template="plotly_dark",
                        color_discrete_sequence=["#10b981"]
                    )
                    st.plotly_chart(fig_age, use_container_width=True)
                    
            with clock_cols[1]:
                st.markdown("### Calculated Epigenetic Age Scores")
                st.dataframe(clock_df.to_pandas(), use_container_width=True)
            
    # ═══════════════════════════════════════════════════════════════════
    # PUBLICATION FIGURES TAB
    # ═══════════════════════════════════════════════════════════════════
    with tab_figures:
        import io as _io
        st.subheader("📊 Publication-Ready Figures")
        st.write(
            "All 8 publication-grade charts — click **📥 Download PNG** under any figure "
            "to save a high-resolution image ready for journal submission."
        )

        def _png_download_button(fig, filename, label="📥 Download PNG"):
            """Render a Streamlit download button for a Plotly figure as PNG."""
            try:
                img_bytes = fig.to_image(format="png", width=1400, height=800, scale=2)
                st.download_button(
                    label=label,
                    data=img_bytes,
                    file_name=filename,
                    mime="image/png",
                    use_container_width=True
                )
            except Exception:
                st.caption("*(PNG export requires `kaleido`: `pip install kaleido`)*")

        # ─── Figure 1: Beta Value Distribution ────────────────────────
        st.markdown("---")
        st.markdown("### Figure 1 — Global CpG Beta Value Distribution")
        st.caption("Shows the overall methylation landscape across your cohort. Bimodal peaks at 0 and 1 confirm good methylation calling quality.")
        fig1_col, fig1_btn = st.columns([5, 1])
        with fig1_col:
            sample_cols = dataset.samples
            beta_pd = dataset.beta_df.select(sample_cols).to_pandas()
            import plotly.figure_factory as ff
            hist_data = []
            group_labels = []
            colors_fig1 = px.colors.qualitative.G10
            for i, s in enumerate(sample_cols[:8]):
                vals = beta_pd[s].dropna().values
                if len(vals) > 0:
                    hist_data.append(vals.tolist())
                    group_labels.append(s)
            if hist_data:
                fig1 = ff.create_distplot(
                    hist_data, group_labels,
                    bin_size=0.02, show_rug=False, show_hist=False,
                    colors=colors_fig1[:len(hist_data)]
                )
                fig1.update_layout(
                    template="plotly_dark",
                    title="Per-Sample CpG Beta Value Density Distribution",
                    xaxis_title="Beta Value (methylation proportion)",
                    yaxis_title="Density",
                    legend_title="Sample",
                    font=dict(family="Inter", size=13)
                )
                st.plotly_chart(fig1, use_container_width=True)
        with fig1_btn:
            st.write("")
            st.write("")
            if hist_data:
                _png_download_button(fig1, "fig1_beta_distribution.png")

        # ─── Figure 2: Top-N CpG Heatmap ──────────────────────────────
        st.markdown("---")
        st.markdown("### Figure 2 — Top Differentially Methylated CpG Heatmap")
        st.caption("Heatmap of the top most variable CpGs across all samples. Reveals sample clustering and methylation patterns at a glance.")
        fig2_col, fig2_btn = st.columns([5, 1])
        with fig2_col:
            n_top = st.slider("Number of top CpGs to display", 20, 200, 50, key="heatmap_n")
            sample_cols_hm = dataset.samples
            beta_mat = dataset.beta_df.select(sample_cols_hm).to_pandas()
            variances = beta_mat.var(axis=1)
            top_idx = variances.nlargest(n_top).index
            heatmap_data = beta_mat.loc[top_idx].fillna(0.5)
            coords = dataset.beta_df.select(["chrom", "pos"]).to_pandas()
            y_labels = [
                f"{coords.loc[i,'chrom']}:{coords.loc[i,'pos']}" 
                for i in top_idx
            ]
            fig2 = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=sample_cols_hm,
                y=y_labels,
                colorscale="RdBu_r",
                zmid=0.5,
                colorbar=dict(title="Beta Value")
            ))
            fig2.update_layout(
                template="plotly_dark",
                title=f"Top {n_top} Most Variable CpGs — Methylation Heatmap",
                xaxis_title="Sample",
                yaxis_title="CpG Coordinate",
                yaxis=dict(tickfont=dict(size=8)),
                height=max(400, n_top * 12),
                font=dict(family="Inter", size=13)
            )
            st.plotly_chart(fig2, use_container_width=True)
        with fig2_btn:
            st.write("")
            st.write("")
            _png_download_button(fig2, "fig2_cpg_heatmap.png")

        # ─── Figure 3: Manhattan Plot ──────────────────────────────────
        st.markdown("---")
        st.markdown("### Figure 3 — Genome-Wide Manhattan Plot")
        st.caption("Genome-wide distribution of DML significance. Peaks above the red line (p < 0.05 after FDR) indicate significant loci.")
        fig3_col, fig3_btn = st.columns([5, 1])
        with fig3_col:
            if dml_df.height > 0:
                manh_pd = dml_df.to_pandas().copy()
                manh_pd["neg_log_p"] = -np.log10(manh_pd["p_value"] + 1e-300)
                # Assign numeric chromosome order
                chrom_order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
                chrom_present = [c for c in chrom_order if c in manh_pd["chrom"].values]
                other_chroms = sorted([c for c in manh_pd["chrom"].unique() if c not in chrom_order])
                chrom_present += other_chroms
                chrom_map = {c: i for i, c in enumerate(chrom_present)}
                manh_pd["chrom_num"] = manh_pd["chrom"].map(chrom_map).fillna(0)
                manh_pd["x_pos"] = manh_pd["chrom_num"] * 1000 + (manh_pd["pos"] / manh_pd["pos"].max() * 900)
                palette = px.colors.qualitative.Alphabet
                manh_pd["color"] = manh_pd["chrom_num"].apply(lambda x: palette[int(x) % len(palette)])
                manh_pd["sig"] = manh_pd["p_value"] <= p_cutoff
                fig3 = go.Figure()
                # Non-significant
                ns = manh_pd[~manh_pd["sig"]]
                fig3.add_trace(go.Scattergl(
                    x=ns["x_pos"], y=ns["neg_log_p"],
                    mode="markers", name="Not Significant",
                    marker=dict(color=ns["color"], size=4, opacity=0.5),
                    text=ns["chrom"].astype(str) + ":" + ns["pos"].astype(str),
                    hovertemplate="%{text}<br>-log10(p)=%{y:.2f}<extra></extra>"
                ))
                # Significant
                sig = manh_pd[manh_pd["sig"]]
                if len(sig) > 0:
                    fig3.add_trace(go.Scattergl(
                        x=sig["x_pos"], y=sig["neg_log_p"],
                        mode="markers", name="Significant",
                        marker=dict(color="#f43f5e", size=7, opacity=0.9),
                        text=sig["chrom"].astype(str) + ":" + sig["pos"].astype(str),
                        hovertemplate="%{text}<br>-log10(p)=%{y:.2f}<extra></extra>"
                    ))
                # Significance threshold line
                sig_line = -np.log10(p_cutoff)
                fig3.add_hline(y=sig_line, line_dash="dash", line_color="#ef4444",
                               annotation_text=f"p = {p_cutoff}", annotation_position="top right")
                # X-axis chromosome labels
                tick_vals = [chrom_map[c] * 1000 + 450 for c in chrom_present if c in chrom_map]
                fig3.update_layout(
                    template="plotly_dark",
                    title="Genome-Wide Association — Manhattan Plot",
                    xaxis=dict(tickmode="array", tickvals=tick_vals, ticktext=chrom_present,
                               tickangle=45, title="Chromosome"),
                    yaxis_title="-log₁₀(P-Value)",
                    height=500,
                    showlegend=True,
                    font=dict(family="Inter", size=13)
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Manhattan plot requires DML results from differential methylation calling.")
                fig3 = None
        with fig3_btn:
            st.write("")
            st.write("")
            if dml_df.height > 0 and fig3 is not None:
                _png_download_button(fig3, "fig3_manhattan_plot.png")

        # ─── Figure 4: Sample-Sample Correlation Matrix ────────────────
        st.markdown("---")
        st.markdown("### Figure 4 — Sample–Sample Pearson Correlation Matrix")
        st.caption("Pairwise correlations between all samples. High correlations within the same group confirm cohort quality. Outliers indicate batch effects or mislabelled samples.")
        fig4_col, fig4_btn = st.columns([5, 1])
        with fig4_col:
            sample_cols_corr = dataset.samples
            beta_corr = dataset.beta_df.select(sample_cols_corr).to_pandas().fillna(0.5)
            corr_matrix = beta_corr.corr(method="pearson")
            fig4 = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=sample_cols_corr,
                y=sample_cols_corr,
                colorscale="RdBu",
                zmid=0,
                zmin=-1, zmax=1,
                colorbar=dict(title="Pearson r"),
                text=np.round(corr_matrix.values, 3),
                texttemplate="%{text}",
                textfont=dict(size=11)
            ))
            fig4.update_layout(
                template="plotly_dark",
                title="Pairwise Sample Methylation Pearson Correlation Matrix",
                xaxis_title="Sample",
                yaxis_title="Sample",
                height=500,
                font=dict(family="Inter", size=13)
            )
            st.plotly_chart(fig4, use_container_width=True)
        with fig4_btn:
            st.write("")
            st.write("")
            _png_download_button(fig4, "fig4_correlation_matrix.png")

        # ─── Figure 5: Coverage Depth Distribution ────────────────────
        st.markdown("---")
        st.markdown("### Figure 5 — Per-Sample CpG Coverage Depth Distribution")
        st.caption("Shows how deeply each CpG was sequenced per sample. Low-coverage samples may need exclusion. Useful for Methods section figures.")
        fig5_col, fig5_btn = st.columns([5, 1])
        with fig5_col:
            if dataset.cov_df is not None:
                cov_pd = dataset.cov_df.select(dataset.samples).to_pandas()
                box_traces = []
                for s in dataset.samples:
                    vals = cov_pd[s].dropna().values
                    box_traces.append(go.Box(
                        y=vals, name=s,
                        boxpoints="outliers",
                        marker=dict(size=3),
                        line=dict(width=2)
                    ))
                fig5 = go.Figure(data=box_traces)
                fig5.update_layout(
                    template="plotly_dark",
                    title="Per-Sample CpG Read Depth (Coverage) Distribution",
                    yaxis_title="Read Depth (×)",
                    xaxis_title="Sample",
                    height=500,
                    showlegend=False,
                    font=dict(family="Inter", size=13)
                )
                st.plotly_chart(fig5, use_container_width=True)
            else:
                # Simulate coverage hint if no cov_df
                st.info("Coverage distribution plot requires a loaded coverage matrix (e.g. from Bismark .cov files). Array data does not include coverage depth.")
                fig5 = None
        with fig5_btn:
            st.write("")
            st.write("")
            if dataset.cov_df is not None and fig5 is not None:
                _png_download_button(fig5, "fig5_coverage_distribution.png")

        # ─── Figure 6: Age Acceleration Violin Plot ────────────────────
        st.markdown("---")
        st.markdown("### Figure 6 — Epigenetic Age Acceleration by Group (Violin Plot)")
        st.caption("Compares the distribution of epigenetic age acceleration across phenotypic groups. Positive values = accelerated aging.")
        fig6_col, fig6_btn = st.columns([5, 1])
        with fig6_col:
            if clock_df.height > 0 and "age_acceleration" in clock_df.columns:
                clock_pd_v = clock_df.to_pandas()
                # Add group labels
                clock_pd_v["Group"] = clock_pd_v["sample"].map(
                    lambda s: dataset.metadata.get(s, "Unknown")
                )
                fig6 = go.Figure()
                groups_v = clock_pd_v["Group"].unique()
                colors_v = px.colors.qualitative.G10
                for idx_g, grp in enumerate(groups_v):
                    grp_data = clock_pd_v[clock_pd_v["Group"] == grp]["age_acceleration"].dropna()
                    fig6.add_trace(go.Violin(
                        y=grp_data,
                        name=grp,
                        box_visible=True,
                        meanline_visible=True,
                        points="all",
                        jitter=0.3,
                        marker=dict(size=8),
                        line_color=colors_v[idx_g % len(colors_v)]
                    ))
                fig6.add_hline(y=0, line_dash="dash", line_color="#94a3b8",
                               annotation_text="No Acceleration", annotation_position="top right")
                fig6.update_layout(
                    template="plotly_dark",
                    title="Epigenetic Age Acceleration Distribution by Phenotypic Group",
                    yaxis_title="Age Acceleration (Years)",
                    xaxis_title="Cohort Group",
                    height=500,
                    font=dict(family="Inter", size=13)
                )
                st.plotly_chart(fig6, use_container_width=True)
            else:
                st.info("Age Acceleration violin plot requires chronological ages to be provided and epigenetic clock estimation to be run.")
                fig6 = None
        with fig6_btn:
            st.write("")
            st.write("")
            if clock_df.height > 0 and "age_acceleration" in clock_df.columns and fig6 is not None:
                _png_download_button(fig6, "fig6_age_acceleration_violin.png")

        # ─── Figure 7: DMR Lollipop Plot ──────────────────────────────
        st.markdown("---")
        st.markdown("### Figure 7 — DMR Lollipop Plot")
        st.caption("Publication-standard display of top called DMRs. Lollipop height = effect size (mean Δβ). Red = hypermethylated, Blue = hypomethylated.")
        fig7_col, fig7_btn = st.columns([5, 1])
        with fig7_col:
            if dmr_df.height > 0:
                lollipop_pd = dmr_df.to_pandas().copy()
                lollipop_pd["label"] = lollipop_pd["chrom"] + ":" + lollipop_pd["start"].astype(str)
                lollipop_pd["size"] = lollipop_pd["num_sites"] * 4
                lollipop_pd["color"] = lollipop_pd["mean_diff"].apply(
                    lambda x: "#f43f5e" if x > 0 else "#38bdf8"
                )
                lollipop_pd = lollipop_pd.sort_values("mean_diff", key=abs, ascending=False).head(30)
                fig7 = go.Figure()
                # Stems
                for _, row in lollipop_pd.iterrows():
                    fig7.add_shape(
                        type="line",
                        x0=row["label"], x1=row["label"],
                        y0=0, y1=row["mean_diff"],
                        line=dict(color="#64748b", width=1.5)
                    )
                # Heads
                fig7.add_trace(go.Scatter(
                    x=lollipop_pd["label"],
                    y=lollipop_pd["mean_diff"],
                    mode="markers",
                    marker=dict(
                        color=lollipop_pd["color"],
                        size=lollipop_pd["size"].clip(6, 24),
                        line=dict(color="white", width=1)
                    ),
                    text=lollipop_pd.apply(
                        lambda r: f"{r['label']}<br>Δβ={r['mean_diff']:.3f}<br>{r['num_sites']} CpGs", axis=1
                    ),
                    hovertemplate="%{text}<extra></extra>"
                ))
                fig7.add_hline(y=0, line_dash="solid", line_color="#94a3b8")
                fig7.update_layout(
                    template="plotly_dark",
                    title="Top Called DMRs — Lollipop Plot (Red=Hyper, Blue=Hypo)",
                    xaxis=dict(title="DMR Coordinate", tickangle=45, tickfont=dict(size=9)),
                    yaxis_title="Mean Methylation Difference (Δβ)",
                    height=550,
                    showlegend=False,
                    font=dict(family="Inter", size=13)
                )
                st.plotly_chart(fig7, use_container_width=True)
            else:
                st.info("DMR Lollipop plot requires called DMRs from differential methylation analysis.")
                fig7 = None
        with fig7_btn:
            st.write("")
            st.write("")
            if dmr_df.height > 0 and fig7 is not None:
                _png_download_button(fig7, "fig7_dmr_lollipop.png")

        # ─── Figure 8: Cell-Type Composition Stacked Area ─────────────
        st.markdown("---")
        st.markdown("### Figure 8 — Immune Cell Composition Stacked Area Chart")
        st.caption("Per-sample immune landscape as a continuous stacked area. Shows the relative contribution of each immune cell type across your cohort.")
        fig8_col, fig8_btn = st.columns([5, 1])
        with fig8_col:
            if decon_df is not None:
                decon_pd_area = decon_df.to_pandas()
                cell_types_area = [c for c in decon_pd_area.columns if c != "sample"]
                fig8 = go.Figure()
                area_colors = px.colors.qualitative.Vivid
                for idx_c, ct in enumerate(cell_types_area):
                    fig8.add_trace(go.Scatter(
                        x=decon_pd_area["sample"],
                        y=decon_pd_area[ct],
                        mode="lines+markers",
                        name=ct,
                        stackgroup="one",
                        fillcolor=area_colors[idx_c % len(area_colors)],
                        line=dict(width=0.8),
                        hovertemplate=f"{ct}: %{{y:.1%}}<extra></extra>"
                    ))
                fig8.update_layout(
                    template="plotly_dark",
                    title="Per-Sample Immune Cell-Type Composition (Stacked Area)",
                    xaxis_title="Sample",
                    yaxis=dict(title="Cell-Type Proportion", tickformat=".0%", range=[0, 1]),
                    legend_title="Cell Type",
                    height=500,
                    font=dict(family="Inter", size=13)
                )
                st.plotly_chart(fig8, use_container_width=True)
            else:
                st.info("Cell composition area chart requires cell-type deconvolution to be run.")
                fig8 = None
        with fig8_btn:
            st.write("")
            st.write("")
            if decon_df is not None and fig8 is not None:
                _png_download_button(fig8, "fig8_cell_composition_area.png")

        st.markdown("---")
        st.caption("💡 Tip: All figures use dark-mode Plotly. PNG exports are 1400×800px @2× scale — ready for journal submission.")

    with tab_ai:
        st.subheader("🤖 Autonomous AI Research Copilot")
        st.write(
            "Synthesize your differential methylation, biological clocks, pathway enrichment, and cell-type "
            "deconvolution results into an authoritative, publication-ready scientific discussion document:"
        )
        
        ai_cols = st.columns([2, 3])
        
        with ai_cols[0]:
            st.markdown("### ⚙️ Copilot Configuration")
            openai_key = st.text_input(
                "Endpoint API Key (Optional)", 
                type="password", 
                help="Leave empty to run the offline high-fidelity mock literature synthesis fallback!"
            )
            
            openai_base = st.text_input(
                "API Base URL",
                value="https://api.openai.com/v1",
                help="Enter any OpenAI-compatible base URL (e.g. https://api.deepseek.com/v1, http://localhost:11434/v1 for Ollama)."
            )
            
            model_name = st.text_input(
                "Model Name",
                value="gpt-4o-mini",
                help="Enter the exact model name identifier (e.g., gpt-4o, deepseek-reasoner, o3-mini, llama3.1)."
            )
            
            is_thinking = st.checkbox(
                "Reasoning/Thinking Model (e.g., o3-mini, deepseek-reasoner)",
                value=False,
                help="Check this for reasoning models that have strict API requirements (sets temp=1.0 and merges system instructions)."
            )
            
            ai_focus = st.selectbox(
                "Scientific Focus Area",
                [
                    "General Epigenetics",
                    "Aging and Senescence",
                    "Immunology and Inflammation",
                    "Oncogenesis and Proliferation",
                    "Metabolism and Epigenetics"
                ]
            )
            
            generate_ai_report = st.button("🪄 Generate Literature Synthesis & Discussion", use_container_width=True)
            
        with ai_cols[1]:
            ai_subtab_report, ai_subtab_chat = st.tabs(["📝 Discussion Report", "💬 Interactive Chat"])
            
            with ai_subtab_report:
                st.markdown("### Generated Biological Synthesis")
                if generate_ai_report:
                    with st.spinner("Synthesizing multi-omics parameters and drafting scientific discussion..."):
                        from epichronos.ai import EpigeneticCopilot
                        
                        enrich_df = st.session_state.enrich_df
                        decon_df = st.session_state.decon_df
                        
                        report_text = EpigeneticCopilot.generate_discussion_draft(
                            dataset=dataset,
                            dmr_df=dmr_df,
                            enrich_df=enrich_df,
                            clock_df=clock_df,
                            decon_df=decon_df,
                            focus_area=ai_focus,
                            api_key=openai_key if openai_key else None,
                            base_url=openai_base,
                            model_name=model_name,
                            is_thinking_model=is_thinking
                        )
                        
                        st.session_state.ai_report_text = report_text
                        st.success("Draft generated successfully!")
                        
                if "ai_report_text" in st.session_state:
                    st.markdown(st.session_state.ai_report_text, unsafe_allow_html=True)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Discussion Markdown",
                        data=st.session_state.ai_report_text,
                        file_name="epichronos_discussion.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
            
            with ai_subtab_chat:
                st.markdown("### Ask the AI Copilot")
                if "ai_chat_messages" not in st.session_state:
                    st.session_state.ai_chat_messages = []
                
                # Display chat messages from history
                chat_container = st.container(height=400)
                with chat_container:
                    for message in st.session_state.ai_chat_messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                # React to user input
                if prompt := st.chat_input("Ask about your epigenetic analysis..."):
                    st.session_state.ai_chat_messages.append({"role": "user", "content": prompt})
                    with chat_container:
                        with st.chat_message("user"):
                            st.markdown(prompt)
                            
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                from epichronos.ai import EpigeneticCopilot
                                response = EpigeneticCopilot.chat_completion(
                                    messages=st.session_state.ai_chat_messages,
                                    dataset=dataset,
                                    dmr_df=dmr_df,
                                    enrich_df=st.session_state.enrich_df,
                                    clock_df=clock_df,
                                    decon_df=st.session_state.decon_df,
                                    api_key=openai_key if openai_key else None,
                                    base_url=openai_base,
                                    model_name=model_name,
                                    is_thinking_model=is_thinking
                                )
                                st.markdown(response)
                        st.session_state.ai_chat_messages.append({"role": "assistant", "content": response})
            
    with tab_export:
        st.subheader("📥 Download Analysis Artifacts")
        st.write("Export your fully aligned, normalized data matrices and compile interactive standalone reports:")
        
        import io
        import zipfile
        import datetime
        
        # ─────────────────────────────────────────────────────────────
        # Section 1: Individual CSV Downloads
        # ─────────────────────────────────────────────────────────────
        st.markdown("### 📄 Individual CSV Exports")
        csv_col1, csv_col2 = st.columns(2)
        
        with csv_col1:
            # Download aligned beta matrix
            csv_beta = dataset.beta_df.to_pandas().to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Aligned Beta Matrix (CSV)",
                data=csv_beta,
                file_name="epichronos_aligned_betas.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            if dml_df.height > 0:
                csv_dml = dml_df.to_pandas().to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 DML Loci Statistics (CSV)",
                    data=csv_dml,
                    file_name="epichronos_dml_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            if dmr_df.height > 0:
                csv_dmr = dmr_df.to_pandas().to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 DMR Regions (CSV)",
                    data=csv_dmr,
                    file_name="epichronos_dmr_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            if clock_df.height > 0:
                csv_clock = clock_df.to_pandas().to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Epigenetic Clock Ages (CSV)",
                    data=csv_clock,
                    file_name="epichronos_clock_ages.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with csv_col2:
            enrich_df = st.session_state.enrich_df
            if enrich_df is not None and enrich_df.height > 0:
                csv_enrich = enrich_df.to_pandas().to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 GO Pathway Enrichment (CSV)",
                    data=csv_enrich,
                    file_name="epichronos_enrichment_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            if decon_df is not None:
                csv_decon = decon_df.to_pandas().to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Immune Cell Proportions (CSV)",
                    data=csv_decon,
                    file_name="epichronos_cell_proportions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            if ieaa_df is not None:
                csv_ieaa = ieaa_df.to_pandas().to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Age Acceleration IEAA (CSV)",
                    data=csv_ieaa,
                    file_name="epichronos_ieaa_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            if meqtl_df is not None and meqtl_df.height > 0:
                csv_meqtl = meqtl_df.to_pandas().to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 meQTL Correlations (CSV)",
                    data=csv_meqtl,
                    file_name="epichronos_meqtl_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # ─────────────────────────────────────────────────────────────
        # Section 2: BED File Export (Genome Browser)
        # ─────────────────────────────────────────────────────────────
        st.write("---")
        st.markdown("### 🧬 BED File Export (Genome Browser)")
        st.write("Load your DMRs directly into UCSC Genome Browser, IGV, or any BED-compatible viewer:")
        
        bed_col1, bed_col2 = st.columns(2)
        
        with bed_col1:
            if dmr_df.height > 0:
                bed_lines = ['track name="EpiChronos_DMRs" description="Differentially Methylated Regions" visibility=2 itemRgb="On"']
                dmr_pd = dmr_df.to_pandas()
                for _, row in dmr_pd.iterrows():
                    chrom = row["chrom"]
                    start = int(row["start"])
                    end = int(row["end"])
                    diff = row.get("mean_diff", 0)
                    n_sites = int(row.get("num_sites", 1))
                    score = min(1000, int(abs(diff) * 1000))
                    # Red for hyper, Blue for hypo
                    color = "255,0,0" if diff > 0 else "0,0,255"
                    name = f"DMR_{chrom}_{start}_{n_sites}sites"
                    bed_lines.append(f"{chrom}\t{start}\t{end}\t{name}\t{score}\t.\t{start}\t{end}\t{color}")
                bed_data = "\n".join(bed_lines).encode("utf-8")
                st.download_button(
                    label="🧬 Download DMR BED File",
                    data=bed_data,
                    file_name="epichronos_dmrs.bed",
                    mime="text/plain",
                    use_container_width=True
                )
            else:
                st.info("No DMRs called — BED export requires at least one significant region.")
        
        with bed_col2:
            if dml_df.height > 0:
                sig_dml = dml_df.filter(pl.col("p_value") <= 0.05)
                if sig_dml.height > 0:
                    bed_lines_dml = ['track name="EpiChronos_DMLs" description="Significant Differentially Methylated Loci" visibility=1 itemRgb="On"']
                    sig_pd = sig_dml.to_pandas()
                    for _, row in sig_pd.iterrows():
                        chrom = row["chrom"]
                        pos = int(row["pos"])
                        diff = row.get("mean_diff", 0)
                        score = min(1000, int(abs(diff) * 1000))
                        color = "220,50,50" if diff > 0 else "50,50,220"
                        bed_lines_dml.append(f"{chrom}\t{pos}\t{pos+1}\tDML_{chrom}_{pos}\t{score}\t.\t{pos}\t{pos+1}\t{color}")
                    bed_dml_data = "\n".join(bed_lines_dml).encode("utf-8")
                    st.download_button(
                        label="🧬 Download Significant DML BED File",
                        data=bed_dml_data,
                        file_name="epichronos_significant_dmls.bed",
                        mime="text/plain",
                        use_container_width=True
                    )
        
        # ─────────────────────────────────────────────────────────────
        # Section 3: Excel Multi-Sheet Workbook
        # ─────────────────────────────────────────────────────────────
        st.write("---")
        st.markdown("### 📊 Excel Workbook Export")
        st.write("All results in a single `.xlsx` workbook with separate sheets — ready for collaborators:")
        
        try:
            xlsx_buffer = io.BytesIO()
            import pandas as pd
            with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
                dataset.beta_df.to_pandas().to_excel(writer, sheet_name="Beta_Matrix", index=False)
                if dml_df.height > 0:
                    dml_df.to_pandas().to_excel(writer, sheet_name="DML_Results", index=False)
                if dmr_df.height > 0:
                    dmr_df.to_pandas().to_excel(writer, sheet_name="DMR_Results", index=False)
                if clock_df.height > 0:
                    clock_df.to_pandas().to_excel(writer, sheet_name="Clock_Ages", index=False)
                if enrich_df is not None and enrich_df.height > 0:
                    enrich_df.to_pandas().to_excel(writer, sheet_name="Pathway_Enrichment", index=False)
                if decon_df is not None:
                    decon_df.to_pandas().to_excel(writer, sheet_name="Cell_Deconvolution", index=False)
                if ieaa_df is not None:
                    ieaa_df.to_pandas().to_excel(writer, sheet_name="IEAA_Acceleration", index=False)
                if meqtl_df is not None and meqtl_df.height > 0:
                    meqtl_df.to_pandas().to_excel(writer, sheet_name="meQTL_Correlations", index=False)
                    
                # Summary sheet
                summary_data = {
                    "Metric": [
                        "Total CpG Sites", "Total Samples", "Significant DMLs (p<0.05)",
                        "Called DMRs", "Clock Models Used", "Analysis Date"
                    ],
                    "Value": [
                        str(dataset.shape[0]), str(len(dataset.samples)),
                        str(dml_df.filter(pl.col("p_value") <= 0.05).height) if dml_df.height > 0 else "0",
                        str(dmr_df.height), "Horvath + Hannum",
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
            
            xlsx_buffer.seek(0)
            st.download_button(
                label="📊 Download Complete Excel Workbook (.xlsx)",
                data=xlsx_buffer.getvalue(),
                file_name="epichronos_complete_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except ImportError:
            st.warning("Install `openpyxl` for Excel export: `pip install openpyxl`")
        
        # ─────────────────────────────────────────────────────────────
        # Section 4: Standalone HTML Report
        # ─────────────────────────────────────────────────────────────
        st.write("---")
        st.markdown("### 🖥️ Interactive HTML Dashboard Report")
        st.write("A portable, fully offline standalone HTML dashboard with embedded Plotly charts:")
        
        temp_report = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        generate_report(
            dataset, dml_df, dmr_df, clock_df, temp_report.name, 
            decon_df=decon_df, ieaa_df=ieaa_df, expression_df=expression_df, meqtl_df=meqtl_df,
            enrich_df=st.session_state.enrich_df
        )
        
        with open(temp_report.name, "r", encoding="utf-8") as f:
            html_data = f.read()
            
        st.download_button(
            label="🖥️ Compile & Download Interactive HTML Report",
            data=html_data,
            file_name="epichronos_analytics_dashboard.html",
            mime="text/html",
            use_container_width=True
        )
        
        # ─────────────────────────────────────────────────────────────
        # Section 5: Download ALL as ZIP
        # ─────────────────────────────────────────────────────────────
        st.write("---")
        st.markdown("### 📦 Download All Results (ZIP)")
        st.write("One click — every CSV, BED file, Excel workbook, and HTML report bundled together:")
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # CSVs
            zf.writestr("csv/aligned_betas.csv", dataset.beta_df.to_pandas().to_csv(index=False))
            if dml_df.height > 0:
                zf.writestr("csv/dml_results.csv", dml_df.to_pandas().to_csv(index=False))
            if dmr_df.height > 0:
                zf.writestr("csv/dmr_results.csv", dmr_df.to_pandas().to_csv(index=False))
            if clock_df.height > 0:
                zf.writestr("csv/clock_ages.csv", clock_df.to_pandas().to_csv(index=False))
            if enrich_df is not None and enrich_df.height > 0:
                zf.writestr("csv/enrichment_results.csv", enrich_df.to_pandas().to_csv(index=False))
            if decon_df is not None:
                zf.writestr("csv/cell_proportions.csv", decon_df.to_pandas().to_csv(index=False))
            if ieaa_df is not None:
                zf.writestr("csv/ieaa_results.csv", ieaa_df.to_pandas().to_csv(index=False))
            if meqtl_df is not None and meqtl_df.height > 0:
                zf.writestr("csv/meqtl_results.csv", meqtl_df.to_pandas().to_csv(index=False))
            
            # BED files
            if dmr_df.height > 0:
                bed_lines_zip = ['track name="EpiChronos_DMRs" description="Differentially Methylated Regions" visibility=2 itemRgb="On"']
                for _, row in dmr_df.to_pandas().iterrows():
                    chrom = row["chrom"]
                    start = int(row["start"])
                    end = int(row["end"])
                    diff = row.get("mean_diff", 0)
                    n_sites = int(row.get("num_sites", 1))
                    score = min(1000, int(abs(diff) * 1000))
                    color = "255,0,0" if diff > 0 else "0,0,255"
                    bed_lines_zip.append(f"{chrom}\t{start}\t{end}\tDMR_{chrom}_{start}_{n_sites}sites\t{score}\t.\t{start}\t{end}\t{color}")
                zf.writestr("bed/dmrs.bed", "\n".join(bed_lines_zip))
            
            # HTML report
            zf.writestr("report/epichronos_dashboard.html", html_data)
            
            # Excel workbook
            try:
                xlsx_zip_buf = io.BytesIO()
                import pandas as pd
                with pd.ExcelWriter(xlsx_zip_buf, engine="openpyxl") as writer:
                    dataset.beta_df.to_pandas().to_excel(writer, sheet_name="Beta_Matrix", index=False)
                    if dml_df.height > 0:
                        dml_df.to_pandas().to_excel(writer, sheet_name="DML_Results", index=False)
                    if dmr_df.height > 0:
                        dmr_df.to_pandas().to_excel(writer, sheet_name="DMR_Results", index=False)
                    if clock_df.height > 0:
                        clock_df.to_pandas().to_excel(writer, sheet_name="Clock_Ages", index=False)
                    if decon_df is not None:
                        decon_df.to_pandas().to_excel(writer, sheet_name="Cell_Deconvolution", index=False)
                xlsx_zip_buf.seek(0)
                zf.writestr("excel/epichronos_results.xlsx", xlsx_zip_buf.getvalue())
            except ImportError:
                pass
            
            # AI report (if generated)
            if "ai_report_text" in st.session_state:
                zf.writestr("report/ai_discussion_draft.md", st.session_state.ai_report_text)
            
            # README
            readme_text = f"""# EpiChronos Analysis Results
Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Contents
- csv/           — Individual result tables
- bed/           — BED files for UCSC Genome Browser / IGV
- excel/         — Complete Excel workbook with all sheets
- report/        — Interactive HTML dashboard + AI discussion draft

## Cohort Summary
- CpG Sites: {dataset.shape[0]:,}
- Samples: {len(dataset.samples)}
- Groups: {', '.join(dataset.get_groups().keys())}

## Software
EpiChronos v0.1.0 — https://github.com/[username]/epichronos
"""
            zf.writestr("README.txt", readme_text)
        
        zip_buffer.seek(0)
        st.download_button(
            label="📦 Download ALL Results (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"epichronos_results_{datetime.datetime.now().strftime('%Y%m%d')}.zip",
            mime="application/zip",
            use_container_width=True
        )
        
        # Clear/Reset button
        st.write("---")
        if st.button("🗑️ Reset and Load New Cohort"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

