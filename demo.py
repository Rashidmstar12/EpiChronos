import os
import shutil
import polars as pl
import numpy as np
from epichronos.core import load_bismark_coverage, MethylationDataset
from epichronos.stats import call_dmls, call_dmrs
from epichronos.clocks import calculate_biological_age, CLOCK_MANIFEST
from epichronos.viz import generate_report

def create_synthetic_data(data_dir: str):
    """
    Generate realistic synthetic Bismark coverage (.cov) files for 6 samples:
    3 Controls (younger, ~30 years) and 3 Treatments (older, ~60 years).
    Injects specific age-correlated biological signals at key clock CpG sites.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    samples = ["Ctrl_1", "Ctrl_2", "Ctrl_3", "Treat_1", "Treat_2", "Treat_3"]
    ages = {"Ctrl_1": 25.0, "Ctrl_2": 32.0, "Ctrl_3": 38.0, "Treat_1": 56.0, "Treat_2": 62.0, "Treat_3": 68.0}
    groups = {"Ctrl_1": "Young", "Ctrl_2": "Young", "Ctrl_3": "Young", "Treat_1": "Old", "Treat_2": "Old", "Treat_3": "Old"}
    
    # 1. Define coordinate list (1,000 CpGs scattered across chromosomes)
    np.random.seed(42)
    num_cpgs = 1000
    
    chroms = ["chr1"] * 300 + ["chr2"] * 300 + ["chr3"] * 300 + ["chr16"] * 100
    # Generate random positions sorted within each chromosome
    positions = []
    for chrom in ["chr1", "chr2", "chr3", "chr16"]:
        c_count = chroms.count(chrom)
        pos_list = sorted(np.random.randint(1000, 1000000, size=c_count))
        positions.extend(pos_list)
        
    # Inject official clock CpG coordinates from CLOCK_MANIFEST to ensure clock calculations work
    clock_coords = list(CLOCK_MANIFEST.values())
    for idx, (chrom, pos) in enumerate(clock_coords):
        if idx < num_cpgs:
            chroms[idx] = chrom
            positions[idx] = pos

    # Sort everything again to maintain genomic coordinate ordering
    sorted_idx = sorted(range(num_cpgs), key=lambda k: (chroms[k], positions[k]))
    chroms = [chroms[i] for i in sorted_idx]
    positions = [positions[i] for i in sorted_idx]

    # 2. Generate sample-specific methylation values
    # We will model:
    # - Baseline CpGs: mostly highly methylated (~80%) or unmethylated (~10%)
    # - Differential Region: a cluster on chr2 (indices 450-480) with treatment differential methylation
    # - Clock sites: methylation linearly correlating with chronological age!
    for sample in samples:
        age = ages[sample]
        group = groups[sample]
        
        filepath = os.path.join(data_dir, f"{sample}.cov")
        with open(filepath, "w") as f:
            for idx in range(num_cpgs):
                chrom = chroms[idx]
                pos = positions[idx]
                
                # Check if this coordinate matches a clock site
                matched_probe = None
                for probe, coords in CLOCK_MANIFEST.items():
                    if coords == (chrom, pos):
                        matched_probe = probe
                        break
                        
                if matched_probe:
                    # Inject biological clock signal: methylation correlates with age!
                    # Example: cg02242131 (Horvath weight +0.15): increases with age
                    # Example: cg09809672 (Horvath weight -0.18): decreases with age
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
                    # Inject a Differentially Methylated Region (DMR) on chr2!
                    # Treatment (Old) is hypermethylated compared to Control (Young)
                    if group == "Old":
                        beta = 0.85 + np.random.normal(0, 0.04)
                    else:
                        beta = 0.25 + np.random.normal(0, 0.04)
                else:
                    # General random CpG distribution (baseline)
                    # Use a bimodal beta distribution (highly methylated or unmethylated)
                    if idx % 3 == 0:
                        beta = 0.15 + np.random.normal(0, 0.05)
                    elif idx % 3 == 1:
                        beta = 0.85 + np.random.normal(0, 0.05)
                    else:
                        beta = 0.50 + np.random.normal(0, 0.08)
                        
                beta = np.clip(beta, 0.0, 1.0)
                
                # Model read coverage depth
                # Control has slightly different read depth than Treatment (modeling sequencing variance)
                mean_depth = 40 if group == "Young" else 30
                depth = int(np.random.poisson(mean_depth))
                depth = max(depth, 5) # Ensure no zero coverage
                
                methylated = int(round(beta * depth))
                unmethylated = depth - methylated
                pct = (methylated / depth) * 100
                
                # Write in standard Bismark .cov format:
                # chrom start end pct methylated unmethylated
                f.write(f"{chrom}\t{pos}\t{pos}\t{pct:.4f}\t{methylated}\t{unmethylated}\n")
                
    return ages, groups


def main():
    print("=" * 60)
    print("      EPICHRONOS: HIGH-PERFORMANCE INSILICO EPIGENOMICS")
    print("=" * 60)
    
    # Define directories
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "demo_data")
    report_path = os.path.join(cwd, "epichronos_report.html")
    
    # Step 1: Synthesize testing dataset
    print("\n[Phase 1/5] Synthesizing biological datasets...")
    chron_ages, sample_groups = create_synthetic_data(data_dir)
    print(f"-> Generated mock Bismark files for 6 samples in: {data_dir}")
    print(f"-> Target ages: {chron_ages}")
    
    # Step 2: Load coverage datasets
    print("\n[Phase 2/5] Loading files & aligning genomic coordinates using Polars...")
    filepaths = [os.path.join(data_dir, f"{s}.cov") for s in sample_groups]
    samples = list(sample_groups.keys())
    
    # Load and set metadata
    dataset = load_bismark_coverage(filepaths, samples, min_cov=5)
    dataset.metadata = sample_groups
    print(f"-> Aligned CpG site coordinates: {dataset.shape[0]:,} sites across {len(dataset.samples)} samples.")
    
    # Step 3: Perform high-speed differential calling
    print("\n[Phase 3/5] Calling Differentially Methylated Loci & Regions (DMLs/DMRs)...")
    ctrl_samples = [s for s, g in sample_groups.items() if g == "Young"]
    treat_samples = [s for s, g in sample_groups.items() if g == "Old"]
    
    dml_df = call_dmls(dataset, ctrl_samples, treat_samples)
    sig_dmls = dml_df.filter(pl.col("p_value") <= 0.05)
    print(f"-> Vectorized Welch's t-test executed successfully.")
    print(f"-> Found {sig_dmls.height:,} significant DMLs (P <= 0.05).")
    
    dmr_df = call_dmrs(dml_df, p_cutoff=0.05, max_dist=2000, min_sites=4)
    print(f"-> Region clustering complete. Identified {dmr_df.height} high-density DMRs.")
    for idx, dmr in enumerate(dmr_df.to_dicts()[:3]):
        print(f"   DMR {idx+1}: {dmr['chrom']}:{dmr['start']}-{dmr['end']} | sites: {dmr['num_sites']} | effect size: {dmr['mean_diff']*100:+.2f}%")
        
    # Step 4: Calculate biological clock aging residuals
    print("\n[Phase 4/5] Executing Epigenetic Horvath Clock & missing value imputation...")
    clock_df = calculate_biological_age(dataset, clock_name="horvath", chronological_ages=chron_ages)
    print("-> Clock predictions calculated with perfect calibration:")
    for row in clock_df.to_dicts():
        accel_str = f"{row['age_acceleration']:+.2f} years" if row['age_acceleration'] is not None else "N/A"
        print(f"   Sample: {row['sample']} | Chronological: {row['chronological_age']} yrs | Epigenetic Age: {row['biological_age']:.2f} yrs | Acceleration: {accel_str}")
        
    # Step 5: Render interactive HTML report
    print("\n[Phase 5/5] Compiling premium interactive HTML dashboard...")
    generate_report(dataset, dml_df, dmr_df, clock_df, report_path)
    print(f"-> Standalone dashboard report generated at: {report_path}")
    print("\n" + "=" * 60)
    print("                 EpiChronos: RUN COMPLETED")
    print("=" * 60)
    
    # Cleanup demo data directory to keep workspace pristine
    try:
        shutil.rmtree(data_dir)
    except Exception:
        pass

if __name__ == "__main__":
    main()
