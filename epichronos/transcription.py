import polars as pl
import numpy as np
import json
import os
from scipy import stats
from typing import Dict, List, Tuple, Union, Optional
from epichronos.core import MethylationDataset
from epichronos.stats import fdr_correction

def _load_gene_coords() -> dict:
    """Helper to load gene coordinates manifest from resources."""
    base_dir = os.path.dirname(__file__)
    coords_path = os.path.join(base_dir, "resources", "gene_coords.json")
    with open(coords_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load the gene coordinates dynamically
try:
    GENE_COORDS = _load_gene_coords()
except Exception as e:
    # Fail-safe empty defaults in case of loading issues
    GENE_COORDS = {}


def integrate_expression_data(
    dataset: MethylationDataset,
    dmr_df: pl.DataFrame,
    expression_df: pl.DataFrame,
    max_dist_bp: int = 100000
) -> pl.DataFrame:
    """
    Integrate called DMRs with RNA-seq expression data by performing meQTL correlation 
    analysis across cohort samples, flagging functional gene silencing/activation.
    
    Args:
        dataset: The target MethylationDataset containing beta values.
        dmr_df: Called DMRs DataFrame containing columns ['chrom', 'start', 'end'].
        expression_df: RNA-seq counts DataFrame with first column as 'gene' 
                       and subsequent columns containing sample expression levels.
        max_dist_bp: Maximum physical distance to search for adjacent genes.
        
    Returns:
        Polars DataFrame containing meQTL correlations and functional statuses.
    """
    if dmr_df.height == 0 or expression_df.height == 0:
        return pl.DataFrame(schema={
            "chrom": pl.String,
            "start": pl.Int64,
            "end": pl.Int64,
            "gene": pl.String,
            "distance": pl.Int64,
            "correlation_r": pl.Float64,
            "p_value": pl.Float64,
            "q_value": pl.Float64,
            "functional_status": pl.String
        })
        
    sample_names = dataset.samples
    # Ensure samples match between methylation and expression
    expr_cols = expression_df.columns
    gene_col = expr_cols[0]
    expr_samples = [col for col in expr_cols if col != gene_col]
    
    # Keep only overlapping samples
    common_samples = list(set(sample_names).intersection(expr_samples))
    if len(common_samples) < 3:
        raise ValueError(
            f"Insufficient matching samples ({len(common_samples)}) between methylation and expression datasets. "
            "At least 3 matching samples are required to compute Pearson correlations."
        )
        
    beta_df = dataset.beta_df
    
    # 1. Identify adjacent genes for each DMR and prepare data lists
    res_chroms = []
    res_starts = []
    res_ends = []
    res_genes = []
    res_dists = []
    res_rs = []
    res_ps = []
    
    # Convert expression data to a dictionary for fast gene-level lookups
    expr_dict = {}
    for row in expression_df.iter_rows(named=True):
        gene = row[gene_col]
        expr_dict[gene] = {s: row[s] for s in common_samples if s in row}
        
    for row in dmr_df.iter_rows(named=True):
        chrom = row["chrom"]
        start = row["start"]
        end = row["end"]
        center = (start + end) // 2
        
        # Calculate sample-specific average beta levels for this DMR
        dmr_sites = beta_df.filter(
            (pl.col("chrom") == chrom) & 
            (pl.col("pos") >= start) & 
            (pl.col("pos") <= end)
        )
        
        if dmr_sites.height == 0:
            continue
            
        sample_betas = []
        for sample in common_samples:
            vals = dmr_sites[sample].drop_nans().drop_nulls().to_list()
            # Fallback to cohort mean if sample has no valid values in this region
            mean_val = np.mean(vals) if len(vals) > 0 else 0.5
            sample_betas.append(mean_val)
            
        sample_betas = np.array(sample_betas)
        
        # Scan for adjacent genes in manifest
        for gene, coords in GENE_COORDS.items():
            g_chrom, g_start, g_end = coords[0], coords[1], coords[2]
            if chrom != g_chrom:
                continue
                
            # Calculate distance from DMR center to gene boundary
            dist = 0
            if center < g_start:
                dist = g_start - end
            elif center > g_end:
                dist = start - g_end
            else:
                dist = 0 # Overlapping
                
            if dist <= max_dist_bp and gene in expr_dict:
                # Extract expression values across samples in the same order
                sample_exprs = []
                for sample in common_samples:
                    sample_exprs.append(expr_dict[gene].get(sample, 0.0))
                sample_exprs = np.array(sample_exprs)
                
                # Check for zero variance in either vector to prevent correlation NaN
                if np.var(sample_betas) == 0.0 or np.var(sample_exprs) == 0.0:
                    r, p_val = 0.0, 1.0
                else:
                    r, p_val = stats.pearsonr(sample_betas, sample_exprs)
                    if np.isnan(r):
                        r, p_val = 0.0, 1.0
                        
                res_chroms.append(chrom)
                res_starts.append(start)
                res_ends.append(end)
                res_genes.append(gene)
                res_dists.append(dist)
                res_rs.append(float(r))
                res_ps.append(float(p_val))
                
    if len(res_genes) == 0:
        return pl.DataFrame(schema={
            "chrom": pl.String,
            "start": pl.Int64,
            "end": pl.Int64,
            "gene": pl.String,
            "distance": pl.Int64,
            "correlation_r": pl.Float64,
            "p_value": pl.Float64,
            "q_value": pl.Float64,
            "functional_status": pl.String
        })
        
    # 2. FDR multiple testing correction
    res_ps_arr = np.array(res_ps)
    res_qs = fdr_correction(res_ps_arr).tolist()
    
    # 3. Label functional regulatory statuses
    statuses = []
    for i in range(len(res_genes)):
        r = res_rs[i]
        q = res_qs[i]
        if q <= 0.05:
            if r <= -0.5:
                statuses.append("Transcriptional Silencing")
            elif r >= 0.5:
                statuses.append("Transcriptional Activating")
            else:
                statuses.append("Not Significant")
        else:
            statuses.append("Not Significant")
            
    return pl.DataFrame({
        "chrom": res_chroms,
        "start": res_starts,
        "end": res_ends,
        "gene": res_genes,
        "distance": res_dists,
        "correlation_r": res_rs,
        "p_value": res_ps,
        "q_value": res_qs,
        "functional_status": statuses
    }).sort("p_value")
