import polars as pl
import numpy as np
from scipy import stats
from typing import Dict, List, Set, Tuple, Union, Optional
from epichronos.stats import fdr_correction

import json
import os

def _load_enrichment_resources():
    """Helper to load gene coordinates and pathway database JSON files dynamically."""
    base_dir = os.path.dirname(__file__)
    
    # 1. Load Gene coordinates
    gene_path = os.path.join(base_dir, "resources", "gene_coords.json")
    if os.path.exists(gene_path):
        with open(gene_path, "r", encoding="utf-8") as f:
            gene_manifest = json.load(f)
            # Map list coordinates to tuple coordinates
            gene_manifest = {k: (v[0], int(v[1]), int(v[2])) for k, v in gene_manifest.items()}
    else:
        gene_manifest = {}
        
    # 2. Load Pathway Database
    pathway_path = os.path.join(base_dir, "resources", "pathway_db.json")
    if os.path.exists(pathway_path):
        with open(pathway_path, "r", encoding="utf-8") as f:
            pathway_db = json.load(f)
    else:
        pathway_db = {}
        
    return gene_manifest, pathway_db

try:
    GENE_MANIFEST, PATHWAY_DATABASE = _load_enrichment_resources()
except Exception:
    GENE_MANIFEST, PATHWAY_DATABASE = {}, {}


def annotate_dmrs_to_genes(
    dmr_df: pl.DataFrame,
    max_dist_bp: int = 100000
) -> pl.DataFrame:
    """
    Annotate called DMR regions to adjacent genes in linear genomic coordinates.
    Calculates physical distance from the DMR center to the gene boundaries.
    
    Args:
        dmr_df: Called DMRs DataFrame.
        max_dist_bp: Maximum allowed distance to map a DMR to a gene.
        
    Returns:
        Polars DataFrame containing DMR coordinates aligned with gene annotations.
    """
    if dmr_df.height == 0 or not GENE_MANIFEST:
        return pl.DataFrame(schema={
            "chrom": pl.String,
            "start": pl.Int64,
            "end": pl.Int64,
            "gene": pl.String,
            "distance": pl.Int64
        })
        
    # Build Polars Gene Reference
    gene_df = pl.DataFrame({
        "gene": list(GENE_MANIFEST.keys()),
        "chrom": [v[0] for v in GENE_MANIFEST.values()],
        "g_start": [v[1] for v in GENE_MANIFEST.values()],
        "g_end": [v[2] for v in GENE_MANIFEST.values()]
    })
    
    # Fast Hash Join on Chromosome
    joined = dmr_df.join(gene_df, on="chrom")
    
    # Compute center of DMR
    center = (pl.col("start") + pl.col("end")) // 2
    
    # Compute distance:
    # if center < g_start: dist = g_start - end
    # else if center > g_end: dist = start - g_end
    # else: dist = 0 (overlapping)
    dist_expr = (
        pl.when(center < pl.col("g_start"))
        .then(pl.col("g_start") - pl.col("end"))
        .when(center > pl.col("g_end"))
        .then(pl.col("start") - pl.col("g_end"))
        .otherwise(0)
    )
    
    # Select, filter and sort
    annotated = (
        joined.with_columns(dist_expr.alias("distance"))
        .filter(pl.col("distance") <= max_dist_bp)
        .select(["chrom", "start", "end", "gene", "distance"])
        .sort(["chrom", "start"])
    )
    
    return annotated


def perform_pathway_enrichment(
    target_genes: List[str],
    genome_background_size: int = 20000
) -> pl.DataFrame:
    """
    Perform high-speed hypergeometric overrepresentation analysis (ORA) 
    to identify significantly enriched biological pathways.
    
    Hypergeometric Math:
    M = total genes in genome (background size)
    n = total genes in pathway (successes in population)
    N = target genes called adjacent to DMRs (sample size)
    k = genes in target list that overlap pathway (successes in sample)
    
    p-value = cumulative probability of observing >= k successes
    """
    unique_targets = list(set(target_genes))
    N = len(unique_targets)
    
    if N == 0:
        return pl.DataFrame(schema={
            "pathway": pl.String,
            "pathway_size": pl.Int64,
            "overlap_genes": pl.String,
            "overlap_count": pl.Int64,
            "p_value": pl.Float64,
            "q_value": pl.Float64
        })
        
    pathway_names = []
    pathway_sizes = []
    overlaps = []
    overlap_counts = []
    p_values = []
    
    for pathway, path_genes in PATHWAY_DATABASE.items():
        # Identify overlap
        overlap_set = set(unique_targets).intersection(path_genes)
        k = len(overlap_set)
        n = len(path_genes)
        M = genome_background_size
        
        # Hypergeometric survival function: sf(k-1) = P(X >= k)
        # sf is 1 - cdf, which is exactly the ORA p-value!
        if k > 0:
            # We subtract 1 because sf is strict inequality (P(X > k-1) = P(X >= k))
            p_val = stats.hypergeom.sf(k - 1, M, n, N)
        else:
            p_val = 1.0
            
        pathway_names.append(pathway)
        pathway_sizes.append(n)
        overlaps.append(", ".join(sorted(list(overlap_set))))
        overlap_counts.append(k)
        p_values.append(p_val)
        
    p_values = np.array(p_values)
    q_values = fdr_correction(p_values)
    
    res_df = pl.DataFrame({
        "pathway": pathway_names,
        "pathway_size": pathway_sizes,
        "overlap_genes": overlaps,
        "overlap_count": overlap_counts,
        "p_value": p_values,
        "q_value": q_values
    })
    
    # Sort by significance
    return res_df.sort("p_value")
