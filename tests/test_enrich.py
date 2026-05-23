import pytest
import polars as pl
from epichronos.enrich import annotate_dmrs_to_genes, perform_pathway_enrichment, GENE_MANIFEST

def test_annotate_dmrs_to_genes():
    # Setup mock DMR that resides directly in NFKB1 coordinate region on chr2
    # NFKB1 is chr2:460000-470000
    dmr_df = pl.DataFrame({
        "chrom": ["chr2"],
        "start": [462000],
        "end": [464000],
        "num_sites": [4],
        "mean_diff": [0.60],
        "min_p_value": [0.001],
        "area": [2.4]
    })
    
    annotated = annotate_dmrs_to_genes(dmr_df, max_dist_bp=10000)
    
    assert annotated.height == 1
    assert annotated["gene"][0] == "NFKB1"
    assert annotated["distance"][0] == 0  # Direct overlap


def test_perform_pathway_enrichment():
    # If our list of genes has high representation of NFKB1, IL6, TNF,
    # the "Inflammatory Response" pathway should be highly significant
    target_genes = ["NFKB1", "IL6", "TNF", "MTOR"]
    
    enrich_res = perform_pathway_enrichment(target_genes, genome_background_size=1000)
    
    # Check that results exist
    assert enrich_res.height > 0
    
    # "Inflammatory Response" pathway has NFKB1, IL6, TNF as members.
    # Our target list has 3/3 of them, so it should be highly enriched (very small p-value)
    infl_row = enrich_res.filter(pl.col("pathway").str.contains("Inflammatory"))
    assert infl_row.height == 1
    assert infl_row["overlap_count"][0] == 3
    assert infl_row["p_value"][0] < 0.01


def test_perform_pathway_enrichment_expanded():
    # Verify that the database loaded a large pathway list
    from epichronos.enrich import PATHWAY_DATABASE, GENE_MANIFEST
    assert len(PATHWAY_DATABASE) >= 20  # Expanded standard pathways loaded
    assert len(GENE_MANIFEST) >= 140    # Expanded genes loaded
    
    # Target list has apoptosis markers: TP53, BCL2, BAX, CASP3
    target_genes = ["TP53", "BCL2", "BAX", "CASP3"]
    enrich_res = perform_pathway_enrichment(target_genes, genome_background_size=1000)
    
    # Check that Apoptosis is highly enriched
    apop_row = enrich_res.filter(pl.col("pathway").str.contains("Apoptosis"))
    assert apop_row.height == 1
    assert apop_row["overlap_count"][0] == 4
    assert apop_row["p_value"][0] < 0.005
