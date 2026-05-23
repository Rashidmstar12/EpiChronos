"""
EpiChronos: A High-Performance, Unified Downstream DNA Methylation & Biological Aging Analysis Suite.
"""

__version__ = "0.2.0"

from epichronos.core import (
    MethylationDataset, 
    load_bismark_coverage, 
    load_array_beta,
    load_nanopore_modkit,
    load_pacbio_bedgraph
)
from epichronos.stats import call_dmls, call_dmrs
from epichronos.clocks import calculate_biological_age, list_available_clocks
from epichronos.pacemaker import EpigeneticPacemaker
from epichronos.viz import generate_report
from epichronos.enrich import annotate_dmrs_to_genes, perform_pathway_enrichment
from epichronos.ai import EpigeneticCopilot

__all__ = [
    "MethylationDataset",
    "load_bismark_coverage",
    "load_array_beta",
    "load_nanopore_modkit",
    "load_pacbio_bedgraph",
    "call_dmls",
    "call_dmrs",
    "calculate_biological_age",
    "list_available_clocks",
    "EpigeneticPacemaker",
    "generate_report",
    "annotate_dmrs_to_genes",
    "perform_pathway_enrichment",
    "EpigeneticCopilot",
]
