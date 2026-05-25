---
title: 'EpiChronos: A High-Performance Python Suite for Unified DNA Methylation and Epigenetic Aging Analysis'
tags:
  - Python
  - DNA Methylation
  - Epigenetics
  - Biological Aging
  - Epigenetic Clock
  - Polars
authors:
  - name: Rashid Kadayil
    orcid: 0009-0009-6398-4557
    affiliation: 1
affiliations:
  - index: 1
    name: Department of Biotechnology, Pondicherry University, Puducherry, India
date: 24 May 2026
bibliography: paper.bib
---
# Summary

`EpiChronos` is a high-performance, open-source Python library designed for unified downstream DNA methylation and biological aging analysis. Architected on a multi-threaded, memory-efficient data engine leveraging Apache Arrow and Polars, the library scales seamlessly to large clinical cohorts on standard consumer hardware. It directly parses standard input formats across diverse platforms, including Bismark Whole Genome Bisulfite Sequencing (WGBS) coverage files, Illumina 450K/EPIC microarray beta matrices, and long-read sequencing outputs from Oxford Nanopore and PacBio. From these inputs, `EpiChronos` provides vectorized calling of differentially methylated loci (DML) and clustered regions (DMR), estimates immune cell-type fractions, predicts biological age using assembly-aware Horvath [@horvath2013dna], Hannum [@Hannum2013GenomewideMP], and non-linear Epigenetic Pacemaker [@snir2016statistical] clocks, and maps transcriptomic correlations (eQTM) alongside pathway enrichment. All analyses are compiled into 100% self-contained, interactive HTML reports for immediate quality control and scientific sharing.

# Statement of Need

Bioinformatics workflows for DNA methylation frequently suffer from the "memory wall" of traditional R packages. Traditional R-Bioconductor tools such as bsseq and minfi can require substantial RAM when processing whole-genome bisulfite sequencing datasets of 25–30 million CpGs, creating barriers for researchers without access to high-memory servers. For reference, processing a typical 450K array cohort (n=100, ~480,000 CpGs) in R using \`minfi\` can require several gigabytes of RAM for large cohorts, while EpiChronos processes 500,000 CpGs using only 61 MB. `EpiChronos` addresses this bottleneck by providing a native Python suite that aligns multi-platform coordinate datasets in contiguous Arrow memory buffers, enabling desktop-level processing of whole-genome profiles. Furthermore, `EpiChronos` bridges the gap between raw methylation values and downstream multi-omics interpretation by integrating epigenetic aging predictions, immune deconvolution, and transcriptomic linkages into a unified suite.

# Mathematical & Implementation Foundations

`EpiChronos` implements a highly optimized, modular pipeline written in pure Python and compiled Rust-backed data frames. The core scientific algorithms are detailed below:

## Data Engine & Vectorized Differential Methylation
Rather than relying on legacy row-oriented Python objects, `EpiChronos` utilizes a Rust-backed columnar Polars engine. Cytosine methylation data are stored in contiguous Apache Arrow memory buffers, allowing vectorized computation of differential methylation. 

For cohort comparisons, `EpiChronos` implements a vectorized Welch's $t$-test with Satterthwaite degrees of freedom that calculates site-specific statistics without assuming equal variances. The test statistic $t$ for each CpG site is defined as:

$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{N_1} + \frac{s_2^2}{N_2}}}$$

with Welch–Satterthwaite degrees of freedom $\nu$ dynamically computed for every locus. False Discovery Rate (FDR) corrections are applied using vectorized Benjamini–Hochberg procedures.

## Cell-Type Deconvolution & Epigenetic Clocks
To isolate confounding cell-type shifts in heterogeneous tissue (such as peripheral blood), `EpiChronos` implements a constrained projection solver based on the Houseman algorithm [@Houseman2012DNAMA]. For a sample methylation vector $\mathbf{y}$ and reference matrix $\mathbf{M}$ compiled from purified blood cell fractions [@reinius2012differential], the cell-type weight vector $\mathbf{w}$ is estimated via constrained OLS using a precomputed Moore-Penrose pseudo-inverse of the Reinius et al. reference panel:

$$\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{M}\mathbf{w}\|^2_2 \quad \text{subject to} \quad w_k \ge 0, \sum_k w_k = 1$$

Biological age is predicted using assembly-aware Horvath [@horvath2013dna] and Hannum [@Hannum2013GenomewideMP] clocks with optional GRCh38 coordinate translation via pyliftover. For non-linear aging dynamics, we implement the Epigenetic Pacemaker (EPM) framework [@snir2016statistical], which models the methylation state $\hat{y}_{ij}$ of CpG site $j$ in sample $i$ using alternating coordinate descent to solve:

$$\hat{y}_{ij} = f(t_i; \mathbf{a}_j)$$

where $t_i$ represents the epigenetic state of sample $i$, and $\mathbf{a}_j$ represents site-specific parameters.

## Multi-Omics Transcription Linkage (eQTM) & Pathway ORA
`EpiChronos` bridges epigenetics and transcriptomics through expression quantitative trait methylation (eQTM) analysis. The library correlates called Differentially Methylated Regions (DMRs) with matching sample RNA-seq expression vectors to assess functional status (such as promoter hypermethylation leading to gene silencing). Downstream pathways are analyzed via Hypergeometric Overrepresentation Analysis (ORA) against MSigDB Hallmark gene sets [@Liberzon2015] using:

$$P(X \ge k) = \sum_{i=k}^{n} \frac{\binom{K}{i} \binom{N-K}{n-i}}{\binom{N}{n}}$$

where $N$ is the total size of the RefSeq coordinates database, $K$ is the size of the pathway gene set, $n$ is the number of target genes, and $k$ is the overlap size.

# Computational Performance

On a consumer Windows machine (Python 3.12, Polars 1.41.0), `EpiChronos` processes 500,000 CpG sites across 6 samples in 0.28 ± 0.03 seconds using 61 MB of peak RAM. The reproducible benchmark script is available at scripts/benchmark_dml.py.


# Validation Results

EpiChronos v0.1.2 was validated across four independent benchmarks:

**Performance:** At 3 million CpGs (WGBS whole-genome scale), EpiChronos required
367.7 MB RAM and completed in 5.86 seconds, representing a **27.3× reduction in
memory** and **64.8× improvement in speed** compared to bsseq/R pipelines
(Park & Wu, 2016; Hansen et al., 2012).

**Statistical accuracy:** On a controlled ground-truth dataset (2,000 true DMLs
in 52,000 CpG background, n=6/group), EpiChronos achieved sensitivity=0.9975,
precision=0.9614, F1=0.9791 — exceeding DSS (F1≈0.90), methylKit (F1≈0.78),
and limma (F1≈0.86). BH-FDR was correctly calibrated with 0 false discoveries
under the global null. Regression and t-test modes showed 100% concordance.

**Clock accuracy:** Horvath clock (353 probes) and Hannum clock (71 probes)
achieved Pearson r=0.918 and r=0.913 respectively against chronological age
(p<10⁻¹²). The Epigenetic Pacemaker achieved MAE=1.60 years on held-out samples.

**Scalability:** EpiChronos scales linearly from 100K to 5M+ CpGs
(12.2 MB → 614.2 MB RAM) without memory fragmentation, confirmed on
simulated WGBS-scale datasets representing the full hg19 CpG landscape.

# References
