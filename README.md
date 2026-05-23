# EpiChronos

[![PyPI Version](https://img.shields.io/badge/pypi-v0.2.0-blue.svg)](https://pypi.org/project/epichronos/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)

**EpiChronos** is a high-performance, unified downstream DNA methylation and biological aging analysis suite. Written in memory-efficient, multi-threaded Python (built on **Polars** and **NumPy/SciPy**), EpiChronos overcomes the memory barriers and platform fragmentation of traditional R-Bioconductor pipelines, providing a scalable solution that integrates microarray, sequencing (WGBS, RRBS, EM-seq), and long-read methylation data in a single tool.

---

## 🚀 Key Features

*   **Unified Multi-Platform Support**: Read and align standard Bismark `.cov` sequencing files, array-based beta matrices, and coordinate datasets seamlessly into a common coordinate-centric format.
*   **Order-of-Magnitude Performance Gains**: Leverage a fully multi-threaded **Polars** data engine to run DML/DMR calling on millions of cytosines in seconds, bypassing the R-Bioconductor memory wall.
*   **Vectorized Welch's t-test**: Fast statistical comparisons between phenotypic cohorts using highly parallelized matrix algebra.
*   **Epigenetic Aging Clock Engine**: Built-in implementations of Horvath's multi-tissue clock and Hannum's clock, supporting automatic mapping from genomic coordinates to array probe IDs.
*   **Robust Missing-Value Imputation**: High-fidelity cohort-mean and standard public reference-mean imputation for missing CpGs in sparse sequencing samples, resolving a major bottleneck where missing sites cause clock calculations to crash.
*   **Interactive Standalone HTML Reports**: Compile quality control, global PCA projections, Volcano plots of differentially methylated loci (DMLs), and epigenetic age acceleration graphs into a single shareable interactive dashboard.

---

## 📦 Installation

To install EpiChronos in development mode:

```bash
git clone https://github.com/Rashidmstar12/EpiChronos.git
cd EpiChronos
pip install -e .
```

### Dependencies
EpiChronos is built to be extremely lightweight and requires only:
*   `polars >= 0.20.0` (for high-speed lazy-evaluated dataframes)
*   `numpy >= 1.24.0` (for vectorized math)
*   `scipy >= 1.10.0` (for statistical distributions)
*   `plotly >= 5.14.0` (for interactive visualization)
*   `pyarrow >= 12.0.0` (for Arrow memory management)

---

## ⚡ Quick Start

Analyze a full sequencing cohort in under 15 lines of Python:

```python
import epichronos as ec

# 1. Load and align sequencing samples by genomic coordinates
samples = ["Ctrl_1", "Ctrl_2", "Treat_1", "Treat_2"]
filepaths = [f"data/{s}.cov" for s in samples]
metadata = {"Ctrl_1": "Young", "Ctrl_2": "Young", "Treat_1": "Old", "Treat_2": "Old"}

dataset = ec.load_bismark_coverage(filepaths, samples, min_cov=5)
dataset.metadata = metadata

# 2. Call Differentially Methylated Loci & Regions (DMLs / DMRs)
dml_df = ec.call_dmls(dataset, ["Ctrl_1", "Ctrl_2"], ["Treat_1", "Treat_2"])
dmr_df = ec.call_dmrs(dml_df, p_cutoff=0.05, max_dist=1000, min_sites=3)

# 3. Calculate Epigenetic Biological Age (Horvath Clock)
true_ages = {"Ctrl_1": 22.0, "Ctrl_2": 26.0, "Treat_1": 60.0, "Treat_2": 65.0}
clock_df = ec.calculate_biological_age(dataset, clock_name="horvath", chronological_ages=true_ages)

# 4. Export a premium interactive HTML report
ec.generate_report(dataset, dml_df, dmr_df, clock_df, "epichronos_dashboard.html")
```

---

## 💾 Memory Efficiency & RAM Benchmarks

By storing aligned coordinates in memory-efficient **Apache Arrow columnar buffers** via **Polars**, EpiChronos eliminates the boxing overhead of Python objects and the R garbage collector. This enables comprehensive analysis of whole-genome datasets on a standard consumer laptop.

### Estimated RAM Footprint (Single File Ingestion)

*   **Microarray Data (EPIC v2 / EPIC / 450K)** (~930k sites): **~35 MB – 50 MB of RAM**
*   **Reduced Representation Sequencing (RRBS)** (~2M sites): **~80 MB – 120 MB of RAM**
*   **Whole Genome Sequencing (WGBS) / Nanopore** (~28M sites, 1.5 GB file on disk): 
    *   *Unfiltered (Full Genome):* **~1.0 GB – 1.2 GB of RAM**
    *   *With Coverage Filtering (`min_cov=5`):* **~500 MB – 700 MB of RAM**

### 📊 In-Memory Scaling vs. R-Bioconductor
To load and align a single Whole-Genome Bisulfite Sequencing (WGBS) sample (28 million CpGs):

| Pipeline / Tool | Backend | Data Structure | RAM Usage (1 WGBS Sample) |
| :--- | :--- | :--- | :--- |
| **Traditional R** (`bsseq` / `minfi`) | R / S4 Objects | Fragmented boxed vectors | **6.0 GB – 12.0 GB** *(Often hits the memory wall)* |
| **EpiChronos v0.2.0** | Python / Polars / Arrow | Contiguous native Arrow buffers | **0.5 GB – 1.2 GB** *(Order-of-magnitude reduction)* |

---

## 📐 Pipeline Blueprint

```
Raw Methylation Input
  ├── Bismark Coverage (.cov)
  ├── Microarray Beta-Value Matrix
  └── Long-Read bedGraph
       │
       ▼
 epichronos.core.MethylationDataset (Polars-aligned coordinate framework)
  ├── filter_by_coverage()
  ├── filter_by_variance()
  └── impute_missing()
       │
       ├──────────────────────────────┐
       ▼                              ▼
epichronos.stats               epichronos.clocks
  ├── call_dmls()                ├── calculate_biological_age()
  └── call_dmrs()                └── (Cohort & Ref-mean Imputation)
       │                              │
       └──────────────┬───────────────┘
                      ▼
               epichronos.viz
                 └── generate_report() -> Standalone HTML Report
```

---

## 🛡️ License

Distributed under the MIT License. See `LICENSE` for more information.
