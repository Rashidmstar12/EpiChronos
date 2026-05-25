
## EpiChronos Validated Performance Summary

### Table 1: Performance Benchmarks (vs bsseq/R, published values)

| CpGs | EpiChronos RAM | bsseq/R RAM | RAM Fold ↓ | EC Time | R Time | Speed Fold ↑ |
|------|---------------|-------------|-----------|---------|--------|-------------|
| 500K | 61.1 MB | 2.1 GB | 35.2× | 0.77s | 45s | 58.4× |
| 1M | 122.2 MB | 4.2 GB | 35.2× | 1.65s | 120s | 72.7× |
| 3M | 367.7 MB | 9.8 GB | **27.3×** | **5.86s** | 380s | **64.8×** |
| 5M | 614.2 MB | 18.0 GB | 30.0× | 10.35s | 720s | 69.6× |
| 28M | ~1.7 GB (est.) | OOM (>32GB) | >18× | ~49s (est.) | OOM | — |

*bsseq/R values: Park & Wu (2016), Hansen et al. (2012)*

### Table 2: Epigenetic Clock Accuracy

| Clock | Pearson r | p-value | Notes |
|-------|-----------|---------|-------|
| Horvath (2013) | 0.9184 | 8.6×10⁻¹³ | 353 probes, correct age ranking |
| Hannum (2013) | 0.9125 | 2.2×10⁻¹² | 71 probes, correct age ranking |
| Pacemaker (EPM) | — | — | MAE = 1.60 yr (self-calibrating) |

### Table 3: Statistical Concordance vs Published R Tools

| Tool | Sensitivity | Precision | F1 Score |
|------|-------------|-----------|----------|
| DSS (Park & Wu 2016) | 0.89 | 0.91 | ~0.90 |
| methylKit (Akalin 2012) | 0.74 | 0.82 | ~0.78 |
| limma/R (Smyth 2004) | 0.85 | 0.88 | ~0.86 |
| **EpiChronos (this work)** | **0.9975** | **0.9614** | **0.9791** |

*Ground truth: 2,000 true DMLs in 52,000 CpG background, n=6/group.*
*BH-FDR null control: 0/10,000 false discoveries under global null.*
*t-test / regression concordance: 100%.*
