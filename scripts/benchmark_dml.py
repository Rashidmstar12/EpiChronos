import polars as pl
import numpy as np
import time
import tracemalloc
import os
from epichronos.core import MethylationDataset
from epichronos.stats import call_dmls

def main():
    print("=" * 60)
    print("           EPICHRONOS DML PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # 1. Setup synthetic dataset parameters
    num_cpgs = 500000
    samples_a = ["A1", "A2", "A3"]
    samples_b = ["B1", "B2", "B3"]
    samples = samples_a + samples_b
    
    print(f"Generating synthetic dataset with {num_cpgs:,} CpGs across {len(samples)} samples...")
    
    np.random.seed(42)
    
    # Generate coordinates
    chroms = ["chr1"] * 200000 + ["chr2"] * 200000 + ["chr3"] * 100000
    positions = []
    for chrom in ["chr1", "chr2", "chr3"]:
        c_count = chroms.count(chrom)
        positions.extend(sorted(np.random.randint(1, 100000000, size=c_count)))
        
    # Generate beta values: Group A has baseline 0.5, Group B has 0.5 except for 5% of CpGs where we inject DMLs (+0.3)
    data = {
        "chrom": chroms,
        "pos": positions
    }
    
    # Baseline Group A
    for s in samples_a:
        data[s] = np.clip(0.5 + np.random.normal(0, 0.05, size=num_cpgs), 0.0, 1.0)
        
    # Group B with injected DMLs
    # Inject DMLs at index 10,000 to 35,000 (25,000 loci)
    dml_indices = np.arange(10000, 35000)
    for s in samples_b:
        beta = 0.5 + np.random.normal(0, 0.05, size=num_cpgs)
        beta[dml_indices] += 0.3 # Strong hypermethylation in Group B
        data[s] = np.clip(beta, 0.0, 1.0)
        
    beta_df = pl.DataFrame(data)
    dataset = MethylationDataset(beta_df)
    
    print("Dataset generation complete. Starting benchmark trials...")
    
    times = []
    peak_mems = []
    dml_counts = []
    
    for trial in range(1, 4):
        print(f"\nTrial {trial}/3:")
        
        # Start tracking memory
        tracemalloc.start()
        tracemalloc.reset_peak()
        
        # Start timer
        start_time = time.perf_counter()
        
        # Run differential calling
        dml_df = call_dmls(dataset, samples_a, samples_b)
        
        # End timer
        end_time = time.perf_counter()
        
        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        elapsed = end_time - start_time
        peak_mb = peak / (1024 * 1024)
        
        # Count significant DMLs (q-value < 0.05)
        sig_dmls = dml_df.filter(pl.col("q_value") < 0.05).height
        
        times.append(elapsed)
        peak_mems.append(peak_mb)
        dml_counts.append(sig_dmls)
        
        print(f"  Elapsed Time: {elapsed:.4f} seconds")
        print(f"  Peak Memory:  {peak_mb:.2f} MB")
        print(f"  Significant DMLs (Q < 0.05): {sig_dmls:,}")
        
    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    mean_mem = np.mean(peak_mems)
    std_mem = np.std(peak_mems)
    
    print("\n" + "=" * 60)
    print("                    BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Loci Processed: {num_cpgs:,} CpG sites")
    print(f"Execution Time: {mean_time:.4f} ± {std_time:.4f} seconds")
    print(f"Peak Memory:    {mean_mem:.2f} ± {std_mem:.2f} MB")
    print(f"Average DMLs:   {int(np.mean(dml_counts)):,} called significant (Q < 0.05)")
    print("=" * 60)

if __name__ == "__main__":
    main()
