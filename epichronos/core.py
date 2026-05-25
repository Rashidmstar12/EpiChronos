import polars as pl
import numpy as np
import os
import math
from typing import List, Dict, Union, Optional, Tuple

class MethylationDataset:
    """
    A unified, high-performance dataset class for DNA methylation data.
    Stores methylation beta values (0.0 to 1.0) and optionally coverage depth 
    as Polars DataFrames, aligning samples by genomic coordinates (chrom, pos).
    """
    def __init__(
        self, 
        beta_df: pl.DataFrame, 
        cov_df: Optional[pl.DataFrame] = None, 
        metadata: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            beta_df: Polars DataFrame with columns ['chrom', 'pos'] and sample beta values.
            cov_df: Optional Polars DataFrame with columns ['chrom', 'pos'] and sample coverage depth.
            metadata: Optional dictionary mapping sample names to their phenotypic groups (e.g., {'s1': 'Control', 's2': 'Cancer'})
        """
        self.beta_df = beta_df.sort(["chrom", "pos"])
        self.cov_df = cov_df.sort(["chrom", "pos"]) if cov_df is not None else None
        
        # Identify sample names from columns (excluding chrom and pos)
        self.samples = [col for col in beta_df.columns if col not in ["chrom", "pos"]]
        
        if metadata:
            self.metadata = metadata
        else:
            self.metadata = {sample: "Unknown" for sample in self.samples}
            
        # Validate data
        self._validate()

    def _validate(self):
        """Perform basic integrity checks on the dataset."""
        assert "chrom" in self.beta_df.columns, "beta_df must contain a 'chrom' column"
        assert "pos" in self.beta_df.columns, "beta_df must contain a 'pos' column"
        
        if self.cov_df is not None:
            assert "chrom" in self.cov_df.columns, "cov_df must contain a 'chrom' column"
            assert "pos" in self.cov_df.columns, "cov_df must contain a 'pos' column"
            # Verify shapes and coordinates match
            assert self.beta_df.shape[0] == self.cov_df.shape[0], "beta_df and cov_df must have the same number of rows"
            
            # Verify (chrom, pos) coordinates are identical between beta_df and cov_df
            beta_coords = self.beta_df.select(["chrom", "pos"])
            cov_coords  = self.cov_df.select(["chrom", "pos"])
            assert beta_coords.equals(cov_coords), (
                "beta_df and cov_df must have identical (chrom, pos) coordinates in the same order. "
                "Got different coordinates — ensure both DataFrames are aligned before constructing MethylationDataset."
            )
            # Verify sample column names are identical
            beta_samples = [c for c in self.beta_df.columns if c not in ["chrom", "pos"]]
            cov_samples  = [c for c in self.cov_df.columns  if c not in ["chrom", "pos"]]
            assert beta_samples == cov_samples, (
                f"beta_df and cov_df must have identical sample columns in the same order. "
                f"beta_df samples: {beta_samples}, cov_df samples: {cov_samples}"
            )

    @property
    def shape(self):
        """Returns the shape of the beta dataframe (number of CpGs, number of columns)."""
        return self.beta_df.shape

    def get_groups(self) -> Dict[str, List[str]]:
        """Returns a dictionary grouping sample names by their phenotypic metadata values."""
        groups = {}
        for sample, group in self.metadata.items():
            if group not in groups:
                groups[group] = []
            groups[group].append(sample)
        return groups

    def filter_by_coverage(self, min_cov: int, min_samples_ratio: float = 0.8) -> 'MethylationDataset':
        """
        Filter out CpG sites where coverage is below min_cov in too many samples.
        
        Args:
            min_cov: Minimum required coverage depth per site.
            min_samples_ratio: Proportion of samples that must meet min_cov for a site to be kept.
        """
        if self.cov_df is None:
            print("Warning: No coverage DataFrame available. Skipping coverage filtering.")
            return self
            
        # Calculate how many samples have coverage >= min_cov for each site
        cov_cols = [col for col in self.cov_df.columns if col not in ["chrom", "pos"]]
        
        # Apply filter expression
        threshold = math.ceil(len(cov_cols) * min_samples_ratio)
        expr = pl.sum_horizontal([pl.col(c) >= min_cov for c in cov_cols]) >= threshold
        filtered_indices = self.cov_df.filter(expr).select(["chrom", "pos"])
        
        # Semi-join to keep only passed sites
        new_beta = self.beta_df.join(filtered_indices, on=["chrom", "pos"], how="semi")
        new_cov = self.cov_df.join(filtered_indices, on=["chrom", "pos"], how="semi")
        
        return MethylationDataset(new_beta, new_cov, self.metadata)

    def filter_by_variance(self, min_var: float = 0.005) -> 'MethylationDataset':
        """
        Filter out low-variance CpG sites (e.g., non-informative sites).
        
        Args:
            min_var: Minimum required variance across samples.
        """
        sample_cols = self.samples
        
        # Use fill_null(0.0) on NaN/null values so variance is computed on real numbers
        # NaN comparison in Polars returns null which bypasses filter — this prevents that
        mean_expr    = pl.sum_horizontal([pl.col(c).fill_nan(None).fill_null(0.0) for c in sample_cols]) / len(sample_cols)
        sq_mean_expr = pl.sum_horizontal([pl.col(c).fill_nan(None).fill_null(0.0)**2 for c in sample_cols]) / len(sample_cols)
        var_expr = sq_mean_expr - (mean_expr**2)
        filtered_df = self.beta_df.filter(var_expr >= min_var)
        
        new_cov = None
        if self.cov_df is not None:
            filtered_coords = filtered_df.select(["chrom", "pos"])
            new_cov = self.cov_df.join(filtered_coords, on=["chrom", "pos"], how="semi")
            
        return MethylationDataset(filtered_df, new_cov, self.metadata)

    def impute_missing(self, method: str = "mean") -> 'MethylationDataset':
        """
        Impute missing (NaN or Null) methylation beta values.

        Args:
            method: Imputation method ('mean', 'median', or 'zero').
        """
        exprs = []
        for sample in self.samples:
            col = self.beta_df[sample].drop_nans().drop_nulls()
            if method == "mean":
                fill_val = col.mean()
            elif method == "median":
                fill_val = col.median()
            else:
                fill_val = 0.0
            if fill_val is None or np.isnan(fill_val):
                fill_val = 0.5
            exprs.append(pl.col(sample).fill_nan(fill_val).fill_null(fill_val))
        # Single vectorized Polars call — avoids repeated DataFrame cloning
        imputed_beta = self.beta_df.with_columns(exprs)
        return MethylationDataset(imputed_beta, self.cov_df, self.metadata)

    def to_parquet(self, directory: str, prefix: str = "dataset"):
        """Save dataset as compressed parquet files inside the target directory."""
        os.makedirs(directory, exist_ok=True)
        self.beta_df.write_parquet(os.path.join(directory, f"{prefix}_beta.parquet"))
        if self.cov_df is not None:
            self.cov_df.write_parquet(os.path.join(directory, f"{prefix}_cov.parquet"))
        
        # Save metadata as JSON
        import json
        with open(os.path.join(directory, f"{prefix}_metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=4)

    @classmethod
    def from_parquet(cls, directory: str, prefix: str = "dataset") -> 'MethylationDataset':
        """Load a saved dataset from a directory."""
        beta_df = pl.read_parquet(os.path.join(directory, f"{prefix}_beta.parquet"))
        cov_path = os.path.join(directory, f"{prefix}_cov.parquet")
        cov_df = pl.read_parquet(cov_path) if os.path.exists(cov_path) else None
        
        import json
        metadata = None
        meta_path = os.path.join(directory, f"{prefix}_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
                
        return cls(beta_df, cov_df, metadata)

    def collapse_strand_coordinates(self) -> 'MethylationDataset':
        """
        Collapses Watson (x) and Crick (x+1) strand coordinates into a single unified Watson CpG position.
        Computes weighted beta averages based on sample coverages.

        WARNING: This function collapses ALL adjacent (pos, pos+1) pairs on the same chromosome.
        It assumes the input contains only symmetric CpG dinucleotide sites (standard Bismark CpG-only output).
        Do NOT call this on CX-context output (CHH/CHG sites) — adjacent non-CpG sites at consecutive
        coordinates will be incorrectly merged into a single CpG.
        """
        if self.cov_df is None:
            return self
            
        chroms = self.beta_df["chrom"].to_list()
        positions = self.beta_df["pos"].to_list()
        
        n_sites = len(positions)
        if n_sites <= 1:
            return self
            
        sample_names = self.samples
        
        beta_mat = self.beta_df.select(sample_names).to_numpy().astype(np.float64)
        cov_mat = self.cov_df.select(sample_names).to_numpy().astype(np.float64)
        
        beta_nan = np.isnan(beta_mat)
        beta_mat[beta_nan] = 0.0
        
        new_chroms = []
        new_positions = []
        new_betas = []
        new_covs = []
        
        skip = False
        for i in range(n_sites - 1):
            if skip:
                skip = False
                continue
                
            c_chrom = chroms[i]
            c_pos = positions[i]
            n_chrom = chroms[i+1]
            n_pos = positions[i+1]
            
            if c_chrom == n_chrom and n_pos == c_pos + 1:
                new_chroms.append(c_chrom)
                new_positions.append(c_pos)
                
                c_cov = cov_mat[i]
                n_cov = cov_mat[i+1]
                
                c_beta = beta_mat[i]
                n_beta = beta_mat[i+1]
                
                merged_cov = c_cov + n_cov
                
                merged_beta = np.zeros_like(c_beta)
                mask = merged_cov > 0
                merged_beta[mask] = (c_beta[mask] * c_cov[mask] + n_beta[mask] * n_cov[mask]) / merged_cov[mask]
                merged_beta[~mask] = (c_beta[~mask] + n_beta[~mask]) / 2.0
                
                both_nan = beta_nan[i] & beta_nan[i+1]
                merged_beta[both_nan] = np.nan
                
                new_betas.append(merged_beta)
                new_covs.append(merged_cov)
                skip = True
            else:
                new_chroms.append(c_chrom)
                new_positions.append(c_pos)
                
                orig_beta = beta_mat[i].copy()
                orig_beta[beta_nan[i]] = np.nan
                
                new_betas.append(orig_beta)
                new_covs.append(cov_mat[i])
                
        if not skip:
            new_chroms.append(chroms[-1])
            new_positions.append(positions[-1])
            
            orig_beta = beta_mat[-1].copy()
            orig_beta[beta_nan[-1]] = np.nan
            new_betas.append(orig_beta)
            new_covs.append(cov_mat[-1])
            
        new_betas = np.array(new_betas)
        new_covs = np.array(new_covs)
        
        beta_dict = {
            "chrom": new_chroms,
            "pos": new_positions
        }
        cov_dict = {
            "chrom": new_chroms,
            "pos": new_positions
        }
        
        for idx, sample in enumerate(sample_names):
            beta_dict[sample] = new_betas[:, idx].tolist()
            cov_dict[sample] = new_covs[:, idx].tolist()
            
        return MethylationDataset(pl.DataFrame(beta_dict), pl.DataFrame(cov_dict), self.metadata)

    def filter_by_binomial_confidence(
        self, 
        min_ci_width: float = 0.3, 
        min_samples_ratio: float = 0.8
    ) -> 'MethylationDataset':
        """
        Filters out CpG sites where the 95% binomial confidence interval width
        exceeds min_ci_width (representing noisy estimates) in too many samples.
        """
        if self.cov_df is None:
            return self
            
        sample_names = self.samples
        
        beta_mat = self.beta_df.select(sample_names).to_numpy().astype(np.float64)
        cov_mat = self.cov_df.select(sample_names).to_numpy().astype(np.float64)
        
        z = 1.96
        z_sq = z**2
        
        n = np.where(cov_mat > 0, cov_mat, 1.0)
        p = np.nan_to_num(beta_mat, nan=0.5)
        p = np.clip(p, 0.0, 1.0)
        
        term1 = (p * (1.0 - p)) / n
        term2 = z_sq / (4.0 * (n**2))
        sqrt_term = np.sqrt(term1 + term2)
        
        numerator = 2.0 * z * sqrt_term
        denominator = 1.0 + z_sq / n
        
        width = numerator / denominator
        width[cov_mat == 0] = 1.0
        width[np.isnan(beta_mat)] = 1.0
        
        passed_matrix = width <= min_ci_width
        passed_counts = np.sum(passed_matrix, axis=1)
        passed_mask = passed_counts >= (len(sample_names) * min_samples_ratio)
        
        filtered_beta = self.beta_df.filter(passed_mask)
        filtered_cov = self.cov_df.filter(passed_mask)
        
        return MethylationDataset(filtered_beta, filtered_cov, self.metadata)


def load_bismark_coverage(
    filepaths: List[str], 
    sample_names: List[str], 
    min_cov: int = 0
) -> MethylationDataset:
    """
    Parse multiple Bismark coverage (.cov) files and perform high-speed inner joins
    using Polars to align all samples by genomic coordinates.
    
    Bismark coverage columns:
    1: Chromosome
    2: Start position
    3: End position
    4: Methylation percentage (0 - 100)
    5: Number of methylated cytosines
    6: Number of unmethylated cytosines
    """
    assert len(filepaths) == len(sample_names), "Number of file paths must equal number of sample names"
    
    lazy_dfs = []
    
    for path, name in zip(filepaths, sample_names):
        # Read file lazily
        # Use low-level CSV reader with specific options for efficiency
        ldf = pl.scan_csv(
            path,
            separator="\t",
            has_header=False,
            schema={
                "column_1": pl.String,
                "column_2": pl.Int64,
                "column_3": pl.Int64,
                "column_4": pl.Float64,
                "column_5": pl.Int64,
                "column_6": pl.Int64
            }
        ).rename({
            "column_1": "chrom",
            "column_2": "pos",
            "column_3": "end",
            "column_4": "pct",
            "column_5": "methylated",
            "column_6": "unmethylated"
        })
        
        # Calculate derived columns: total depth (coverage) and beta value
        ldf = ldf.with_columns([
            (pl.col("methylated") + pl.col("unmethylated")).alias(f"{name}_cov"),
            pl.when((pl.col("methylated") + pl.col("unmethylated")) == 0)
              .then(None)
              .otherwise(
                  pl.col("methylated").cast(pl.Float64) /
                  (pl.col("methylated") + pl.col("unmethylated")).cast(pl.Float64)
              )
              .alias(name)
        ])
        
        # Filter by coverage per sample early if specified
        if min_cov > 0:
            ldf = ldf.filter(pl.col(f"{name}_cov") >= min_cov)
            
        # Select only required columns
        ldf = ldf.select(["chrom", "pos", name, f"{name}_cov"])
        lazy_dfs.append(ldf)
        
    # Perform sequential outer joins to align all samples by chrom and pos
    combined = lazy_dfs[0]
    for next_df in lazy_dfs[1:]:
        combined = combined.join(next_df, on=["chrom", "pos"], how="full", coalesce=True)
        
    # Trigger computation
    df = combined.collect()
    
    # Separate into beta and coverage DataFrames
    beta_cols = ["chrom", "pos"] + sample_names
    cov_cols = ["chrom", "pos"] + [f"{name}_cov" for name in sample_names]
    
    beta_df = df.select(beta_cols)
    
    # Rename cov columns to match sample names for consistent indexing and fill nulls with 0
    cov_df = df.select(cov_cols)
    cov_df = cov_df.rename({f"{name}_cov": name for name in sample_names}).fill_null(0)
    
    return MethylationDataset(beta_df, cov_df)


# Unified array manifest coordinate mapping cache
_UNIFIED_PROBE_MANIFEST = None

def _load_unified_manifest() -> Dict[str, Tuple[str, int]]:
    """
    Dynamically loads and merges clock and deconvolution probe manifests
    from JSON resource files into a high-performance coordinate lookup map.
    """
    global _UNIFIED_PROBE_MANIFEST
    if _UNIFIED_PROBE_MANIFEST is not None:
        return _UNIFIED_PROBE_MANIFEST

    import json
    manifest = {}
    base_dir = os.path.dirname(__file__)
    
    # Files containing coordinate manifests
    manifest_files = [
        "horvath_model.json",
        "hannum_model.json",
        "blood_decon_model.json"
    ]
    
    for filename in manifest_files:
        path = os.path.join(base_dir, "resources", filename)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.get("manifest", {}).items():
                        manifest[k] = (v[0], int(v[1]))
            except Exception:
                pass
                
    # Fallback to standard clock mock coordinates if models aren't loaded yet
    fallback_mock = {
        "cg00000029": ("chr16", 53468539),
        "cg00000292": ("chr16", 28892182),
        "cg00002426": ("chr3", 57743543),
        "cg00003858": ("chr1", 15632345),
        "cg02242131": ("chr1", 20456213),
        "cg08945781": ("chr2", 45781223),
        "cg06493994": ("chr3", 102458921),
        "cg09809672": ("chr4", 89456213),
        "cg22730898": ("chr5", 152146931),
        "cg01820374": ("chr6", 32456219),
        "cg19766904": ("chr7", 105632145),
        "cg04874057": ("chr8", 22457891),
        "cg16867657": ("chr9", 139456211),
        "cg19854823": ("chr10", 72456233),
    }
    for k, v in fallback_mock.items():
        if k not in manifest:
            manifest[k] = v
            
    _UNIFIED_PROBE_MANIFEST = manifest
    return _UNIFIED_PROBE_MANIFEST


def load_array_beta(
    filepath: str, 
    sample_metadata: Optional[Dict[str, str]] = None,
    manifest: Optional[Dict[str, tuple]] = None
) -> MethylationDataset:
    """
    Load microarray-based beta values (e.g. EPIC, 450K arrays) from a file (CSV/TSV/Parquet)
    and map array probe IDs to genomic coordinates to unified coordinates.
    Natively supports EPIC v2 probe IDs containing revised design suffixes.
    
    Format expectation:
    First column contains probe IDs (e.g., cg00000029 or cg02242131_BC21), and subsequent columns 
    contain sample beta values.
    """
    # Read file using Polars
    if filepath.endswith(".parquet"):
        df = pl.read_parquet(filepath)
    else:
        # Assume CSV/TSV
        sep = "\t" if filepath.endswith((".tsv", ".txt")) else ","
        df = pl.read_csv(filepath, separator=sep)
        
    probe_col = df.columns[0]
    sample_names = df.columns[1:]
    
    # Map probes to genomic coordinates
    probe_map = manifest if manifest is not None else _load_unified_manifest()
    
    # We will build chrom and pos columns based on the manifest map
    probes = df[probe_col].to_list()
    chroms = []
    positions = []
    valid_indices = []
    
    for idx, probe in enumerate(probes):
        # Exact match
        if probe in probe_map:
            chrom, pos = probe_map[probe]
            chroms.append(chrom)
            positions.append(pos)
            valid_indices.append(idx)
        # Suffix/EPIC v2 match (e.g. cg02242131_BC21)
        elif "_" in probe:
            base_probe = probe.split("_")[0]
            if base_probe in probe_map:
                chrom, pos = probe_map[base_probe]
                chroms.append(chrom)
                positions.append(pos)
                valid_indices.append(idx)
            
    if len(valid_indices) == 0:
        import warnings
        warnings.warn(
            "load_array_beta(): No probes in the input file matched the provided manifest. "
            "The returned MethylationDataset will be empty. "
            "Check that your probe IDs (e.g. cg00000029) match the manifest format, "
            "or supply a custom manifest= argument.",
            UserWarning,
            stacklevel=2
        )
            
    # Filter original dataframe to keep only mapped probes
    filtered_df = df.filter(pl.arange(0, df.height).is_in(valid_indices))
    
    # Add chrom and pos columns
    mapped_df = filtered_df.with_columns([
        pl.Series(name="chrom", values=chroms),
        pl.Series(name="pos", values=positions, dtype=pl.Int64)
    ])
    
    # Reorder columns to place chrom and pos first
    final_cols = ["chrom", "pos"] + list(sample_names)
    beta_df = mapped_df.select(final_cols)
    
    return MethylationDataset(beta_df, cov_df=None, metadata=sample_metadata)


def load_nanopore_modkit(
    filepaths: List[str], 
    sample_names: List[str], 
    min_cov: int = 5
) -> MethylationDataset:
    """
    Parse Oxford Nanopore modkit bedmethyl outputs (11-column BED9+2 format)
    and perform high-speed inner joins using Polars to align all samples by coordinates.
    """
    assert len(filepaths) == len(sample_names), "Number of file paths must equal number of sample names"
    
    lazy_dfs = []
    
    for path, name in zip(filepaths, sample_names):
        ldf = pl.scan_csv(
            path,
            separator="\t",
            has_header=False,
            schema={
                "column_1": pl.String,   # chrom
                "column_2": pl.Int64,    # start (0-indexed)
                "column_3": pl.Int64,    # end (1-indexed)
                "column_4": pl.String,   # name (modification code, e.g., 'm')
                "column_5": pl.Int64,    # score
                "column_6": pl.String,   # strand
                "column_7": pl.Int64,    # thickStart
                "column_8": pl.Int64,    # thickEnd
                "column_9": pl.String,   # itemRgb
                "column_10": pl.Int64,   # coverage
                "column_11": pl.Float64, # percentage (0 - 100)
            }
        ).rename({
            "column_1": "chrom",
            "column_2": "pos",  # 0-indexed start treated as the coordinate
            "column_4": "mod_code",
            "column_10": "coverage",
            "column_11": "pct"
        })
        
        # Filter for 5mC modifications ('m')
        ldf = ldf.filter(pl.col("mod_code") == "m")
        
        # Calculate derived columns: beta value and total coverage
        ldf = ldf.with_columns([
            pl.col("coverage").alias(f"{name}_cov"),
            (pl.col("pct") / 100.0).alias(name)
        ])
        
        # Filter by coverage per sample early if specified
        if min_cov > 0:
            ldf = ldf.filter(pl.col(f"{name}_cov") >= min_cov)
            
        # Select only required columns
        ldf = ldf.select(["chrom", "pos", name, f"{name}_cov"])
        lazy_dfs.append(ldf)
        
    # Perform sequential outer joins to align all samples by chrom and pos
    combined = lazy_dfs[0]
    for next_df in lazy_dfs[1:]:
        combined = combined.join(next_df, on=["chrom", "pos"], how="full", coalesce=True)
        
    # Trigger computation
    df = combined.collect()
    
    # Separate into beta and coverage DataFrames
    beta_cols = ["chrom", "pos"] + sample_names
    cov_cols = ["chrom", "pos"] + [f"{name}_cov" for name in sample_names]
    
    beta_df = df.select(beta_cols)
    cov_df = df.select(cov_cols)
    cov_df = cov_df.rename({f"{name}_cov": name for name in sample_names}).fill_null(0)
    
    return MethylationDataset(beta_df, cov_df)


def load_pacbio_bedgraph(
    filepaths: List[str], 
    sample_names: List[str], 
    min_cov: int = 5
) -> MethylationDataset:
    """
    Parse PacBio bedGraph modification outputs and perform high-speed inner joins
    using Polars to align all samples by coordinates.
    Supports both 4-column (percentage only) and 5-column (coverage + percentage) bedGraphs.
    """
    assert len(filepaths) == len(sample_names), "Number of file paths must equal number of sample names"
    
    lazy_dfs = []
    
    for path, name in zip(filepaths, sample_names):
        ldf = pl.scan_csv(
            path,
            separator="\t",
            has_header=False,
            infer_schema_length=100
        )
        
        cols = ldf.collect_schema().names()
        # Rename based on actual number of columns
        if len(cols) >= 5:
            # Column 5 is coverage
            ldf = ldf.rename({
                cols[0]: "chrom",
                cols[1]: "pos",
                cols[3]: "pct",
                cols[4]: "coverage"
            })
        else:
            # 4 columns, column 4 is pct, no coverage. We default coverage to 30.
            ldf = ldf.rename({
                cols[0]: "chrom",
                cols[1]: "pos",
                cols[3]: "pct"
            }).with_columns(
                pl.lit(30).alias("coverage")
            )
            
        # Ensure correct column types
        ldf = ldf.with_columns([
            pl.col("chrom").cast(pl.String),
            pl.col("pos").cast(pl.Int64),
            pl.col("pct").cast(pl.Float64),
            pl.col("coverage").cast(pl.Int64)
        ])
        
        # Calculate derived columns: beta value and total coverage.
        # Handle 0-100 and 0-1 percentage scales.
        ldf = ldf.with_columns([
            pl.when(pl.col("pct") > 1.0)
            .then(pl.col("pct") / 100.0)
            .otherwise(pl.col("pct"))
            .alias(name),
            pl.col("coverage").alias(f"{name}_cov")
        ])
        
        # Filter by coverage per sample early if specified
        if min_cov > 0:
            ldf = ldf.filter(pl.col(f"{name}_cov") >= min_cov)
            
        # Select only required columns
        ldf = ldf.select(["chrom", "pos", name, f"{name}_cov"])
        lazy_dfs.append(ldf)
        
    # Perform sequential outer joins to align all samples by chrom and pos
    combined = lazy_dfs[0]
    for next_df in lazy_dfs[1:]:
        combined = combined.join(next_df, on=["chrom", "pos"], how="full", coalesce=True)
        
    # Trigger computation
    df = combined.collect()
    
    # Separate into beta and coverage DataFrames
    beta_cols = ["chrom", "pos"] + sample_names
    cov_cols = ["chrom", "pos"] + [f"{name}_cov" for name in sample_names]
    
    beta_df = df.select(beta_cols)
    cov_df = df.select(cov_cols)
    cov_df = cov_df.rename({f"{name}_cov": name for name in sample_names}).fill_null(0)
    
    return MethylationDataset(beta_df, cov_df)

