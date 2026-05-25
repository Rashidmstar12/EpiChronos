import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Optional
from epichronos.core import MethylationDataset

class EpigeneticPacemaker:
    """
    Epigenetic Pacemaker (EPM) biological age estimation model.
    Utilizes an alternating coordinate-descent algorithm to capture non-linear
    aging trajectories across individual CpG sites without assuming linear relationships
    with chronological age.
    
    Algorithm based on the Epigenetic Pacemaker model described in Snir et al. (2016) PLOS Computational Biology, doi:10.1371/journal.pcbi.1004913
    """
    def __init__(self, max_iter: int = 150, tol: float = 1e-5):
        """
        Initialize the EPM model.
        
        Args:
            max_iter: Maximum coordinate-descent iterations.
            tol: Convergence tolerance (minimum change in biological age values).
        """
        self.max_iter = max_iter
        self.tol = tol
        self.rates_ = None          # rates r_i
        self.intercepts_ = None     # intercepts b_i
        self.chroms_ = None         # chrom coordinates
        self.positions_ = None      # pos coordinates
        
        # Chronological scaling factors to align biological ages with real age scales
        self.mean_t_ = 0.0
        self.std_t_ = 1.0
        self.mean_chron_ = 0.0
        self.std_chron_ = 1.0
        
        # Training loss history for visualization
        self.loss_history_ = []

    def fit(self, dataset: MethylationDataset, chronological_ages: Dict[str, float]) -> 'EpigeneticPacemaker':
        """
        Fit the Epigenetic Pacemaker model on a cohort dataset on-the-fly.
        
        Args:
            dataset: MethylationDataset containing cohort beta values.
            chronological_ages: Dictionary mapping sample names to true chronological ages.
        """
        # 1. Extract sample coordinates and beta values
        samples = [s for s in dataset.samples if s in chronological_ages]
        if len(samples) < 3:
            raise ValueError("Epigenetic Pacemaker requires at least 3 samples with chronological age to train.")
            
        n_samples = len(samples)
        chron_list = np.array([chronological_ages[s] for s in samples])
        
        self.mean_chron_ = float(np.mean(chron_list))
        self.std_chron_ = float(np.std(chron_list))
        if self.std_chron_ == 0.0:
            self.std_chron_ = 1.0  # Avoid divide-by-zero
            
        # Extract beta values matrix (n_sites, n_samples)
        beta_df = dataset.beta_df
        self.chroms_ = beta_df["chrom"].to_list()
        self.positions_ = beta_df["pos"].to_list()
        
        # Keep only samples with chronological age
        S = beta_df.select(samples).to_numpy().astype(np.float64)
        
        # Impute missing beta values (EPM cannot have NaNs)
        n_sites = S.shape[0]
        for i in range(n_sites):
            nan_mask = np.isnan(S[i])
            if np.any(nan_mask):
                valid_mean = np.mean(S[i, ~nan_mask]) if np.any(~nan_mask) else 0.5
                S[i, nan_mask] = valid_mean
                
        # 2. Alternating Coordinate Descent Optimization
        # Initialize biological age t to chronological age
        t = chron_list.copy()
        
        self.loss_history_ = []
        
        for iteration in range(self.max_iter):
            t_prev = t.copy()
            
            # Step A: Optimize site parameters (r_i, b_i) for all sites in parallel (vectorized)
            # y = S[i, :], X = t
            t_mean = np.mean(t)
            dt = t - t_mean
            var_t = np.sum(dt**2)
            
            if var_t == 0.0:
                var_t = 1.0  # Avoid division by zero
                
            S_mean = np.mean(S, axis=1)
            # Vectorized covariance calculation: S_diff shape (n_sites, n_samples)
            S_diff = S - S_mean[:, np.newaxis]
            
            # r = Cov(S, t) / Var(t)
            r = np.dot(S_diff, dt) / var_t
            # b = mean(S) - r * mean(t)
            b = S_mean - r * t_mean
            
            # Step B: Optimize sample biological ages (t_j) for all samples in parallel (vectorized)
            # S[:, j] = r * t_j + b  => t_j = sum(r_i * (S_{i,j} - b_i)) / sum(r_i^2)
            r_sum_sq = np.sum(r**2)
            if r_sum_sq == 0.0:
                r_sum_sq = 1.0  # Avoid division by zero
                
            # Vectorized dot product
            t_raw = np.dot(r, S - b[:, np.newaxis]) / r_sum_sq
            
            # Save raw scaling factors in the last iteration
            if iteration == self.max_iter - 1:
                self.mean_t_ = float(np.mean(t_raw))
                self.std_t_ = float(np.std(t_raw))
                if self.std_t_ == 0.0:
                    self.std_t_ = 1.0
            
            # Step C: Rescale biological ages to align with chronological age scale
            t_std = np.std(t_raw)
            if t_std == 0.0:
                t_std = 1.0
            t = (t_raw - np.mean(t_raw)) / t_std * self.std_chron_ + self.mean_chron_
            
            # Step D: Calculate Loss (Mean Squared Error)
            pred_S = r[:, np.newaxis] * t + b[:, np.newaxis]
            loss = float(np.mean((S - pred_S)**2))
            self.loss_history_.append(loss)
            
            # Convergence check
            diff = np.mean(np.abs(t - t_prev))
            if diff < self.tol:
                # Save scale factors at convergence
                self.mean_t_ = float(np.mean(t_raw))
                self.std_t_ = float(np.std(t_raw))
                if self.std_t_ == 0.0:
                    self.std_t_ = 1.0
                break
                
        self.rates_ = r
        self.intercepts_ = b
        return self

    def predict(self, dataset: MethylationDataset) -> pl.DataFrame:
        """
        Predict Epigenetic Pacemaker biological ages for samples in a dataset.
        
        Args:
            dataset: MethylationDataset containing sample beta values.
            
        Returns:
            Polars DataFrame containing sample names and predicted Pacemaker ages.
        """
        if self.rates_ is None or self.intercepts_ is None:
            raise ValueError("Epigenetic Pacemaker model must be fitted before prediction.")
            
        sample_names = dataset.samples
        n_samples = len(sample_names)
        
        # Align dataset coordinate values with model sites using Polars high-speed inner join
        model_df = pl.DataFrame({
            "chrom": self.chroms_,
            "pos": self.positions_,
            "rate": self.rates_,
            "intercept": self.intercepts_
        })
        
        # Inner join to align coordinate space
        aligned = model_df.join(dataset.beta_df, on=["chrom", "pos"], how="inner")
        
        if aligned.height == 0:
            # Fallback if no overlap
            t_predicted = np.full(n_samples, self.mean_chron_)
        else:
            S = aligned.select(sample_names).to_numpy().astype(np.float64)
            rates = aligned["rate"].to_numpy().astype(np.float64)
            intercepts = aligned["intercept"].to_numpy().astype(np.float64)
            
            # Impute NaNs with intercepts
            for j in range(n_samples):
                nan_mask = np.isnan(S[:, j])
                if np.any(nan_mask):
                    S[nan_mask, j] = intercepts[nan_mask]
                    
            # Vectorized prediction
            r_sum_sq = np.sum(rates**2)
            if r_sum_sq == 0.0:
                r_sum_sq = 1.0
                
            t_raw = np.dot(rates, S - intercepts[:, np.newaxis]) / r_sum_sq
            t_predicted = (t_raw - self.mean_t_) / self.std_t_ * self.std_chron_ + self.mean_chron_
            
        # Compile results
        out_dict = {
            "sample": sample_names,
            "biological_age": t_predicted.tolist()
        }
        return pl.DataFrame(out_dict).sort("sample")
