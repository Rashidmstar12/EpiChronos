import polars as pl
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Union, Optional
from plotly.offline import plot, get_plotlyjs
import plotly.express as px
import plotly.graph_objects as go

from epichronos.core import MethylationDataset

def _compute_pca(dataset: MethylationDataset) -> pl.DataFrame:
    """Perform self-contained SVD-based PCA on the dataset beta values."""
    samples = dataset.samples
    # Shape: (CpGs, samples) -> Transpose to (samples, CpGs) for SVD
    beta_mat = dataset.beta_df.select(samples).to_numpy().T
    
    # Impute any remaining NaNs with column means
    col_means = np.nanmean(beta_mat, axis=0)
    # Default to 0.5 if all are NaN
    col_means = np.nan_to_num(col_means, nan=0.5)
    
    for col in range(beta_mat.shape[1]):
        nans = np.isnan(beta_mat[:, col])
        if np.any(nans):
            beta_mat[nans, col] = col_means[col]
            
    # Center the matrix
    centered_mat = beta_mat - np.mean(beta_mat, axis=0)
    
    # Singular Value Decomposition (SVD)
    # centered_mat = U * S * Vt
    try:
        u, s, vt = np.linalg.svd(centered_mat, full_matrices=False)
        pcs = u * s
        pc1 = pcs[:, 0]
        pc2 = pcs[:, 1]
    except Exception:
        # Fallback to zeros if SVD fails (e.g. extremely low variance)
        pc1 = np.zeros(len(samples))
        pc2 = np.zeros(len(samples))
        
    pca_df = pl.DataFrame({
        "sample": samples,
        "PC1": pc1,
        "PC2": pc2,
        "group": [dataset.metadata[s] for s in samples]
    })
    return pca_df


def generate_report(
    dataset: MethylationDataset,
    dml_df: pl.DataFrame,
    dmr_df: pl.DataFrame,
    clock_df: pl.DataFrame,
    output_html_path: str,
    decon_df: Optional[pl.DataFrame] = None,
    ieaa_df: Optional[pl.DataFrame] = None,
    expression_df: Optional[pl.DataFrame] = None,
    meqtl_df: Optional[pl.DataFrame] = None,
    enrich_df: Optional[pl.DataFrame] = None
):
    """
    Generate an extremely beautiful, interactive standalone HTML report
    incorporating Plotly visualizations, statistical summaries, biological age insights,
    immune cell-type deconvolution, and multi-omics transcriptomic correlations.
    
    Args:
        dataset: The aligned MethylationDataset.
        dml_df: Called DMLs DataFrame.
        dmr_df: Called DMRs DataFrame.
        clock_df: Predicted biological ages DataFrame.
        output_html_path: Filepath where the HTML report will be written.
        decon_df: Optional cell deconvolution proportions DataFrame.
        ieaa_df: Optional Intrinsic Epigenetic Age Acceleration DataFrame.
        expression_df: Optional RNA-seq gene expression DataFrame.
        meqtl_df: Optional meQTL transcription integration DataFrame.
    """
    # 1. Compute and generate PCA plot
    pca_df = _compute_pca(dataset)
    fig_pca = px.scatter(
        pca_df.to_pandas(), 
        x="PC1", 
        y="PC2", 
        color="group",
        hover_name="sample",
        title="Global Methylation PCA (SVD-centered)",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_pca.update_layout(
        plot_bgcolor="rgba(30, 41, 59, 0.4)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font_color="#cbd5e1",
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155")
    )
    pca_div = plot(fig_pca, output_type="div", include_plotlyjs=False)

    # 2. Generate Volcano Plot of DMLs
    volc_data = dml_df.to_pandas()
    # Log transform p-values
    volc_data["neg_log_p"] = -np.log10(volc_data["p_value"] + 1e-300)
    volc_data["hover_text"] = volc_data["chrom"] + ":" + volc_data["pos"].astype(str)
    
    # Define significance status
    volc_data["status"] = "Not Significant"
    volc_data.loc[(volc_data["p_value"] <= 0.05) & (volc_data["mean_diff"].abs() >= 0.1), "status"] = "Significant"
    
    fig_volc = px.scatter(
        volc_data, 
        x="mean_diff", 
        y="neg_log_p", 
        color="status",
        hover_name="hover_text",
        title="DML Volcano Plot (Effect Size vs Significance)",
        labels={"mean_diff": "Methylation Difference (Mean_B - Mean_A)", "neg_log_p": "-log10(P-Value)"},
        color_discrete_map={"Significant": "#ef4444", "Not Significant": "#94a3b8"}
    )
    fig_volc.update_layout(
        plot_bgcolor="rgba(30, 41, 59, 0.4)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        font_color="#cbd5e1",
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155")
    )
    volc_div = plot(fig_volc, output_type="div", include_plotlyjs=False)

    # 3. Generate Epigenetic Clock Age Acceleration Bar Chart (if chronological age is present)
    has_chron = "chronological_age" in clock_df.columns
    age_div = ""
    if has_chron:
        clock_data = clock_df.filter(pl.col("age_acceleration").is_not_null()).to_pandas()
        fig_age = px.bar(
            clock_data,
            x="sample",
            y="age_acceleration",
            color="age_acceleration",
            title="Epigenetic Age Acceleration (Residuals)",
            labels={"age_acceleration": "Age Acceleration (Years)", "sample": "Sample ID"},
            color_continuous_scale=px.colors.sequential.RdBu_r
        )
        fig_age.update_layout(
            plot_bgcolor="rgba(30, 41, 59, 0.4)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            font_color="#cbd5e1",
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155")
        )
        age_div = plot(fig_age, output_type="div", include_plotlyjs=False)
    else:
        # Just plot predicted vs sample bar chart
        clock_data = clock_df.to_pandas()
        fig_age = px.bar(
            clock_data,
            x="sample",
            y="biological_age",
            title="Predicted Epigenetic Biological Age",
            labels={"biological_age": "Biological Age (Years)", "sample": "Sample ID"},
            color_discrete_sequence=["#10b981"]
        )
        fig_age.update_layout(
            plot_bgcolor="rgba(30, 41, 59, 0.4)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            font_color="#cbd5e1",
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155")
        )
        age_div = plot(fig_age, output_type="div", include_plotlyjs=False)

    # 3.5 Generate Deconvolution Plots if data is available
    decon_div = ""
    comp_div = ""
    if decon_df is not None:
        cell_types = ["Neutrophils", "NK", "Bcell", "CD4T", "CD8T", "Monocytes"]
        fig_decon = px.bar(
            decon_df.to_pandas(),
            x="sample",
            y=cell_types,
            title="Blood Immune Cell-Type Proportions",
            labels={"value": "Proportion", "sample": "Sample ID", "variable": "Cell Type"},
            template="plotly_dark",
            barmode="stack",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_decon.update_layout(
            plot_bgcolor="rgba(30, 41, 59, 0.4)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            font_color="#cbd5e1",
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155", range=[0, 1.0])
        )
        decon_div = plot(fig_decon, output_type="div", include_plotlyjs=False)
        
        if ieaa_df is not None and has_chron:
            comp_df = clock_df.join(ieaa_df, on="sample", how="inner")
            comp_pd = comp_df.to_pandas()
            melted = comp_pd.melt(
                id_vars=["sample"],
                value_vars=["age_acceleration", "intrinsic_age_acceleration"],
                var_name="Metric",
                value_name="Acceleration"
            )
            melted["Metric"] = melted["Metric"].map({
                "age_acceleration": "EEAA (Unadjusted)",
                "intrinsic_age_acceleration": "IEAA (Cell-Type Adjusted)"
            })
            fig_comp = px.bar(
                melted,
                x="sample",
                y="Acceleration",
                color="Metric",
                barmode="group",
                title="Epigenetic Age Acceleration comparison (EEAA vs. IEAA)",
                template="plotly_dark",
                color_discrete_map={"EEAA (Unadjusted)": "#ef4444", "IEAA (Cell-Type Adjusted)": "#10b981"}
            )
            fig_comp.update_layout(
                plot_bgcolor="rgba(30, 41, 59, 0.4)",
                paper_bgcolor="rgba(0, 0, 0, 0)",
                font_color="#cbd5e1",
                xaxis=dict(gridcolor="#334155"),
                yaxis=dict(gridcolor="#334155")
            )
            comp_div = plot(fig_comp, output_type="div", include_plotlyjs=False)

    # 3.7 Generate meQTL Plots if data is available
    meqtl_div = ""
    if meqtl_df is not None and meqtl_df.height > 0 and expression_df is not None:
        # Find the most significant meQTL
        top_meqtl = meqtl_df.sort("p_value").head(1).to_dicts()[0]
        gene = top_meqtl["gene"]
        chrom = top_meqtl["chrom"]
        start = top_meqtl["start"]
        end = top_meqtl["end"]
        
        # Calculate sample-specific DMR average beta
        dmr_sites = dataset.beta_df.filter(
            (pl.col("chrom") == chrom) & 
            (pl.col("pos") >= start) & 
            (pl.col("pos") <= end)
        )
        
        samples = dataset.samples
        beta_vals = []
        expr_vals = []
        
        gene_col = expression_df.columns[0]
        gene_rows = expression_df.filter(pl.col(gene_col) == gene).to_dicts()
        if gene_rows:
            gene_row = gene_rows[0]
            for s in samples:
                cpg_betas = dmr_sites[s].drop_nans().drop_nulls().to_list()
                beta_vals.append(np.mean(cpg_betas) if len(cpg_betas) > 0 else 0.5)
                expr_vals.append(gene_row.get(s, 0.0))
                
            beta_vals = np.array(beta_vals)
            expr_vals = np.array(expr_vals)
            
            # Scatter plot
            fig_meqtl = px.scatter(
                x=beta_vals,
                y=expr_vals,
                text=samples,
                title=f"meQTL Linkage: DMR {chrom}:{start}-{end} vs. {gene} Expression",
                labels={"x": "DMR Methylation (Beta)", "y": "Gene Expression Level"},
                template="plotly_dark",
                color_discrete_sequence=["#10b981"]
            )
            fig_meqtl.update_traces(textposition='top center', marker=dict(size=12))
            
            if len(beta_vals) >= 2:
                slope, intercept = np.polyfit(beta_vals, expr_vals, 1)
                x_line = np.linspace(beta_vals.min() - 0.05, beta_vals.max() + 0.05, 100)
                y_line = slope * x_line + intercept
                fig_meqtl.add_trace(go.Scatter(
                    x=x_line, y=y_line, mode="lines", name="Linear Fit",
                    line=dict(color="#38bdf8", dash="dash")
                ))
                
            fig_meqtl.update_layout(
                plot_bgcolor="rgba(30, 41, 59, 0.4)",
                paper_bgcolor="rgba(0, 0, 0, 0)",
                font_color="#cbd5e1",
                xaxis=dict(gridcolor="#334155"),
                yaxis=dict(gridcolor="#334155")
            )
            meqtl_div = plot(fig_meqtl, output_type="div", include_plotlyjs=False)

    # 3.8 Generate Pathway Enrichment Plots if data is available
    enrich_div = ""
    if enrich_df is not None and enrich_df.height > 0:
        enrich_data = enrich_df.to_pandas()
        enrich_data["neg_log_p"] = -np.log10(enrich_data["p_value"] + 1e-300)
        enrich_data = enrich_data.sort_values(by="p_value", ascending=True)
        
        fig_enrich = px.bar(
            enrich_data,
            x="neg_log_p",
            y="pathway",
            orientation="h",
            color="overlap_count",
            title="Significantly Enriched Biological Pathways (ORA)",
            labels={"neg_log_p": "-log10(P-Value)", "pathway": "Pathway Name", "overlap_count": "Overlapping Genes"},
            template="plotly_dark",
            color_continuous_scale=px.colors.sequential.Teal
        )
        fig_enrich.update_layout(
            plot_bgcolor="rgba(30, 41, 59, 0.4)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            font_color="#cbd5e1",
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155")
        )
        enrich_div = plot(fig_enrich, output_type="div", include_plotlyjs=False)

    # 4. Compile HTML Report using high-aesthetic styling and Jinja-like template string
    summary_metrics = {
        "cpgs": len(dataset.beta_df),
        "samples": len(dataset.samples),
        "dmls": len(dml_df.filter(pl.col("p_value") <= 0.05)),
        "dmrs": len(dmr_df)
    }
    
    # Get top 5 DMRs by absolute area
    top_dmrs = dmr_df.sort(pl.col("area").abs(), descending=True).head(5).to_dicts()
    top_clocks = clock_df.to_dicts()

    # Load offline bundled Plotly JS and Tailwind CSS
    plotly_js_content = get_plotlyjs()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tailwind_path = os.path.join(base_dir, "resources", "tailwind.min.js")
    with open(tailwind_path, "r", encoding="utf-8") as tf:
        tailwind_js_content = tf.read()

    html_content = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EpiChronos Epigenomics Analysis Report</title>
    <!-- Tailwind CSS (Offline Bundled) -->
    <script type="text/javascript">{tailwind_js_content}</script>
    <!-- Plotly Core JS (Offline Bundled) -->
    <script type="text/javascript">{plotly_js_content}</script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
        }}
        .glass-card {{
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
    </style>
</head>
<body class="p-6 md:p-12 min-h-screen">
    <div class="max-w-7xl mx-auto space-y-8">
        <!-- Header -->
        <div class="flex flex-col md:flex-row md:items-center md:justify-between border-b border-slate-800 pb-6 gap-4">
            <div>
                <h1 class="text-4xl font-extrabold tracking-tight bg-gradient-to-r from-teal-400 to-indigo-400 bg-clip-text text-transparent">
                    EpiChronos Dashboard
                </h1>
                <p class="text-slate-400 mt-1">High-Performance Epigenomics & Biological Age Report</p>
            </div>
            <div class="text-sm text-slate-500 font-mono">
                Generated: 2026-05-23
            </div>
        </div>

        <!-- Metrics Overview Grid -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="glass-card p-6 rounded-2xl">
                <span class="text-xs font-semibold text-teal-400 tracking-wider uppercase">Aligned CpGs</span>
                <p class="text-3xl font-bold mt-1 text-slate-100">{summary_metrics['cpgs']:,}</p>
            </div>
            <div class="glass-card p-6 rounded-2xl">
                <span class="text-xs font-semibold text-indigo-400 tracking-wider uppercase">Samples Analysed</span>
                <p class="text-3xl font-bold mt-1 text-slate-100">{summary_metrics['samples']}</p>
            </div>
            <div class="glass-card p-6 rounded-2xl">
                <span class="text-xs font-semibold text-rose-400 tracking-wider uppercase">Significant DMLs</span>
                <p class="text-3xl font-bold mt-1 text-slate-100">{summary_metrics['dmls']}</p>
            </div>
            <div class="glass-card p-6 rounded-2xl">
                <span class="text-xs font-semibold text-amber-400 tracking-wider uppercase">Called DMRs</span>
                <p class="text-3xl font-bold mt-1 text-slate-100">{summary_metrics['dmrs']}</p>
            </div>
        </div>

        <!-- Main Content Area: Grid of Plots -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <!-- PCA glass container -->
            <div class="glass-card p-6 rounded-3xl flex flex-col justify-between">
                <div>
                    <h2 class="text-xl font-bold text-slate-100">Dimensionality Reduction</h2>
                    <p class="text-xs text-slate-400 mt-1 mb-4">Unsupervised grouping metrics of samples using centering SVD projection.</p>
                </div>
                <div class="w-full flex-grow">{pca_div}</div>
            </div>

            <!-- Volcano Plot container -->
            <div class="glass-card p-6 rounded-3xl flex flex-col justify-between">
                <div>
                    <h2 class="text-xl font-bold text-slate-100">Differential Loci Calling</h2>
                    <p class="text-xs text-slate-400 mt-1 mb-4">Welch's t-test comparing individual CpG sites. Significant loci have P <= 0.05 and diff >= 10%.</p>
                </div>
                <div class="w-full flex-grow">{volc_div}</div>
            </div>
            
            <!-- Biological Aging clocks container -->
            <div class="glass-card p-6 rounded-3xl flex flex-col justify-between lg:col-span-2">
                <div>
                    <h2 class="text-xl font-bold text-slate-100">Biological Clock Calculations</h2>
                    <p class="text-xs text-slate-400 mt-1 mb-4">Predictions of cellular biological age alongside true chronological equivalents and acceleration statistics.</p>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    <div class="w-full">{age_div}</div>
                    
                    <!-- Clocks metrics table -->
                    <div class="overflow-x-auto">
                        <table class="w-full text-left border-collapse text-sm">
                            <thead>
                                <tr class="border-b border-slate-700 text-slate-400 font-semibold">
                                    <th class="py-3 px-4">Sample ID</th>
                                    <th class="py-3 px-4">Biological Age</th>
                                    {"<th class='py-3 px-4'>Chronological Age</th><th class='py-3 px-4'>Acceleration</th>" if has_chron else ""}
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-800 text-slate-300">
                                {"".join([
                                    f"<tr class='hover:bg-slate-800/40'><td class='py-3 px-4 font-mono'>{row['sample']}</td><td class='py-3 px-4 font-bold text-emerald-400'>{row['biological_age']:.2f} yrs</td>" +
                                    (f"<td class='py-3 px-4'>{row['chronological_age']:.1f} yrs</td><td class='py-3 px-4 font-bold " + 
                                     ("text-red-400" if row['age_acceleration'] > 0 else "text-emerald-400") + 
                                     f"'>{row['age_acceleration']:.2f} yrs</td>" if has_chron else "") + "</tr>"
                                    for row in top_clocks
                                ])}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        
        {f"""
        <!-- Deconvolution and Intrinsic Aging Container -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-8">
            <!-- Cell proportions stacked bar chart -->
            <div class="glass-card p-6 rounded-3xl flex flex-col justify-between">
                <div>
                    <h2 class="text-xl font-bold text-slate-100">Immune Cell-Type Deconvolution</h2>
                    <p class="text-xs text-slate-400 mt-1 mb-4">Estimated blood immune cell-type proportions from the high-accuracy IDOL reference panel.</p>
                </div>
                <div class="w-full flex-grow">{decon_div}</div>
            </div>

            <!-- Comparative age acceleration bar chart -->
            <div class="glass-card p-6 rounded-3xl flex flex-col justify-between">
                <div>
                    <h2 class="text-xl font-bold text-slate-100">Intrinsic Epigenetic Aging Rates</h2>
                    <p class="text-xs text-slate-400 mt-1 mb-4">Comparison of unadjusted age acceleration (EEAA) and cell-type adjusted intrinsic acceleration (IEAA).</p>
                </div>
                <div class="w-full flex-grow">{comp_div if comp_div else '<div class="py-12 text-center text-slate-500">Chronological age or cell-type regression residuals are not available for this run.</div>'}</div>
            </div>
        </div>
        """ if decon_df is not None else ""}

        {f"""
        <!-- Transcription meQTL Linker Container -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-8">
            <!-- meQTL Scatter Plot -->
            <div class="glass-card p-6 rounded-3xl flex flex-col justify-between">
                <div>
                    <h2 class="text-xl font-bold text-slate-100">Transcription meQTL Linkage</h2>
                    <p class="text-xs text-slate-400 mt-1 mb-4">Top called meQTL hit showing correlation between DMR methylation and downstream gene expression.</p>
                </div>
                <div class="w-full flex-grow">{meqtl_div}</div>
            </div>

            <!-- meQTL Table -->
            <div class="glass-card p-6 rounded-3xl flex flex-col justify-between">
                <div>
                    <h2 class="text-xl font-bold text-slate-100">Functional Regulatory Statuses</h2>
                    <p class="text-xs text-slate-400 mt-1 mb-4">Called DMRs aligned with downstream gene expression, corrected for multiple testing using FDR Benjamini-Hochberg.</p>
                </div>
                <div class="overflow-x-auto my-auto">
                    <table class="w-full text-left border-collapse text-sm">
                        <thead>
                            <tr class="border-b border-slate-700 text-slate-400 font-semibold">
                                <th class="py-2 px-3">DMR Coordinates</th>
                                <th class="py-2 px-3">Gene</th>
                                <th class="py-2 px-3 text-right">Pearson r</th>
                                <th class="py-2 px-3 text-right">FDR q-val</th>
                                <th class="py-2 px-3 text-right">Status</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-800 text-slate-300">
                            {"".join([
                                f"<tr class='hover:bg-slate-800/40'>"
                                f"<td class='py-2 px-3 font-mono'>{row['chrom']}:{row['start']}-{row['end']}</td>"
                                f"<td class='py-2 px-3 font-bold text-teal-300'>{row['gene']}</td>"
                                f"<td class='py-2 px-3 text-right font-mono'>{row['correlation_r']:.4f}</td>"
                                f"<td class='py-2 px-3 text-right font-mono'>{row['q_value']:.4e}</td>"
                                f"<td class='py-2 px-3 text-right font-bold " + 
                                ("#f43f5e" if "Silencing" in row['functional_status'] else "#10b981" if "Activating" in row['functional_status'] else "text-slate-400") + 
                                f"'>{row['functional_status']}</td>"
                                f"</tr>"
                                for row in meqtl_df.head(5).to_dicts()
                            ])}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        """ if meqtl_df is not None else ""}

        {f"""
        <!-- Pathway Enrichment Container -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-8">
            <!-- Pathway Bar Chart -->
            <div class="glass-card p-6 rounded-3xl flex flex-col justify-between">
                <div>
                    <h2 class="text-xl font-bold text-slate-100">GO/KEGG Pathway Enrichment (ORA)</h2>
                    <p class="text-xs text-slate-400 mt-1 mb-4">Statistically enriched biological pathways adjacent to called DMRs (FDR-corrected).</p>
                </div>
                <div class="w-full flex-grow">{enrich_div}</div>
            </div>

            <!-- Pathway Table -->
            <div class="glass-card p-6 rounded-3xl flex flex-col justify-between">
                <div>
                    <h2 class="text-xl font-bold text-slate-100">Enriched Pathway Overlap Database</h2>
                    <p class="text-xs text-slate-400 mt-1 mb-4">Top enriched pathways showing active overlapping genes inside regional epigenetic coordinates.</p>
                </div>
                <div class="overflow-x-auto my-auto">
                    <table class="w-full text-left border-collapse text-sm">
                        <thead>
                            <tr class="border-b border-slate-700 text-slate-400 font-semibold">
                                <th class="py-2 px-3">Pathway</th>
                                <th class="py-2 px-3 text-center">Overlaps</th>
                                <th class="py-2 px-3 text-right">Raw p-val</th>
                                <th class="py-2 px-3 text-right">Genes</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-800 text-slate-300">
                            {"".join([
                                f"<tr class='hover:bg-slate-800/40'>"
                                f"<td class='py-2 px-3 font-bold text-amber-300'>{row['pathway']}</td>"
                                f"<td class='py-2 px-3 text-center font-mono font-bold text-slate-200'>{row['overlap_count']}</td>"
                                f"<td class='py-2 px-3 text-right font-mono'>{row['p_value']:.4e}</td>"
                                f"<td class='py-2 px-3 text-right text-xs truncate max-w-xs text-slate-400' title='{row['overlap_genes']}'>{row['overlap_genes']}</td>"
                                f"</tr>"
                                for row in enrich_df.filter(pl.col("overlap_count") > 0).head(5).to_dicts()
                            ])}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        """ if enrich_df is not None else ""}

        <!-- Tables section: DMRs -->
        <div class="glass-card p-6 rounded-3xl space-y-4">
            <div>
                <h2 class="text-xl font-bold text-slate-100">Top Called Differentially Methylated Regions (DMRs)</h2>
                <p class="text-xs text-slate-400 mt-1">Highest ranking clusters of contiguous differential methylation called in linear density coordinates.</p>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-left border-collapse text-sm">
                    <thead>
                        <tr class="border-b border-slate-700 text-slate-400 font-semibold">
                            <th class="py-3 px-4">Genomic Coordinates</th>
                            <th class="py-3 px-4 text-center">No. CpGs</th>
                            <th class="py-3 px-4 text-right">Methylation Diff</th>
                            <th class="py-3 px-4 text-right">Min P-value</th>
                            <th class="py-3 px-4 text-right">Region Area</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-slate-800 text-slate-300">
                        {"".join([
                            f"<tr class='hover:bg-slate-800/40'>"
                            f"<td class='py-3 px-4 font-mono text-teal-300'>{row['chrom']}:{row['start']}-{row['end']}</td>"
                            f"<td class='py-3 px-4 text-center'>{row['num_sites']}</td>"
                            f"<td class='py-3 px-4 text-right font-bold " + ("text-red-400" if row['mean_diff'] > 0 else "text-blue-400") + f"'>{row['mean_diff']*100:+.2f}%</td>"
                            f"<td class='py-3 px-4 text-right font-mono'>{row['min_p_value']:.4e}</td>"
                            f"<td class='py-3 px-4 text-right'>{row['area']:.4f}</td>"
                            f"</tr>"
                            for row in top_dmrs
                        ]) if len(top_dmrs) > 0 else "<tr><td colspan='5' class='py-8 text-center text-slate-500'>No significant regions called based on current thresholds.</td></tr>"}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>
"""
    # Write to file
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Report successfully compiled and saved to: {output_html_path}")


