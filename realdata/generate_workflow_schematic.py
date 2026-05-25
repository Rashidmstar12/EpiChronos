import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis('off')

# Colors
color_input = '#eff6ff'  # light blue
border_input = '#3b82f6' # blue
color_engine = '#ecfdf5' # light green
border_engine = '#10b981' # green
color_module = '#faf5ff' # light purple
border_module = '#8b5cf6' # purple
color_output = '#fff7ed' # light orange
border_output = '#f97316' # orange

# Function to draw a box
def draw_box(ax, x, y, w, h, text, title="", bg_color="#f8fafc", border_color="#cbd5e1"):
    rect = patches.FancyBboxPatch(
        (x, y), w, h, 
        boxstyle="round,pad=0.1", 
        linewidth=2, 
        edgecolor=border_color, 
        facecolor=bg_color,
        mutation_scale=15
    )
    ax.add_patch(rect)
    if title:
        ax.text(x + w/2, y + h - 0.35, title, ha='center', va='center', fontsize=11, fontweight='bold', color='#1e293b')
        ax.text(x + w/2, y + (h - 0.35)/2, text, ha='center', va='center', fontsize=9.5, color='#475569')
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='medium', color='#1e293b')

# Function to draw an arrow
def draw_arrow(ax, x1, y1, x2, y2, color="#64748b"):
    ax.annotate(
        "", 
        xy=(x2, y2), 
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=15)
    )

# --- 1. Draw Columns ---

# Title
ax.text(6, 6.5, "EpiChronos Unified Downstream Analysis Workflow", ha='center', va='center', fontsize=16, fontweight='bold', color='#0f172a')

# Column 1: INPUTS
ax.text(1.25, 5.7, "1. MULTI-PLATFORM INPUTS", ha='center', va='center', fontsize=11, fontweight='bold', color=border_input)
draw_box(ax, 0.2, 4.3, 2.1, 0.9, "Bismark .cov files\n(GRCh37/hg19)", "WGBS Data", color_input, border_input)
draw_box(ax, 0.2, 3.0, 2.1, 0.9, "Beta values matrix\n(450K / EPIC)", "Microarrays", color_input, border_input)
draw_box(ax, 0.2, 1.7, 2.1, 0.9, "Nanopore / PacBio\n(Methylation frequencies)", "Long-Read Seq", color_input, border_input)

# Connect Inputs to Engine
draw_arrow(ax, 2.4, 4.75, 3.3, 3.8)
draw_arrow(ax, 2.4, 3.45, 3.3, 3.45)
draw_arrow(ax, 2.4, 2.15, 3.3, 3.1)

# Column 2: ENGINE
ax.text(4.4, 5.7, "2. CORE DATA ENGINE", ha='center', va='center', fontsize=11, fontweight='bold', color=border_engine)
draw_box(ax, 3.4, 2.2, 2.0, 2.5, "Apache Arrow Buffers\n\nVectorized Alignment\n\nColumnar Polars Frame\n\nHigh-Performance RAM", "Polars Core", color_engine, border_engine)

# Connect Engine to Modules
draw_arrow(ax, 5.5, 3.45, 6.4, 4.75)
draw_arrow(ax, 5.5, 3.45, 6.4, 3.65)
draw_arrow(ax, 5.5, 3.45, 6.4, 2.55)
draw_arrow(ax, 5.5, 3.45, 6.4, 1.45)

# Column 3: MODULES
ax.text(7.65, 5.7, "3. DOWNSTREAM ANALYTICAL MODULES", ha='center', va='center', fontsize=11, fontweight='bold', color=border_module)
draw_box(ax, 6.5, 4.3, 2.3, 0.9, "Vectorized Welch's t-test\n& BH-FDR DMR calling", "DML/DMR calling", color_module, border_module)
draw_box(ax, 6.5, 3.2, 2.3, 0.9, "Constrained Houseman solver\nIDOL Reference panel", "Deconvolution", color_module, border_module)
draw_box(ax, 6.5, 2.1, 2.3, 0.9, "Horvath / Hannum / EPM\ncoordinate-aware clocks", "Epigenetic Clocks", color_module, border_module)
draw_box(ax, 6.5, 1.0, 2.3, 0.9, "eQTM transcript linkage\nMSigDB pathway ORA", "Functional Annotation", color_module, border_module)

# Connect Modules to Outputs
draw_arrow(ax, 8.9, 4.75, 9.6, 3.8)
draw_arrow(ax, 8.9, 3.65, 9.6, 3.6)
draw_arrow(ax, 8.9, 2.55, 9.6, 3.3)
draw_arrow(ax, 8.9, 1.45, 9.6, 3.1)

# Column 4: OUTPUTS
ax.text(10.75, 5.7, "4. REPRODUCIBLE OUTPUTS", ha='center', va='center', fontsize=11, fontweight='bold', color=border_output)
draw_box(ax, 9.7, 3.0, 2.1, 1.0, "Self-contained dashboard\nwith Plotly interactive viz\n& full statistical tables", "HTML Report", color_output, border_output)
draw_box(ax, 9.7, 1.7, 2.1, 1.0, "Comprehensive JSON log\nfor pipeline parameters\n& exact output metrics", "Results JSON", color_output, border_output)

plt.tight_layout()
output_path = r"C:\Users\rashi\Desktop\PYTHON CODES\new 23\realdata\workflow_schematic.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"[PLOT] Saved: {output_path}")
