import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a clean, high-resolution canvas with generous spacing
fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
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

# Function to draw a beautifully proportioned rounded box
def draw_box(ax, x, y, w, h, text, title="", bg_color="#f8fafc", border_color="#cbd5e1"):
    # Using mutation_scale=1 to keep padding and rounding exactly in data coordinates!
    rect = patches.FancyBboxPatch(
        (x, y), w, h, 
        boxstyle="round,pad=0.03,rounding_size=0.08", 
        linewidth=2, 
        edgecolor=border_color, 
        facecolor=bg_color,
        mutation_scale=1
    )
    ax.add_patch(rect)
    if title:
        # Title text
        ax.text(x + w/2, y + h - 0.3, title, ha='center', va='center', fontsize=11, fontweight='bold', color='#0f172a')
        # Subtext details
        ax.text(x + w/2, y + (h - 0.3)/2, text, ha='center', va='center', fontsize=9, color='#334155', linespacing=1.3)
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9.5, color='#334155', linespacing=1.3)

# Function to draw an arrow with clean terminals
def draw_arrow(ax, x1, y1, x2, y2, color="#64748b"):
    ax.annotate(
        "", 
        xy=(x2, y2), 
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=12)
    )

# --- Draw Title ---
ax.text(7, 7.5, "EpiChronos Unified Downstream Analysis Workflow", ha='center', va='center', fontsize=16, fontweight='bold', color='#0f172a')

# --- Column 1: INPUTS ---
ax.text(1.6, 6.8, "1. MULTI-PLATFORM INPUTS", ha='center', va='center', fontsize=11, fontweight='bold', color=border_input)
draw_box(ax, 0.4, 5.0, 2.4, 1.1, "Bismark .cov files\n(GRCh37/hg19)", "WGBS Data", color_input, border_input)
draw_box(ax, 0.4, 3.4, 2.4, 1.1, "Beta values matrix\n(450K / EPIC)", "Microarrays", color_input, border_input)
draw_box(ax, 0.4, 1.8, 2.4, 1.1, "Nanopore / PacBio\n(Methylation frequencies)", "Long-Read Seq", color_input, border_input)

# Connect Inputs to Engine (using box edges with 0.03 padding accounted for)
draw_arrow(ax, 2.83, 5.55, 3.97, 4.3)
draw_arrow(ax, 2.83, 3.95, 3.97, 3.9)
draw_arrow(ax, 2.83, 2.35, 3.97, 3.5)

# --- Column 2: ENGINE ---
ax.text(5.2, 6.8, "2. CORE DATA ENGINE", ha='center', va='center', fontsize=11, fontweight='bold', color=border_engine)
draw_box(ax, 4.0, 2.1, 2.4, 3.6, "Apache Arrow Buffers\n\nVectorized Alignment\n\nColumnar Polars Frame\n\nHigh-Performance RAM", "Polars Core Engine", color_engine, border_engine)

# Connect Engine to Modules
draw_arrow(ax, 6.43, 3.9, 7.67, 5.55)
draw_arrow(ax, 6.43, 3.9, 7.67, 4.25)
draw_arrow(ax, 6.43, 3.9, 7.67, 2.95)
draw_arrow(ax, 6.43, 3.9, 7.67, 1.65)

# --- Column 3: MODULES ---
ax.text(9.0, 6.8, "3. DOWNSTREAM ANALYTICAL MODULES", ha='center', va='center', fontsize=11, fontweight='bold', color=border_module)
draw_box(ax, 7.7, 5.0, 2.6, 1.1, "Vectorized Welch's t-test\n& BH-FDR DMR calling", "DML / DMR Calling", color_module, border_module)
draw_box(ax, 7.7, 3.7, 2.6, 1.1, "Constrained Houseman solver\nIDOL Reference panel", "Cell Deconvolution", color_module, border_module)
draw_box(ax, 7.7, 2.4, 2.6, 1.1, "Horvath / Hannum / EPM\ncoordinate-aware clocks", "Epigenetic Clocks", color_module, border_module)
draw_box(ax, 7.7, 1.1, 2.6, 1.1, "eQTM transcript linkage\nMSigDB pathway ORA", "Functional Annotation", color_module, border_module)

# Connect Modules to Outputs (Top two modules to HTML Report, bottom two to Results JSON)
draw_arrow(ax, 10.33, 5.55, 11.27, 4.7)
draw_arrow(ax, 10.33, 4.25, 11.27, 4.2)
draw_arrow(ax, 10.33, 2.95, 11.27, 2.9)
draw_arrow(ax, 10.33, 1.65, 11.27, 2.4)

# --- Column 4: OUTPUTS ---
ax.text(12.5, 6.8, "4. REPRODUCIBLE OUTPUTS", ha='center', va='center', fontsize=11, fontweight='bold', color=border_output)
draw_box(ax, 11.3, 3.8, 2.4, 1.3, "Self-contained dashboard\nwith Plotly interactive viz\n& full statistical tables", "HTML Dashboard", color_output, border_output)
draw_box(ax, 11.3, 2.0, 2.4, 1.3, "Comprehensive JSON log\nfor pipeline parameters\n& exact output metrics", "Results JSON Log", color_output, border_output)

plt.tight_layout()
output_path = r"C:\Users\rashi\Desktop\PYTHON CODES\new 23\realdata\workflow_schematic.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"[PLOT] Saved: {output_path}")
