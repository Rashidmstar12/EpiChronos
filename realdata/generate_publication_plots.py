import matplotlib.pyplot as plt
import numpy as np
import os

# Data
cpgs = np.array([500, 1000, 3000, 5000])  # in thousands (K)
cpgs_labels = ['500K', '1M', '3M', '5M']

ec_ram = np.array([61.1, 122.2, 367.7, 614.2]) # MB
r_ram = np.array([2.1 * 1024, 4.2 * 1024, 9.8 * 1024, 18.0 * 1024]) # MB

ec_time = np.array([0.77, 1.65, 5.86, 10.35]) # seconds
r_time = np.array([45, 120, 380, 720]) # seconds

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

# Colors
color_ec = '#0d9488' # teal
color_r = '#dc2626' # red-ish
marker_size = 8

# Left Panel: RAM Comparison
ax1.plot(cpgs, r_ram, color=color_r, marker='o', markersize=marker_size, linewidth=2, label='bsseq / R')
ax1.plot(cpgs, ec_ram, color=color_ec, marker='s', markersize=marker_size, linewidth=2, label='EpiChronos')
ax1.set_title('Peak RAM Usage Comparison', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Dataset Size (Number of CpGs)', fontsize=10)
ax1.set_ylabel('Memory Consumption (MB)', fontsize=10)
ax1.set_xticks(cpgs)
ax1.set_xticklabels(cpgs_labels)
ax1.set_yscale('log') # Log scale is perfect since there is a ~30x reduction!
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0')

# Annotate fold reduction
for i, (ec, r) in enumerate(zip(ec_ram, r_ram)):
    fold = r / ec
    ax1.annotate(f"{fold:.1f}x", (cpgs[i], ec), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, fontweight='bold', color=color_ec)

# Right Panel: Runtime Comparison
ax2.plot(cpgs, r_time, color=color_r, marker='o', markersize=marker_size, linewidth=2, label='bsseq / R')
ax2.plot(cpgs, ec_time, color=color_ec, marker='s', markersize=marker_size, linewidth=2, label='EpiChronos')
ax2.set_title('Execution Time Comparison', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Dataset Size (Number of CpGs)', fontsize=10)
ax2.set_ylabel('Execution Time (seconds)', fontsize=10)
ax2.set_xticks(cpgs)
ax2.set_xticklabels(cpgs_labels)
ax2.set_yscale('log') # Log scale is perfect since there is a ~60x speedup!
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0')

# Annotate speedup fold
for i, (ec, r) in enumerate(zip(ec_time, r_time)):
    fold = r / ec
    ax2.annotate(f"{fold:.1f}x", (cpgs[i], ec), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, fontweight='bold', color=color_ec)

plt.tight_layout()
output_path = r"C:\Users\rashi\Desktop\PYTHON CODES\new 23\realdata\benchmark_performance.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"[PLOT] Saved: {output_path}")
