#!/usr/bin/env python3
"""Plot gmode_probe.csv to see the v(t) signal shape."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

csv = Path(sys.argv[1])
data = np.loadtxt(csv, comments="#")
t, v = data[:, 0], data[:, 1]
print(f"loaded {len(t)} samples, t ∈ [{t[0]}, {t[-1]}],  |v|_max = {np.max(np.abs(v)):.3e}")

fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(t[:2000], v[:2000], lw=0.6)
ax[0].set_xlabel("t")
ax[0].set_ylabel("v_center")
ax[0].set_title("first 2000 samples")
ax[0].grid(alpha=0.3)

# zoom on early, mid, late
stride = max(1, len(t) // 3000)
ax[1].plot(t[::stride], v[::stride], lw=0.5)
ax[1].set_xlabel("t")
ax[1].set_ylabel("v_center")
ax[1].set_title(f"full range (stride {stride})")
ax[1].grid(alpha=0.3)

out = csv.with_suffix(".png")
fig.tight_layout()
fig.savefig(out, dpi=120)
print(f"  saved {out}")
