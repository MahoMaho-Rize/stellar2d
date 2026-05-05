#!/usr/bin/env python3
"""Standalone test: given v IC as pure Fourier-Dirichlet mode
    v(x,y) = sin(k_x x) sin(k_y π y / Ly), u=0
the exact Chorin projection sends v → v · k_x²/(k_x² + k_y²).

This script extracts v from CUDA's kh_final.csv after an IC-only projection
(i.e. t=0 run) and checks the projection ratio.
"""
import sys
import numpy as np
from pathlib import Path

csv = Path(sys.argv[1])
# Parse CSV same way as verify_anelastic_sl_kh.py
lines = csv.read_text().splitlines()
hdr = lines[0]
meta = dict(kv.strip().split("=") for kv in hdr[2:].split(",") if "=" in kv)
nx = int(meta["nx"]); ny = int(meta["ny"])
Lx = float(meta["Lx"]); Ly = float(meta["Ly"])
blocks = {"y": None, "omega": None, "u": None, "v": None}
cur = None
buf = []
def push():
    if cur is not None:
        blocks[cur] = np.array(buf, dtype=float)
for ln in lines[1:]:
    s = ln.strip()
    if s.startswith("#"):
        push(); buf = []
        if "y_cgl" in s:    cur_new = "y"
        elif "omega" in s:  cur_new = "omega"
        elif s.endswith("u (ny × nx row-major):"): cur_new = "u"
        elif s.endswith("v (ny × nx row-major):"): cur_new = "v"
        else: cur_new = None
        cur = cur_new
    elif cur is not None and s:
        buf.append(float(s))
push()

y = blocks["y"].reshape(ny)
u = blocks["u"].reshape(ny, nx)
v = blocks["v"].reshape(ny, nx)

# IC was vshear=0, amp=1e-3, k_x_int=1 (sin(kx_phys·x)·Gaussian-bump taper)
# But we want a pure sine IC — that's init_kh_shear's Gaussian-bump
# shape × wall taper × amp * sin(kx·x), not a pure sin(k_y π y/Ly).
# So this test wasn't set up for pure mode.

# Instead compare: extract Fourier component at k_x=1 (first nonzero bin)
# and inspect its y-shape.
U_hat = np.fft.rfft(u, axis=1) / nx * 2   # factor 2 for imag-sin component
V_hat = np.fft.rfft(v, axis=1) / nx * 2
# We expect v_hat[k=1](y) dominant, imaginary (since v ∝ sin(k_x x))
print(f"grid: {nx}x{ny}, Lx={Lx}, Ly={Ly}")
print(f"|v|∞ = {np.max(np.abs(v)):.4e}")
for k in range(4):
    print(f"  k_x={k}: max|Re U_hat| = {np.max(np.abs(U_hat[:,k].real)):.3e}, "
          f"max|Im V_hat| = {np.max(np.abs(V_hat[:,k].imag)):.3e}")
