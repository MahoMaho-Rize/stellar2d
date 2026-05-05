#!/usr/bin/env python3
"""Plot radial1d diagnostics + profile evolution.

Usage:
  scripts/plot_radial1d.py runs/<run_dir>

Produces:
  <run_dir>/evolution.png    — diagnostics vs t (M, E, KE/IE, Mach, H/He)
  <run_dir>/profiles.png     — radial profiles at 6 snapshots (rho, T-proxy, v, X)
"""
import sys, os, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)

rd = sys.argv[1]
csv_path = os.path.join(rd, "diagnostics.csv")
prof_paths = sorted(glob.glob(os.path.join(rd, "profile_*.txt")))

# --- diagnostics ---
import csv
with open(csv_path) as f:
    rows = list(csv.DictReader(f))
if not rows:
    print("No diagnostics rows", file=sys.stderr); sys.exit(1)

t = np.array([float(r["t"]) for r in rows])
M = np.array([float(r["mass"]) for r in rows])
E = np.array([float(r["total_E"]) for r in rows])
KE = np.array([float(r["KE"]) for r in rows])
IE = np.array([float(r["IE"]) for r in rows])
PE = np.array([float(r["PE"]) for r in rows])
Mach = np.array([float(r["max_mach"]) for r in rows])
has_species = "mass_H" in rows[0]
if has_species:
    mH = np.array([float(r["mass_H"])  for r in rows])
    mHe= np.array([float(r["mass_He"]) for r in rows])
    Xc = np.array([float(r["X_core"]) for r in rows])
    Xs = np.array([float(r["X_surf"]) for r in rows])

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
axs[0,0].plot(t, (M - M[0]) / M[0]); axs[0,0].set_title("(M − M₀)/M₀  (mass drift)")
axs[0,0].set_xlabel("t [s]"); axs[0,0].grid()
axs[0,1].plot(t, E, label="E_total")
axs[0,1].plot(t, IE, "--", label="IE"); axs[0,1].plot(t, PE, ":", label="PE")
axs[0,1].set_title("Energies"); axs[0,1].set_xlabel("t [s]"); axs[0,1].legend(); axs[0,1].grid()
axs[0,2].semilogy(t, KE); axs[0,2].set_title("KE (log)")
axs[0,2].set_xlabel("t [s]"); axs[0,2].grid()
axs[1,0].plot(t, Mach); axs[1,0].set_title("max Mach")
axs[1,0].set_xlabel("t [s]"); axs[1,0].grid()
if has_species:
    axs[1,1].plot(t, Xc, label="X_core")
    axs[1,1].plot(t, Xs, label="X_surf")
    axs[1,1].set_title("Hydrogen mass fraction"); axs[1,1].set_xlabel("t [s]")
    axs[1,1].legend(); axs[1,1].grid()
    axs[1,2].plot(t, (mH - mH[0]) / mH[0], label="ΔM_H / M_H₀")
    axs[1,2].plot(t, (mHe- mHe[0])/ mHe[0], label="ΔM_He/ M_He₀")
    axs[1,2].set_title("Species drift"); axs[1,2].set_xlabel("t [s]")
    axs[1,2].legend(); axs[1,2].grid()
else:
    axs[1,1].axis("off"); axs[1,2].axis("off")

fig.suptitle(f"radial1d diagnostics — {os.path.basename(rd.rstrip('/'))}")
fig.tight_layout()
evo_out = os.path.join(rd, "evolution.png")
fig.savefig(evo_out, dpi=110)
print(f"wrote {evo_out}")

# --- profiles at 6 snapshots ---
if not prof_paths:
    sys.exit(0)
picks = [0, len(prof_paths)//5, 2*len(prof_paths)//5,
         3*len(prof_paths)//5, 4*len(prof_paths)//5, len(prof_paths)-1]
picks = sorted(set(p for p in picks if p < len(prof_paths)))
fig, axs = plt.subplots(2, 2, figsize=(13, 9))
cmap = plt.cm.viridis
for j, i in enumerate(picks):
    path = prof_paths[i]
    with open(path) as f:
        first = f.readline()
    m = re.search(r"t\s*=\s*([0-9eE.+-]+)", first)
    t_s = float(m.group(1)) if m else j
    # skip header rows starting with '#'
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.split()
            # last face has "-" for cell quantities; skip
            if "-" in parts[3:6]: continue
            data.append([float(x) for x in parts])
    if not data: continue
    d = np.array(data)
    r, v, rho, P, e = d[:,1], d[:,2], d[:,3], d[:,4], d[:,5]
    color = cmap(j / max(len(picks)-1, 1))
    label = f"t={t_s:.3g}s"
    axs[0,0].semilogy(r, rho,  color=color, label=label)
    axs[0,1].semilogy(r, P,    color=color, label=label)
    axs[1,0].plot    (r, v,    color=color, label=label)
    if has_species and d.shape[1] >= 8:
        X = d[:,6]
        axs[1,1].plot(r, X, color=color, label=label)

axs[0,0].set_title("ρ(r)"); axs[0,0].set_xlabel("r [cm]"); axs[0,0].set_ylabel("g/cc")
axs[0,1].set_title("P(r)"); axs[0,1].set_xlabel("r [cm]"); axs[0,1].set_ylabel("erg/cc")
axs[1,0].set_title("v(r)"); axs[1,0].set_xlabel("r [cm]"); axs[1,0].set_ylabel("cm/s")
axs[1,1].set_title("X(r) hydrogen") if has_species else axs[1,1].axis("off")
axs[1,1].set_xlabel("r [cm]") if has_species else None
for a in axs.ravel(): a.grid(True, alpha=0.3); a.legend(fontsize=8)
fig.suptitle(f"radial1d profiles — {os.path.basename(rd.rstrip('/'))}")
fig.tight_layout()
prof_out = os.path.join(rd, "profiles.png")
fig.savefig(prof_out, dpi=110)
print(f"wrote {prof_out}")
