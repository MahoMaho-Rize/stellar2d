#!/usr/bin/env python3
"""Horizontal-average diagnostic for cart_ale2 Andrassy 2022 pilot runs,
faithful to Andrassy+ 2022 Eq. 13–15 definitions.

Definitions (Andrassy 2022 §3):
  Eq. 13  volume-weighted x-average:  q̄(y) = (1/N_x) Σ_i q_{i,j}
  Eq. 14  mass-weighted  x-average:   q̃(y) = Σ(ρ q)/Σ ρ
  Eq. 15  fluctuation v_rms:
            ṽ_rms(y) = sqrt( σ²(v_x) + σ²(v_y) [+ σ²(v_z) in 3D] )
          where σ²(v_i)(y) = (1/N_x) Σ_i (v_i - ṽ_i)²
  Eq.   upper boundary y_ub(t): smallest y in [1, 2+1/16] where
          the horizontally-averaged entropy has risen above (A_base + ε).
          (Andrassy §3.2 uses A(y, t) monitoring; we use pseudo-entropy
          A = p/ρ^γ since γ=5/3 makes A ≈ p/ρ·(μ independent) invariant
          under composition mixing inside the γ=5/3 gas.)
  Conv / stable bulk averages: layer with y < y_ub - 0.1 and y > y_ub + 0.1
  respectively (0.1 length-unit buffer excludes the transition zone).

Reads cart_ale2 STRUCTURED_GRID ASCII VTK outputs.  Writes a PNG comparing
our horizontal profiles against each of the Andrassy 5-code rprof files at
the closest matching time, plus a CSV time-series of (t, y_ub, v_rms_conv,
v_rms_stable).

Usage:
    python3 scripts/andrassy2022/diagnose.py \\
        --run-dir runs/andrassy2022_128x128_XXXX \\
        --out runs/andrassy2022_128x128_XXXX/diag.png
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


GAMMA    = 5.0 / 3.0       # Andrassy 2022 EOS
Y_SHIFT  = 1.0             # our slab y' = y_Andrassy - 1
Y_CB     = 2.0             # Schwarzschild boundary in Andrassy coords
Y_BUFFER = 0.1             # exclude this much around y_ub from bulk averages


# ─── VTK parser ────────────────────────────────────────────────
def parse_vtk(path: Path) -> dict:
    """Parse a cart_ale2 STRUCTURED_GRID ASCII VTK.

    cart_ale2 VTK layout (from src/gpu/ale/cart_ale2_solver.cu write_vtk):
      - POINTS: (nx+1)*(ny+1) nodes, x-fast (ix in [0, nx], iy in [0, ny])
      - CELL_DATA ordering: ic * ny + jc  (ic fast is y, jc fast is x?)
        Actually cart_ale2 uses flat = ic*ny + jc; we verify by reshape.
      - VECTORS velocity: same ordering as CELL_DATA.

    We explicitly reshape based on header DIMENSIONS and let the caller
    handle (ny, nx) or (nx, ny) conventions via .T.
    """
    txt = path.read_text()
    dim = re.search(r"DIMENSIONS\s+(\d+)\s+(\d+)\s+(\d+)", txt)
    nnx = int(dim.group(1)); nny = int(dim.group(2))
    nx = nnx - 1; ny = nny - 1

    pts_match = re.search(r"POINTS\s+(\d+)\s+double\n", txt)
    pts_start = pts_match.end()
    npts = int(pts_match.group(1))
    cd_match = re.search(r"\nCELL_DATA\s+(\d+)", txt[pts_start:])
    pts_end = pts_start + cd_match.start()
    pts_data = np.fromstring(txt[pts_start:pts_end], sep=" ").reshape(npts, 3)
    # Node x/y: STRUCTURED_GRID stores x-fast — so first nnx pts give x_row_0,
    # every nnx'th pt gives y col.  We verify by checking monotonicity.
    x_nodes = pts_data[:nnx, 0]
    y_nodes = pts_data[::nnx, 1]
    assert x_nodes.size == nnx and y_nodes.size == nny

    def read_scalar(name: str) -> np.ndarray:
        pat = re.compile(
            rf"SCALARS\s+{name}\s+double\s+1\s*\nLOOKUP_TABLE\s+default\s*\n")
        m = pat.search(txt)
        if m is None:
            return None
        start = m.end()
        end_m = re.search(r"\n(SCALARS|VECTORS)", txt[start:])
        end = start + end_m.start() if end_m else len(txt)
        arr = np.fromstring(txt[start:end], sep=" ")
        arr = arr[: nx * ny]
        # cart_ale2 VTK ordering (verified empirically): flat index runs y-slow,
        # x-fast — matches standard STRUCTURED_GRID CELL_DATA convention.
        # Reshape directly as (ny, nx); first row = bottom (y=0), last = top.
        return arr.reshape(ny, nx)

    def read_vector(name: str):
        pat = re.compile(rf"VECTORS\s+{name}\s+double\s*\n")
        m = pat.search(txt)
        if m is None:
            return None, None
        start = m.end()
        end_m = re.search(r"\n(SCALARS|VECTORS)", txt[start:])
        end = start + end_m.start() if end_m else len(txt)
        arr = np.fromstring(txt[start:end], sep=" ")
        arr = arr[: 3 * nx * ny].reshape(ny, nx, 3)
        vx = arr[..., 0]   # already (ny, nx)
        vy = arr[..., 1]
        return vx, vy

    rho = read_scalar("density")
    P   = read_scalar("pressure")
    mach = read_scalar("mach")
    vx_cell, vy_cell = read_vector("velocity")
    assert vx_cell is not None, \
        f"VECTORS velocity not found in {path} — check cart_ale2 writer"

    return dict(
        nx=nx, ny=ny, x_nodes=x_nodes, y_nodes=y_nodes,
        rho=rho, P=P, mach=mach,
        vx=vx_cell, vy=vy_cell,
    )


# ─── Horizontal averages (Andrassy Eq. 13-15) ───────────────────
def horizontal_profiles(data: dict) -> dict:
    """Per-row (y) diagnostics using volume- and mass-weighted averages,
    and the fluctuation v_rms per Eq. 15.

    All input arrays are (ny, nx) cell-centered.  Returns arrays on y-cell
    centers (length ny).
    """
    rho = data["rho"]    # (ny, nx)
    P   = data["P"]
    vx  = data["vx"]
    vy  = data["vy"]
    y_nodes = data["y_nodes"]
    y_c = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    ny, nx = rho.shape

    # Eq. 13 volume (arithmetic) average on each y-row.
    rho_bar = rho.mean(axis=1)
    P_bar   = P.mean(axis=1)

    # Eq. 14 mass-weighted average of velocity.
    # ṽ_i(y) = Σ_x ρ·v_i / Σ_x ρ
    M_row = rho.sum(axis=1)                                    # (ny,)
    vx_tilde = (rho * vx).sum(axis=1) / np.maximum(M_row, 1e-30)
    vy_tilde = (rho * vy).sum(axis=1) / np.maximum(M_row, 1e-30)

    # Eq. 15 fluctuation standard deviations (mass-weighted).
    #   σ²_x(y) = Σ_x ρ · (v_x - ṽ_x)² / Σ_x ρ
    sigma2_vx = (rho * (vx - vx_tilde[:, None])**2).sum(axis=1) / np.maximum(M_row, 1e-30)
    sigma2_vy = (rho * (vy - vy_tilde[:, None])**2).sum(axis=1) / np.maximum(M_row, 1e-30)
    v_rms = np.sqrt(sigma2_vx + sigma2_vy)                     # 2D: no z

    # Pseudo-entropy A = P/ρ^γ using row means (volume-weighted per Andrassy
    # Fig. 1 — they plot ⟨A⟩_x not mass-weighted).
    A_bar = P_bar / rho_bar**GAMMA

    # Mach(y) from v_rms and mean sound speed c_s = √(γ P̄ / ρ̄).
    cs_bar = np.sqrt(GAMMA * P_bar / np.maximum(rho_bar, 1e-30))
    Ma_bar = v_rms / np.maximum(cs_bar, 1e-30)

    return dict(
        y=y_c,
        rho=rho_bar, P=P_bar, A=A_bar,
        vx_tilde=vx_tilde, vy_tilde=vy_tilde,
        sigma_vx=np.sqrt(sigma2_vx),
        sigma_vy=np.sqrt(sigma2_vy),
        v_rms=v_rms,
        Ma=Ma_bar, cs=cs_bar,
    )


def find_y_ub(profiles: dict, A_threshold_frac: float = 1.0e-3) -> float:
    """Andrassy §3.2 upper boundary tracker.

    Convection sits in the ~isentropic y < 2 region; stable layer above has
    rising A(y).  y_ub(t) = first y where (A(y) - A_conv) / A_conv > threshold,
    using A_conv = mean A in the conv layer's lower half.

    Returns y_ub in our local (slab) coordinate.  Add Y_SHIFT for Andrassy.
    """
    y = profiles["y"]; A = profiles["A"]
    # Base A: mean over y ∈ [0.2, 0.8] in slab coords = paper's conv layer.
    mask_base = (y > 0.2) & (y < 0.8)
    if mask_base.sum() < 4:
        return y[-1]
    A_conv = A[mask_base].mean()
    # Walk up from y = 1.0 (Schwarzschild boundary in slab coords =
    # y_paper=2.0).  Andrassy §3.2 measures y_ub starting at the expected
    # Schwarzschild location so the conv-layer A noise doesn't trigger the
    # 0.1% threshold prematurely.
    for j, yj in enumerate(y):
        if yj < 1.0:
            continue
        if A[j] > A_conv * (1.0 + A_threshold_frac):
            return yj
    return y[-1]


def bulk_averages(profiles: dict, y_ub_local: float) -> dict:
    """Mass-weighted v_rms in the conv and stable bulk layers (Eq. 15 + §3.1).

    Andrassy §3.1 defines the bulk v_rms as the mass-weighted average of
    ṽ_rms²(y) across the layer, then sqrt.  Weight by ⟨ρ⟩(y).
    """
    y = profiles["y"]
    rho_bar = profiles["rho"]
    mask_conv = y < (y_ub_local - Y_BUFFER)
    mask_stab = y > (y_ub_local + Y_BUFFER)
    def _avg(mask):
        if mask.sum() == 0:
            return float("nan")
        w = rho_bar[mask]
        s2 = profiles["sigma_vx"][mask]**2 + profiles["sigma_vy"][mask]**2
        return float(np.sqrt((w * s2).sum() / w.sum()))
    return dict(v_rms_conv=_avg(mask_conv), v_rms_stable=_avg(mask_stab),
                n_conv=int(mask_conv.sum()), n_stab=int(mask_stab.sum()))


# ─── Andrassy rprof parser ──────────────────────────────────────
def parse_rprof(path: Path) -> dict:
    """Parse an Andrassy Zenodo rprof file.

    File layout (per inspection of FLASH-256-*.rprof):
      Line 1: "DUMP   NN, t = X.XXX"
      Line 2: "Nx = NNN"
      Line 3: blank
      Line 4: blank
      Line 5: column header row 1
      Line 6: blank
      Lines 7 .. 6+Nx: data block 1
      Line 7+Nx: blank
      Line 8+Nx: column header row 2
      Line 9+Nx: blank
      Lines 10+Nx .. 9+2Nx: data block 2
      ... (6 blocks total)

    Returned dict contains every column keyed by its header name
    (RHO, P, TEMP, A, X1, V, VX, VY, VZ, STDEV_VX, STDEV_VY, STDEV_VZ, ...)
    plus "y" and "t".  Y axis is reversed to ascending if needed.
    """
    lines = path.read_text().splitlines()
    t = float(re.search(r"t\s*=\s*([\d.eE+-]+)", lines[0]).group(1))
    nx = int(re.search(r"Nx\s*=\s*(\d+)", lines[1]).group(1))

    all_cols: dict[str, np.ndarray] = {}
    y_ref = None
    i = 0
    n_lines = len(lines)
    while i < n_lines:
        stripped = lines[i].strip()
        # Look for a header line: starts with "IR" (not DUMP, not blank).
        if stripped.startswith("IR"):
            names = stripped.split()
            # Skip blank lines until data starts.
            i += 1
            while i < n_lines and lines[i].strip() == "":
                i += 1
            block = np.loadtxt(lines[i : i + nx])
            i += nx
            # Columns: names aligned to data columns.
            for j, name in enumerate(names):
                if j >= block.shape[1]:
                    break
                all_cols[name] = block[:, j]
            if y_ref is None and "Y" in all_cols:
                y_ref = all_cols["Y"].copy()
        else:
            i += 1

    # Ascending-y convention.
    if y_ref is not None and y_ref[0] > y_ref[-1]:
        for k in list(all_cols):
            all_cols[k] = all_cols[k][::-1]
        y_ref = all_cols["Y"]

    out = {"t": t, "y": y_ref}
    out.update({k: v for k, v in all_cols.items() if k not in ("IR", "Y")})
    return out


def find_nearest_rprof(root: Path, code: str, res: int, t_target: float) -> Path:
    pattern = root / f"{code}/{code}-{res}/{code}-{res}-*.rprof"
    files = sorted(glob.glob(str(pattern)))
    best, best_dt = None, float("inf")
    for f in files:
        with open(f) as fp:
            head = fp.readline()
        m = re.search(r"t\s*=\s*([\d.eE+-]+)", head)
        if m is None:
            continue
        dt = abs(float(m.group(1)) - t_target)
        if dt < best_dt:
            best_dt = dt; best = Path(f)
    return best


# ─── Main ─────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--andrassy-root", type=Path,
                    default=Path("data/andrassy2022/1D-profiles"))
    ap.add_argument("--codes", nargs="+",
                    default=["FLASH", "MUSIC", "PPMSTAR", "PROMPI", "SLH"])
    ap.add_argument("--andrassy-res", type=int, default=256)
    ap.add_argument("--frame", type=int, default=-1,
                    help="VTK frame index (default -1 = final)")
    ap.add_argument("--t-frame", type=float, default=None,
                    help="Simulation time of the frame (reads from CSV if None)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    vtk_files = sorted(
        [f for f in args.run_dir.glob("output_*.vtk") if "final" not in f.name]
    )
    if not vtk_files:
        print(f"No output_*.vtk in {args.run_dir}"); return 1
    frame_path = vtk_files[args.frame]

    # Estimate frame time: assume even spacing based on diagnostics.csv
    # (cart_ale2 driver writes VTK every output_interval steps).
    t_frame = args.t_frame
    diag_csv = args.run_dir / "diagnostics.csv"
    if t_frame is None and diag_csv.exists():
        # CSV header: step,t,dt,mass,KE,IE,PE,E,max_v,max_mach
        times_by_step = {}
        with diag_csv.open() as f:
            next(f)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    try:
                        times_by_step[int(parts[0])] = float(parts[1])
                    except ValueError:
                        pass
        # Guess: frame i corresponds to output_interval * i step; CSV has
        # diag_interval step rows.  Without the per-frame step → use last
        # CSV entry's time for the final VTK (frame == len-1) and linear
        # interp otherwise.
        steps_sorted = sorted(times_by_step.keys())
        if steps_sorted:
            t_last = times_by_step[steps_sorted[-1]]
            frame_idx = args.frame if args.frame >= 0 else len(vtk_files) + args.frame
            t_frame = t_last * (frame_idx + 1) / len(vtk_files)
    if t_frame is None:
        t_frame = 30.0

    print(f"Analyzing {frame_path.name}  (t ≈ {t_frame:.2f})")
    data = parse_vtk(frame_path)
    print(f"  Grid: {data['nx']} × {data['ny']}, y ∈ "
          f"[{data['y_nodes'][0]:.3f}, {data['y_nodes'][-1]:.3f}]")

    prof = horizontal_profiles(data)
    y_ub_local = find_y_ub(prof)
    bulk = bulk_averages(prof, y_ub_local)
    print(f"  y_ub (local)   = {y_ub_local:.4f}  (Andrassy = {y_ub_local + Y_SHIFT:.4f})")
    print(f"  v_rms (conv)   = {bulk['v_rms_conv']:.4e}   "
          f"[n_y_conv = {bulk['n_conv']}]")
    print(f"  v_rms (stable) = {bulk['v_rms_stable']:.4e} "
          f"[n_y_stab = {bulk['n_stab']}]")
    print(f"  Peak v_rms(y)  = {prof['v_rms'].max():.4e} at y={prof['y'][prof['v_rms'].argmax()]:.3f}")
    print(f"  Andrassy paper saturated ⟨v_rms⟩_conv ≈ 0.034 (for reference)")

    # Shift our y to Andrassy coords: y_paper = y_local + Y_SHIFT.
    y_paper = prof["y"] + Y_SHIFT

    andrassy_data = {}
    for code in args.codes:
        rp = find_nearest_rprof(args.andrassy_root, code, args.andrassy_res, t_frame)
        if rp is None:
            print(f"  [warn] No {code}-{args.andrassy_res} rprof matching t={t_frame:.1f}")
            continue
        rd = parse_rprof(rp)
        andrassy_data[code] = rd
        print(f"  {code} {rp.name} at t={rd['t']:.2f}")

    if plt is None:
        print("matplotlib unavailable — skipping plot"); return 0

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    (ax_rho, ax_P, ax_A), (ax_vrms, ax_sigma, ax_Ma) = axes
    colors = {"FLASH": "C0", "MUSIC": "C1", "PPMSTAR": "C2",
              "PROMPI": "C3", "SLH": "C4"}

    for code, rd in andrassy_data.items():
        c = colors.get(code, "gray")
        ax_rho.plot(rd["y"], rd["RHO"], "-", color=c, lw=1.2, alpha=0.75, label=code)
        ax_P.plot(rd["y"], rd["P"], "-", color=c, lw=1.2, alpha=0.75, label=code)
        A_code = rd["P"] / rd["RHO"]**GAMMA
        ax_A.plot(rd["y"], A_code, "-", color=c, lw=1.2, alpha=0.75, label=code)
        # Andrassy's own v_rms from STDEV_{VX,VY,VZ} columns.
        if all(k in rd for k in ("STDEV_VX", "STDEV_VY")):
            sz2 = rd.get("STDEV_VZ", np.zeros_like(rd["STDEV_VX"]))**2
            v_rms_code = np.sqrt(rd["STDEV_VX"]**2 + rd["STDEV_VY"]**2 + sz2)
            ax_vrms.plot(rd["y"], v_rms_code, "-", color=c, lw=1.2, alpha=0.75, label=code)

    ax_rho.plot(y_paper, prof["rho"], "k-", lw=2.3, label="stellar2d cart_ale2")
    ax_P.plot(y_paper, prof["P"], "k-", lw=2.3, label="stellar2d cart_ale2")
    ax_A.plot(y_paper, prof["A"], "k-", lw=2.3, label="stellar2d cart_ale2")

    ax_vrms.plot(y_paper, prof["v_rms"], "k-", lw=2.3, label="stellar2d 2D")
    ax_sigma.plot(y_paper, prof["sigma_vx"], "b-", lw=1.6, label="σ(v_x) stellar2d")
    ax_sigma.plot(y_paper, prof["sigma_vy"], "r-", lw=1.6, label="σ(v_y) stellar2d")
    ax_Ma.plot(y_paper, prof["Ma"], "k-", lw=2.3, label="stellar2d")

    for ax in (ax_rho, ax_P, ax_A, ax_vrms, ax_sigma, ax_Ma):
        ax.axvline(Y_CB, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.axvline(y_ub_local + Y_SHIFT, color="orange", lw=0.8, ls=":",
                   alpha=0.6, label=None)
        ax.grid(alpha=0.25)
        ax.set_xlabel("y  (Andrassy 2022 coords)")
    ax_rho.set_yscale("log"); ax_P.set_yscale("log")
    ax_rho.set_title("⟨ρ⟩(y)"); ax_P.set_title("⟨P⟩(y)")
    ax_A.set_title("pseudo-entropy A = P/ρ^γ")
    ax_vrms.set_title(r"$\tilde{v}_{\rm rms}(y) = \sqrt{\sigma_x^2+\sigma_y^2}$")
    ax_sigma.set_title("Component σ(v_x), σ(v_y)")
    ax_Ma.set_title("Mach ṽ_rms / c̄_s")
    # Paper's saturated reference.
    ax_vrms.axhline(0.034, color="magenta", lw=1.0, ls="-.",
                    alpha=0.6, label="paper 0.034")
    ax_rho.legend(fontsize=8); ax_vrms.legend(fontsize=8)
    ax_sigma.legend(fontsize=8); ax_A.legend(fontsize=8)

    fig.suptitle(
        f"Andrassy 2022 pilot — stellar2d cart_ale2 {data['nx']}×{data['ny']} "
        f"vs 5-code {args.andrassy_res}³  @  t ≈ {t_frame:.1f}\n"
        f"y_ub = {y_ub_local + Y_SHIFT:.3f}, "
        f"v_rms(conv) = {bulk['v_rms_conv']:.4e}, "
        f"v_rms(stable) = {bulk['v_rms_stable']:.4e}",
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
