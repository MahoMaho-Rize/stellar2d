#!/usr/bin/env python3
"""Plot the MESA-driven ignition trajectory.

Runs ./build/stellar2d run over each profile1..8 (or loads cached values)
and plots T_c(t), ρ_c(t), L_nuc(t). Each point is radial1d's diagnostic
after one integration step from that MESA IC; the trajectory MESA took
to reach that state is summarised by the x-axis (star_age).
"""
from __future__ import annotations
import argparse, subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_one(prof_path, nz=128):
    ic = f"/tmp/mesa_p{Path(prof_path).stem[-1]}.ic"
    subprocess.run(
        ["python3", "/home/kiriko/stellar2d/scripts/convert_mesa_ic.py",
         prof_path, ic], check=True, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    res = subprocess.run([
        "./build/stellar2d", "run", "--solver", "radial1d", "--test", "lane_emden",
        "--nr", str(nz), "--eos", "helmholtz",
        "--ic-mesa", ic, "--ic-mesa-seed-T",
        "--G", "6.674e-8", "--tend", "2", "--output-interval", "1",
        "--nuclear", "--nuc-x", "0.7", "--nuc-t-floor", "0",
        "--nuc-t-scale", "1"],
        cwd="/home/kiriko/stellar2d", capture_output=True, text=True)
    import re
    # line format: "    ignition: T_c=4.142e+06 K  ρ_c=1.415e+00 g/cc  L_nuc=8.783e+28 erg/s  L_nuc/L_surf=..."
    for line in res.stdout.splitlines():
        if "ignition:" in line:
            m_T = re.search(r"T_c=([0-9.eE+-]+)", line)
            m_r = re.search(r"\S_c=([0-9.eE+-]+) g/cc", line)
            m_L = re.search(r"L_nuc=([0-9.eE+-]+)", line)
            if m_T and m_r and m_L:
                return dict(T_c=float(m_T.group(1)),
                            rho_c=float(m_r.group(1)),
                            L_nuc=float(m_L.group(1)))
    return None


def read_mesa_age(prof_path):
    # header block: third line of data, field 'star_age'
    with open(prof_path) as f:
        hdr_names = f.readline().split()
        names = f.readline().split()
        vals = f.readline().split()
    d = dict(zip(names, vals))
    return float(d["star_age"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="/tmp/mesa_work_1Msol/LOGS")
    ap.add_argument("--out",  default="runs/ignition_trajectory.png")
    args = ap.parse_args()

    logs = Path(args.logs)
    profs = sorted(logs.glob("profile*.data"),
                   key=lambda p: int(p.stem.removeprefix("profile")))
    rows = []
    for p in profs:
        age = read_mesa_age(p)
        r = run_one(str(p))
        if r is None:
            print(f"skip {p.name}: no ignition line")
            continue
        rows.append((age, r["T_c"], r["rho_c"], r["L_nuc"]))
        print(f"{p.name} age={age:.3e} yr  T_c={r['T_c']:.3e}  "
              f"ρ_c={r['rho_c']:.3e}  L_nuc={r['L_nuc']:.3e}")

    if not rows:
        print("no data")
        return 1

    age = np.array([r[0] for r in rows])
    T_c = np.array([r[1] for r in rows])
    rho_c = np.array([r[2] for r in rows])
    L_nuc = np.array([r[3] for r in rows])
    L_sun = 3.828e33

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].loglog(age + 1e-4, T_c, "o-", color="#c33")
    axes[0].axhline(1e7, ls="--", color="gray", alpha=0.5, label="H ignition (10⁷ K)")
    axes[0].set_ylabel("T_c [K]")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].loglog(age + 1e-4, rho_c, "s-", color="#39c")
    axes[1].set_ylabel("ρ_c [g/cc]")
    axes[1].grid(alpha=0.3)

    axes[2].loglog(age + 1e-4, L_nuc / L_sun, "^-", color="#693")
    axes[2].axhline(1.0, ls="--", color="gray", alpha=0.5, label="1 L☉")
    axes[2].set_xlabel("MESA star age [yr]")
    axes[2].set_ylabel("L_nuc [L☉]")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.suptitle("radial1d ignition trajectory (MESA 1 M⊙ Z=0.02 ICs)")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
