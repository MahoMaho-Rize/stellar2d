#!/usr/bin/env python3
"""從既有 VTK 幀重算 E(k),寫 `spectrum.csv`(匹配 solver 實時輸出格式)。

動機:過去的 pseudo_spectral / cart_ale2 kh_shear runs 已經 rm 了 VTK 或即將
刪除 —— 先跑這個腳本把譜資訊「燒進」永久 CSV,之後隨便 `find -name '*.vtk'
-delete`。

與 solver 端 (pseudo_spectral_solver.cu:compute_spectrum_bins) 完全同規則:
  dk     = 2π / max(Lx, Ly)
  nbins  = min(nx, ny) // 2 + 1
  E_mode = (Lx·Ly / N⁴) · ½·(|û|²+|v̂|²)
  bin    = round(|k|/dk)
  E_k    = Σ_bin / dk   (密度,Σ E_k·dk = KE)

Usage:
  scripts/backfill_spectrum_csv.py <run_dir> [--force]

若 spectrum.csv 已存在會跳過(--force 覆蓋)。
"""
import os, sys, glob, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_pseudo_spectral import parse_vtk_dims, parse_vtk_cells
from spectrum_pseudo_spectral import compute_spectrum, load_frame_times, frame_number


def load_uv(path, nx, ny):
    """嘗試多種欄位命名 — pseudo_spectral 存 velocity_{x,y};cart_ale 也同。"""
    fields = parse_vtk_cells(path, nx, ny, {"velocity_x", "velocity_y"})
    if fields is None or "velocity_x" not in fields:
        raise RuntimeError(f"缺 velocity 欄位: {path}")
    return fields["velocity_x"], fields["velocity_y"]


def is_binary_vtk(p):
    try:
        with open(p, "rb") as fh:
            return b"BINARY\n" in fh.read(256)
    except Exception:
        return False


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(1)
    run_dir = sys.argv[1].rstrip("/")
    force = ("--force" in sys.argv[2:])

    out_csv = os.path.join(run_dir, "spectrum.csv")
    if os.path.exists(out_csv) and not force:
        print(f"已有 {out_csv},跳過 (--force 覆蓋)")
        return

    vtks = sorted(glob.glob(os.path.join(run_dir, "output_????.vtk")))
    vtks = [p for p in vtks if os.path.getsize(p) > 1024 and is_binary_vtk(p)]
    if not vtks:
        print(f"{run_dir}: 無 binary VTK,跳過")
        return

    nx, ny = parse_vtk_dims(vtks[0])
    ftimes = load_frame_times(run_dir)
    frames_csv_has = os.path.exists(os.path.join(run_dir, "frames.csv"))

    # dk / nbins 規則與 solver 一致(假 Lx=Ly=1 — 目前 pseudo_spectral 固定)
    Lx = Ly = 1.0
    dk = 2.0 * np.pi / max(Lx, Ly)
    nbins = min(nx, ny) // 2 + 1
    k_axis = np.arange(nbins) * dk

    print(f"{run_dir}: {len(vtks)} frames, grid {nx}×{ny}, nbins={nbins}, dk={dk:.3f}")

    with open(out_csv, "w") as fh:
        fh.write("index,step,t")
        for kv in k_axis:
            fh.write(f",{kv:.6e}")
        fh.write("\n")

        for path in vtks:
            fnum = frame_number(path)
            u, v = load_uv(path, nx, ny)
            k_c, E_c, KE = compute_spectrum(u, v, Lx, Ly)
            # compute_spectrum 的 bin 規則和 solver 一致 → 直接用
            assert len(E_c) == nbins, f"bin 數不符 {len(E_c)} vs {nbins}"
            t = ftimes.get(fnum)
            step = fnum  # step 資訊在 frames.csv 的第二欄;若缺就用 fnum 代
            if frames_csv_has and t is None:
                t = 0.0
            if t is None:
                t = 0.0
            fh.write(f"{fnum},{step},{t:.10e}")
            for ev in E_c:
                fh.write(f",{ev:.6e}")
            fh.write("\n")

    print(f"寫入 {out_csv}  ({len(vtks)} frames)")


if __name__ == "__main__":
    main()
