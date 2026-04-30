#!/usr/bin/env python3
"""Fast 1080p renderer for cart_ale Cartesian-structured VTK -> MP4.

Adapted from render_video_fast.py. Differences:
  - Cartesian grid (x, y) — no (r, theta) -> (x, z) mapping needed.
  - VTK cell arrays are stored jc-outer, ic-inner → shape (ny, nx).
  - Entropy baseline s0(y) = <P/rho^gamma>_x from frame 0 (HSE column).

Usage: python render_cart_ale.py <run_dir> [output.mp4] [fps] [workers]
"""

import numpy as np
import re, os, sys, glob, subprocess, time
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import zoom
import multiprocessing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Layout ──────────────────────────────────────────────────────────────
W_FRAME, H_FRAME = 1920, 1080
N_PANELS = 3
PANEL_W = W_FRAME // N_PANELS
DSZ = 540
BG = (10, 10, 20)
BG_ARR = np.array(BG, dtype=np.uint8)
GAMMA = 5.0 / 3.0

DX, DY = 30, 80
CX = DX + DSZ + 6
CW = 15

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# ── Shared state (set before fork) ─────────────────────────────────────
_s0 = _cmaps = _panels = None
_nx = _ny = 0
_total_frames = 0
_title = "cart_ale"


def parse_vtk_dims(path):
    with open(path) as f:
        head = f.read(4096)
    m = re.search(r"DIMENSIONS\s+(\d+)\s+(\d+)", head)
    nnx, nny = int(m.group(1)), int(m.group(2))
    return nnx - 1, nny - 1


def parse_vtk_cells(path, nx, ny, needed=None):
    with open(path) as f:
        text = f.read()
    pos = text.find("CELL_DATA")
    if pos < 0:
        return None
    return _parse_cell_data(text[pos:], nx, ny, needed)


def _parse_cell_data(text, nx, ny, needed=None):
    n = nx * ny
    fields = {}
    markers = []
    for m in re.finditer(r"^(SCALARS|VECTORS)\s+(\S+)", text, re.MULTILINE):
        markers.append((m.start(), m.end(), m.group(1), m.group(2)))

    for i, (start, hdr_end, kind, name) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        if kind == "SCALARS":
            if needed and name not in needed:
                continue
            eol = text.index("\n", hdr_end)
            data_start = eol + 1
            if text[data_start:data_start + 12] == "LOOKUP_TABLE":
                data_start = text.index("\n", data_start) + 1
            vals = np.fromstring(text[data_start:end], sep=" ")
            if len(vals) >= n:
                # cart_ale writes jc outer, ic inner → reshape (ny, nx)
                fields[name] = vals[:n].reshape(ny, nx)
        else:
            if needed and (name + "_x") not in needed and (name + "_y") not in needed:
                continue
            eol = text.index("\n", hdr_end)
            data_start = eol + 1
            vals = np.fromstring(text[data_start:end], sep=" ")
            if len(vals) >= n * 3:
                v = vals[:n * 3].reshape(ny, nx, 3)
                fields[name + "_x"] = v[:, :, 0]
                fields[name + "_y"] = v[:, :, 1]
    return fields


def field_to_rgb(data, cmap, vmin, vmax, log_scale):
    # Upscale (ny, nx) → (DSZ, DSZ) and flip so y grows upward in image.
    zy = DSZ / data.shape[0]
    zx = DSZ / data.shape[1]
    cart = zoom(data.astype(np.float64), (zy, zx), order=1)
    cart = np.flipud(cart)  # image row 0 = top
    if log_scale:
        cart = np.log10(np.maximum(cart, vmin))
        lo, hi = np.log10(vmin), np.log10(vmax)
    else:
        lo, hi = vmin, vmax
    t = np.clip((cart - lo) / (hi - lo + 1e-30), 0, 1)
    rgba = cmap(t)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def make_cbar(cmap, h):
    t = np.linspace(1, 0, h)[:, None].repeat(CW, axis=1)
    return (cmap(t)[:, :, :3] * 255).astype(np.uint8)


def fmt_val(v):
    a = abs(v)
    if a == 0:
        return "0"
    if a < 0.001 or a >= 1e4:
        return f"{v:.1e}"
    if a < 1:
        return f"{v:.3f}"
    return f"{v:.2f}"


_font_cache = {}


def get_fonts():
    if not _font_cache:
        _font_cache["title"] = ImageFont.truetype(FONT_B, 26)
        _font_cache["panel"] = ImageFont.truetype(FONT_B, 20)
        _font_cache["tick"] = ImageFont.truetype(FONT_R, 14)
        _font_cache["small"] = ImageFont.truetype(FONT_R, 13)
    return _font_cache


def render_one(args):
    idx, path = args
    needed = {"density", "pressure", "mach"}
    fields = parse_vtk_cells(path, _nx, _ny, needed)
    if fields is None or "density" not in fields:
        return None

    rho = np.maximum(fields["density"], 1e-20)
    P = np.maximum(fields.get("pressure", np.ones((_ny, _nx))), 1e-30)
    mach = fields.get("mach", np.zeros((_ny, _nx)))
    # entropy baseline is (ny,) — broadcast over x
    ds = (P / rho ** GAMMA - _s0[:, None]) / np.maximum(_s0[:, None], 1e-20)

    data_map = {"density": rho, "entropy_pert": ds, "mach": mach}

    ds_amp = max(float(np.max(np.abs(ds))), 1e-8)
    mach_max = max(float(np.max(mach)), 1e-8)

    frame = Image.new("RGB", (W_FRAME, H_FRAME), BG)
    draw = ImageDraw.Draw(frame)
    fonts = get_fonts()

    m_num = re.search(r"(\d+)", os.path.basename(path))
    fnum = int(m_num.group(1)) if m_num else idx
    draw.text(
        (W_FRAME // 2, 15),
        f"{_title} \u2014 frame {fnum:04d}",
        fill=(220, 220, 255), font=fonts["title"], anchor="mt",
    )
    for si in range(1, N_PANELS):
        x = si * PANEL_W
        draw.line([(x, 45), (x, H_FRAME - 30)], fill=(40, 40, 60), width=1)

    for pi, cfg in enumerate(_panels):
        px = pi * PANEL_W
        d = data_map[cfg["field"]]
        cm = _cmaps[cfg["cmap"]]

        # Per-frame dynamic range for perturbation fields
        vmin, vmax = cfg["vmin"], cfg["vmax"]
        if cfg["field"] == "entropy_pert":
            vmin, vmax = -ds_amp, ds_amp
        elif cfg["field"] == "mach":
            vmin, vmax = 0.0, mach_max

        rgb = field_to_rgb(d, cm, vmin, vmax, cfg["log"])
        frame.paste(Image.fromarray(rgb), (px + DX, DY))

        cbar = make_cbar(cm, DSZ)
        frame.paste(Image.fromarray(cbar), (px + CX, DY))

        draw.text(
            (px + PANEL_W // 2, 52),
            cfg["title"],
            fill=(200, 200, 220), font=fonts["panel"], anchor="mt",
        )

        n_ticks = 5
        for ti_idx in range(n_ticks):
            frac = ti_idx / (n_ticks - 1)
            y = DY + int(frac * (DSZ - 1))
            if cfg["log"]:
                lo, hi = np.log10(vmin), np.log10(vmax)
                v = 10 ** (hi - frac * (hi - lo))
                txt = f"{v:.0e}"
            else:
                v = vmax - frac * (vmax - vmin)
                txt = fmt_val(v)
            draw.text((px + CX + CW + 3, y), txt,
                      fill=(170, 170, 190), font=fonts["tick"], anchor="lm")

        raw_min = float(np.nanmin(d))
        raw_max = float(np.nanmax(d))
        draw.text((px + PANEL_W // 2, DY + DSZ + 14),
                  f"[{fmt_val(raw_min)}, {fmt_val(raw_max)}]",
                  fill=(140, 140, 160), font=fonts["small"], anchor="mt")

    bar_y = H_FRAME - 18
    bar_x0, bar_x1 = 60, W_FRAME - 60
    draw.rectangle([(bar_x0, bar_y), (bar_x1, bar_y + 6)], fill=(30, 30, 50))
    progress = (idx + 1) / _total_frames
    draw.rectangle(
        [(bar_x0, bar_y), (bar_x0 + int(progress * (bar_x1 - bar_x0)), bar_y + 6)],
        fill=(80, 120, 200),
    )
    draw.text((W_FRAME // 2, bar_y - 4),
              f"frame {idx + 1} / {_total_frames}",
              fill=(120, 120, 150), font=fonts["small"], anchor="mb")

    return np.asarray(frame).tobytes()


def main():
    run_dir = sys.argv[1]
    out_path = (sys.argv[2] if len(sys.argv) > 2
                else os.path.join(run_dir, "cart_ale_fast.mp4"))
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    n_workers = int(sys.argv[4]) if len(sys.argv) > 4 else min(16, os.cpu_count() or 4)

    t0 = time.time()

    files = sorted(glob.glob(os.path.join(run_dir, "output_????.vtk")))
    files = [f for f in files if not f.endswith("output_0000.vtk")]  # skip stale VTK from pre-sim write_vtk
    if not files:
        print("No cart_ale VTK files found (expect output_0001.vtk, 0002, ...)")
        return

    print(f"Found {len(files)} VTK files in {run_dir}")

    nx, ny = parse_vtk_dims(files[0])
    print(f"Grid: {nx}x{ny} cells")

    flds0 = parse_vtk_cells(files[0], nx, ny, {"density", "pressure"})
    rho0 = np.maximum(flds0["density"], 1e-20)
    P0 = np.maximum(flds0["pressure"], 1e-30)
    # average entropy over x to get a 1D baseline s0(y)
    s0 = np.mean(P0 / rho0 ** GAMMA, axis=1)

    # Pre-scan color ranges
    sample_idx = np.linspace(0, len(files) - 1, min(20, len(files))).astype(int)
    ds_all, mach_all, rho_all = [], [], []
    for si in sample_idx:
        f = parse_vtk_cells(files[si], nx, ny, {"density", "pressure", "mach"})
        if f is None:
            continue
        rho = np.maximum(f["density"], 1e-20)
        P = np.maximum(f.get("pressure", np.ones((ny, nx))), 1e-30)
        ds = (P / rho ** GAMMA - s0[:, None]) / np.maximum(s0[:, None], 1e-20)
        ds_all.append(ds.ravel())
        mach_all.append(f.get("mach", np.zeros((ny, nx))).ravel())
        rho_all.append(rho.ravel())

    ds_cat = np.concatenate(ds_all)
    ds_lim = max(np.percentile(np.abs(ds_cat), 99), 1e-5)
    mach_hi = max(np.percentile(np.concatenate(mach_all), 100), 0.01)
    rho_lo = max(np.percentile(np.concatenate(rho_all), 0.5), 1e-8)
    rho_hi = max(np.percentile(np.concatenate(rho_all), 99.9), rho_lo * 10)

    print(f"Color ranges: rho=[{rho_lo:.2e}, {rho_hi:.2e}]  ds/s0=+/-{ds_lim:.4f}  Mach=[0, {mach_hi:.3f}]")

    panels = [
        {"field": "density",      "cmap": "inferno", "vmin": rho_lo, "vmax": rho_hi,
         "log": True,  "title": "Density \u03c1 (log)"},
        {"field": "entropy_pert", "cmap": "RdBu_r",  "vmin": -ds_lim, "vmax": ds_lim,
         "log": False, "title": "Entropy \u03b4s/s\u2080"},
        {"field": "mach",         "cmap": "plasma",  "vmin": 0, "vmax": mach_hi,
         "log": False, "title": "Mach number"},
    ]

    cmaps = {}
    for p in panels:
        if p["cmap"] not in cmaps:
            cmaps[p["cmap"]] = plt.get_cmap(p["cmap"])

    global _s0, _cmaps, _panels, _nx, _ny, _total_frames, _title
    _s0 = s0
    _cmaps = cmaps
    _panels = panels
    _nx, _ny = nx, ny
    _total_frames = len(files)
    _title = os.path.basename(os.path.normpath(run_dir))

    ffproc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "warning",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W_FRAME}x{H_FRAME}", "-r", str(fps),
         "-i", "-",
         "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "18",
         "-pix_fmt", "yuv420p",
         out_path],
        stdin=subprocess.PIPE,
    )

    args = [(i, f) for i, f in enumerate(files)]
    rendered = 0
    t_render = time.time()
    print(f"Rendering {len(files)} frames @ {fps}fps using {n_workers} workers ...")

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        for raw in pool.imap(render_one, args, chunksize=4):
            if raw is not None:
                ffproc.stdin.write(raw)
                rendered += 1
                if rendered % 50 == 0:
                    el = time.time() - t_render
                    spd = rendered / el
                    eta = (len(files) - rendered) / spd
                    print(f"  {rendered}/{len(files)} ({spd:.1f} fps, ~{eta:.0f}s left)")

    ffproc.stdin.close()
    ret = ffproc.wait()
    if ret != 0:
        print(f"ffmpeg exited with code {ret}")
        return

    total = time.time() - t0
    sz = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Done: {out_path}")
    print(f"  {rendered} frames, {sz:.1f} MB, {total:.1f}s total ({rendered/total:.1f} fps)")


if __name__ == "__main__":
    main()
