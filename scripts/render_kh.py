#!/usr/bin/env python3
"""Fast KH renderer for cart_ale2 VTK -> MP4.

Forked from render_pseudo_spectral.py. 兩個 panel:
  - ω = ∂vy/∂x − ∂vx/∂y  (中心差分由 velocity vector 算出),RdBu_r,對稱於 0
  - |v| = √(vx²+vy²),plasma

VTK 讀 'velocity' VECTORS → velocity_x / velocity_y,周期 BC 下用 np.roll
做中心差分,避開邊界特殊處理。

Usage: python render_kh.py <run_dir> [output.mp4] [fps] [workers]
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
W_FRAME, H_FRAME = 2304, 1152    # 雙 1024² panel 原生尺寸 (+ margin),不下採樣
N_PANELS = 2
PANEL_W = W_FRAME // N_PANELS
DSZ = 1024
BG = (10, 10, 20)
BG_ARR = np.array(BG, dtype=np.uint8)
GAMMA = 5.0 / 3.0

DX, DY = 30, 80
CX = DX + DSZ + 6
CW = 15

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# ── Shared state (set before fork) ─────────────────────────────────────
_cmaps = _panels = None
_nx = _ny = 0
_Lx = _Ly = 1.0
_total_frames = 0
_title = "cart_ale2 KH"
_frame_times = None   # dict: filename-index (int) -> physical time


def parse_vtk_dims(path):
    # Legacy STRUCTURED_GRID: read DIMENSIONS, POINTS, then pull first and
    # last node coordinates directly to derive Lx, Ly.
    with open(path, "rb") as f:
        data = f.read()
    head = data[:4096].decode("ascii", errors="replace")
    m = re.search(r"DIMENSIONS\s+(\d+)\s+(\d+)", head)
    nnx, nny = int(m.group(1)), int(m.group(2))
    # find POINTS block
    pm = re.search(rb"POINTS\s+(\d+)\s+double\s*\n", data[:4096])
    Lx = Ly = 1.0
    if pm:
        npts = int(pm.group(1))
        cursor = pm.end()
        # first point: 3 doubles at cursor
        first = np.frombuffer(data, dtype=">f8", count=3, offset=cursor)
        last_off = cursor + 8 * 3 * (npts - 1)
        last = np.frombuffer(data, dtype=">f8", count=3, offset=last_off)
        Lx = float(last[0] - first[0])
        Ly = float(last[1] - first[1])
    return nnx - 1, nny - 1, Lx, Ly


def _is_binary_vtk(path):
    with open(path, "rb") as f:
        head = f.read(256)
    return b"BINARY\n" in head


def parse_vtk_cells(path, nx, ny, needed=None):
    if _is_binary_vtk(path):
        return _parse_vtk_binary(path, nx, ny, needed)
    with open(path) as f:
        text = f.read()
    pos = text.find("CELL_DATA")
    if pos < 0:
        return None
    return _parse_cell_data(text[pos:], nx, ny, needed)


def _parse_vtk_binary(path, nx, ny, needed=None):
    # Legacy VTK binary: ASCII header, then big-endian raw bytes per section.
    n = nx * ny
    nnode = (nx + 1) * (ny + 1)
    with open(path, "rb") as f:
        data = f.read()
    # find CELL_DATA anchor
    cd = data.find(b"CELL_DATA")
    if cd < 0:
        return None
    # skip past CELL_DATA line
    nl = data.index(b"\n", cd)
    cursor = nl + 1
    fields = {}

    def read_bin_block(cursor, n_doubles):
        # dtype '>f8' = big-endian double
        arr = np.frombuffer(data, dtype=">f8", count=n_doubles, offset=cursor)
        return arr.astype(np.float64), cursor + 8 * n_doubles

    while cursor < len(data):
        # find next SCALARS or VECTORS header line
        m = re.match(rb"(SCALARS|VECTORS)\s+(\S+)", data[cursor:cursor+256])
        if not m:
            break
        kind = m.group(1).decode()
        name = m.group(2).decode()
        eol1 = data.index(b"\n", cursor)
        cursor = eol1 + 1
        if kind == "SCALARS":
            # expect LOOKUP_TABLE line
            eol2 = data.index(b"\n", cursor)
            cursor = eol2 + 1
            vals, cursor = read_bin_block(cursor, n)
            if needed is None or name in needed:
                fields[name] = vals.reshape(ny, nx)
        else:  # VECTORS
            vals, cursor = read_bin_block(cursor, 3 * n)
            v = vals.reshape(ny, nx, 3)
            if needed is None or (name + "_x") in needed or (name + "_y") in needed:
                fields[name + "_x"] = v[:, :, 0]
                fields[name + "_y"] = v[:, :, 1]
        # skip trailing '\n' if present
        if cursor < len(data) and data[cursor:cursor+1] == b"\n":
            cursor += 1
    return fields


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
    # Preserve physical aspect ratio: fit data into a DSZ-tall (or DSZ-wide)
    # image whose longer axis is DSZ. Padding is transparent/black.
    ny_d, nx_d = data.shape
    # Target the longer dim of the data to DSZ; compute paired dim.
    # Using physical extent Lx, Ly would be more principled but data aspect
    # is already ny/nx proportional to Ly/Lx under uniform mesh, which is
    # the only case we render. So use array shape directly.
    aspect = ny_d / nx_d    # data rows / columns → Ly / Lx
    if aspect >= 1.0:       # taller than wide → fit height to DSZ
        out_h = DSZ
        out_w = max(1, int(round(DSZ / aspect)))
    else:
        out_w = DSZ
        out_h = max(1, int(round(DSZ * aspect)))
    zy = out_h / ny_d
    zx = out_w / nx_d
    cart = zoom(data.astype(np.float64), (zy, zx), order=1)
    cart = np.flipud(cart)  # image row 0 = top
    if log_scale:
        cart = np.log10(np.maximum(cart, vmin))
        lo, hi = np.log10(vmin), np.log10(vmax)
    else:
        lo, hi = vmin, vmax
    t = np.clip((cart - lo) / (hi - lo + 1e-30), 0, 1)
    rgba = cmap(t)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    # Pad into (DSZ, DSZ) canvas, centred.
    canvas = np.full((DSZ, DSZ, 3), BG_ARR, dtype=np.uint8)
    y0 = (DSZ - out_h) // 2
    x0 = (DSZ - out_w) // 2
    canvas[y0:y0 + out_h, x0:x0 + out_w] = rgb
    return canvas


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
    needed = {"velocity_x", "velocity_y"}
    fields = parse_vtk_cells(path, _nx, _ny, needed)
    if fields is None or "velocity_x" not in fields:
        return None

    vx = fields["velocity_x"]   # shape (ny, nx)
    vy = fields["velocity_y"]
    # 中心差分 + 周期 wrap (axis=1 → x 方向 = column → use np.roll on axis 1)
    # dvy/dx: vy[j, i+1] - vy[j, i-1]  over 2*dx
    # dvx/dy: vx[j+1, i] - vx[j-1, i]  over 2*dy
    dx = _Lx / _nx
    dy = _Ly / _ny
    dvydx = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2.0 * dx)
    dvxdy = (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2.0 * dy)
    omega = dvydx - dvxdy
    speed = np.sqrt(vx * vx + vy * vy)

    data_map = {"vorticity": omega, "speed": speed}

    frame = Image.new("RGB", (W_FRAME, H_FRAME), BG)
    draw = ImageDraw.Draw(frame)
    fonts = get_fonts()

    m_num = re.search(r"(\d+)", os.path.basename(path))
    fnum = int(m_num.group(1)) if m_num else idx
    t_str = ""
    if _frame_times is not None and fnum in _frame_times:
        t_str = f"   t={_frame_times[fnum]:.4f}"
    draw.text(
        (W_FRAME // 2, 15),
        f"{_title} \u2014 frame {fnum:04d}{t_str}",
        fill=(220, 220, 255), font=fonts["title"], anchor="mt",
    )
    for si in range(1, N_PANELS):
        x = si * PANEL_W
        draw.line([(x, 45), (x, H_FRAME - 30)], fill=(40, 40, 60), width=1)

    for pi, cfg in enumerate(_panels):
        px = pi * PANEL_W
        d = data_map[cfg["field"]]
        cm = _cmaps[cfg["cmap"]]

        vmin, vmax = cfg["vmin"], cfg["vmax"]
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
                else os.path.join(run_dir, "kh.mp4"))
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    n_workers = int(sys.argv[4]) if len(sys.argv) > 4 else min(16, os.cpu_count() or 4)

    t0 = time.time()

    files = sorted(glob.glob(os.path.join(run_dir, "output_????.vtk")))
    files = [f for f in files if not f.endswith("output_0000.vtk")]  # skip stale VTK from pre-sim write_vtk
    if not files:
        print("No cart_ale VTK files found (expect output_0001.vtk, 0002, ...)")
        return

    print(f"Found {len(files)} VTK files in {run_dir}")

    nx, ny, Lx, Ly = parse_vtk_dims(files[0])
    print(f"Grid: {nx}x{ny} cells, domain {Lx:.3f} x {Ly:.3f}")

    # Pre-scan color ranges (抽樣 20 幀) — compute ω in pre-scan too.
    sample_idx = np.linspace(0, len(files) - 1, min(20, len(files))).astype(int)
    w_all, sp_all = [], []
    dx_s = Lx / nx
    dy_s = Ly / ny
    for si in sample_idx:
        f = parse_vtk_cells(files[si], nx, ny, {"velocity_x", "velocity_y"})
        if f is None or "velocity_x" not in f:
            continue
        vx = f["velocity_x"]
        vy = f["velocity_y"]
        dvydx = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2.0 * dx_s)
        dvxdy = (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2.0 * dy_s)
        w_all.append((dvydx - dvxdy).ravel())
        sp_all.append(np.sqrt(vx * vx + vy * vy).ravel())

    w_cat = np.concatenate(w_all)
    w_lim = max(np.percentile(np.abs(w_cat), 99), 1e-5)
    sp_hi = max(np.percentile(np.concatenate(sp_all), 99.5), 0.01)

    print(f"Color ranges: ω=+/-{w_lim:.2f}  |v|=[0, {sp_hi:.3f}]")

    panels = [
        {"field": "vorticity", "cmap": "RdBu_r", "vmin": -w_lim, "vmax": w_lim,
         "log": False, "title": "Vorticity \u03c9"},
        {"field": "speed",     "cmap": "plasma", "vmin": 0,      "vmax": sp_hi,
         "log": False, "title": "Speed |v|"},
    ]

    cmaps = {}
    for p in panels:
        if p["cmap"] not in cmaps:
            cmaps[p["cmap"]] = plt.get_cmap(p["cmap"])

    global _cmaps, _panels, _nx, _ny, _Lx, _Ly, _total_frames, _title, _frame_times
    _cmaps = cmaps
    _panels = panels
    _nx, _ny = nx, ny
    _Lx, _Ly = Lx, Ly
    _total_frames = len(files)
    _title = os.path.basename(os.path.normpath(run_dir))
    # Load frames.csv if the simulator wrote one (VRAM-buffered run).
    _frame_times = None
    frames_csv = os.path.join(run_dir, "frames.csv")
    if os.path.exists(frames_csv):
        m = {}
        with open(frames_csv) as f:
            next(f, None)  # header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    m[int(parts[0])] = float(parts[2])
        if m:
            _frame_times = m
            print(f"  Loaded frames.csv: {len(m)} entries, t=[{min(m.values()):.3f}, {max(m.values()):.3f}]")

    ffproc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "warning",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W_FRAME}x{H_FRAME}", "-r", str(fps),
         "-i", "-",
         # yuv420p 通用兼容;high profile + CQ14 + 高 bitrate 保銳利
         "-c:v", "h264_nvenc", "-preset", "p7", "-cq", "14",
         "-profile:v", "high", "-pix_fmt", "yuv420p",
         "-rc", "vbr", "-b:v", "0", "-maxrate", "30M", "-bufsize", "60M",
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
