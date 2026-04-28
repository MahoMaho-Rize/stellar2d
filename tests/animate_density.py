"""
Create a density evolution GIF from stellar2d VTK outputs.

Usage examples:
    python animate_density.py
    python animate_density.py --input-dir ../build-mac
    python animate_density.py --input-dir ../runs/idealrad_64x32_t0.04
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
import numpy as np

from provenance import build_filename


def read_vtk(path):
    with open(path) as f:
        lines = f.readlines()

    dims = None
    for line in lines:
        if line.startswith('DIMENSIONS'):
            dims = list(map(int, line.split()[1:]))
            nt = dims[0] - 1
            nr = dims[1] - 1
            break
    if dims is None:
        raise RuntimeError(f'Could not find DIMENSIONS in {path}')

    idx = next(i for i, line in enumerate(lines) if line.startswith('CELL_DATA'))
    ncells = int(lines[idx].split()[1])

    data = {}
    i = idx + 1
    while i < len(lines):
        if lines[i].startswith('SCALARS'):
            name = lines[i].split()[1]
            i += 2
            vals = []
            while len(vals) < ncells:
                vals.extend(map(float, lines[i].split()))
                i += 1
            data[name] = np.array(vals).reshape(nr, nt)
        elif lines[i].startswith('VECTORS'):
            break
        else:
            i += 1

    idx_pts = next(i for i, line in enumerate(lines) if line.startswith('POINTS'))
    npts = int(lines[idx_pts].split()[1])
    coords = []
    j = idx_pts + 1
    while len(coords) < npts * 3:
        coords.extend(map(float, lines[j].split()))
        j += 1
    coords = np.array(coords).reshape(nr + 1, nt + 1, 3)

    return nr, nt, data, coords


def parse_time_from_log(log_path):
    if not os.path.exists(log_path):
        return None
    last_t = None
    with open(log_path) as f:
        for line in f:
            m = re.search(r't\s*=\s*([\d.e+-]+)', line)
            if m:
                last_t = float(m.group(1))
    return last_t


def solve_lane_emden(n_poly, dxi=0.001):
    xi_arr, theta_arr = [0.0], [1.0]
    xi, theta, dtheta = 1e-10, 1.0, 0.0
    while theta > 0 and xi < 100:
        def f2(x, t, dt):
            if x < 1e-10:
                return -t / 3.0
            return -(t**n_poly if t > 0 else 0.0) - 2 * dt / x
        k1y1, k1y2 = dtheta, f2(xi, theta, dtheta)
        k2y1 = dtheta + 0.5 * dxi * k1y2
        k2y2 = f2(xi + 0.5 * dxi, theta + 0.5 * dxi * k1y1, dtheta + 0.5 * dxi * k1y2)
        k3y1 = dtheta + 0.5 * dxi * k2y2
        k3y2 = f2(xi + 0.5 * dxi, theta + 0.5 * dxi * k2y1, dtheta + 0.5 * dxi * k2y2)
        k4y1 = dtheta + dxi * k3y2
        k4y2 = f2(xi + dxi, theta + dxi * k3y1, dtheta + dxi * k3y2)
        theta += dxi / 6.0 * (k1y1 + 2 * k2y1 + 2 * k3y1 + k4y1)
        dtheta += dxi / 6.0 * (k1y2 + 2 * k2y2 + 2 * k3y2 + k4y2)
        xi += dxi
        xi_arr.append(xi)
        theta_arr.append(max(theta, 0.0))
        if theta <= 0:
            break
    return np.array(xi_arr), np.array(theta_arr), xi


def find_default_input_dir(repo_root):
    candidates = [
        os.path.join(repo_root, 'build'),
        os.path.join(repo_root, 'build-mac'),
        repo_root,
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, 'output_0000.vtk')):
            return candidate
    return candidates[0]


def natural_key(path):
    base = os.path.basename(path)
    m = re.search(r'output_(\d+)\.vtk$', base)
    if m:
        return (0, int(m.group(1)))
    if base == 'output_final.vtk':
        return (1, 10**9)
    return (2, base)


def main():
    parser = argparse.ArgumentParser(description='Animate stellar2d density output.')
    parser.add_argument('--input-dir', default=None,
                        help='Directory containing output_*.vtk and optional run.log')
    parser.add_argument('--fps', type=int, default=3, help='GIF frame rate')
    parser.add_argument('--log-scale', action='store_true',
                        help='Use logarithmic color scale (default on for density-like data)')
    parser.add_argument('--linear-scale', action='store_true',
                        help='Force linear color scale')
    parser.add_argument('--percentile-max', type=float, default=99.5,
                        help='Upper clipping percentile for color scale')
    args = parser.parse_args()

    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    input_dir = args.input_dir or find_default_input_dir(repo_root)

    vtk_files = sorted(glob.glob(os.path.join(input_dir, 'output_*.vtk')), key=natural_key)
    if not vtk_files:
        raise RuntimeError(f'No output_*.vtk files found in {input_dir}')

    frames = []
    x_n = z_n = None
    r_cell = None
    nr = nt = None
    for path in vtk_files:
        nr_i, nt_i, data, coords = read_vtk(path)
        if nr is None:
            nr, nt = nr_i, nt_i
            x_n = coords[:, :, 0]
            z_n = coords[:, :, 2]
            r_nodes = np.sqrt(x_n**2 + z_n**2)
            r_cell = 0.25 * (
                r_nodes[:-1, :-1] + r_nodes[1:, :-1] +
                r_nodes[:-1, 1:] + r_nodes[1:, 1:]
            )
        elif nr_i != nr or nt_i != nt:
            raise RuntimeError(
                f'Mismatched grid in {os.path.basename(path)}: expected {nr}x{nt}, got {nr_i}x{nt_i}'
            )
        frames.append((path, data['density']))

    density_stack = np.stack([frame for _, frame in frames], axis=0)
    positive = density_stack[density_stack > 0]
    if positive.size == 0:
        raise RuntimeError('Density field is non-positive everywhere.')

    vmin = max(np.min(positive), np.percentile(positive, 1.0))
    vmax = np.percentile(positive, args.percentile_max)
    if vmax <= vmin:
        vmax = np.max(positive)

    use_log = not args.linear_scale
    norm = LogNorm(vmin=vmin, vmax=vmax) if use_log else None

    fig, (ax_prof, ax_map) = plt.subplots(1, 2, figsize=(12, 5.8))
    first = frames[0][1]
    mesh = ax_map.pcolormesh(
        x_n, z_n, first, shading='flat', cmap='inferno',
        norm=norm, vmin=None if use_log else vmin, vmax=None if use_log else vmax
    )
    cbar = fig.colorbar(mesh, ax=ax_map)
    cbar.set_label(r'$\rho$' + (' (log scale, clipped)' if use_log else ' (linear scale, clipped)'))
    ax_map.set_xlabel('x (= r sin θ)')
    ax_map.set_ylabel('z (= r cos θ)')
    ax_map.set_aspect('equal')

    n_poly, rho_c, K, G = 1.5, 1.0, 1.0, 1.0
    xi_le, theta_le, xi1 = solve_lane_emden(n_poly)
    alpha = np.sqrt((n_poly + 1) * K * rho_c**(1.0 / n_poly - 1) / (4 * np.pi * G))
    r_analytic = xi_le * alpha
    rho_analytic = rho_c * np.where(theta_le > 0, theta_le**n_poly, 0.0)
    j_eq = nt // 2
    r_1d = r_cell[:, j_eq]
    rho_init_1d = first[:, j_eq]

    analytic_line, = ax_prof.plot(r_analytic, rho_analytic, 'k-', lw=2, label='analytic')
    init_line, = ax_prof.plot(r_1d, rho_init_1d, 'b--', lw=1.4, label='t = 0')
    current_line, = ax_prof.plot(r_1d, rho_init_1d, 'r.', ms=4, label='current')
    ax_prof.set_xlim(0.0, np.max(r_analytic) * 1.1)
    ax_prof.set_ylim(0.0, max(np.max(rho_analytic), np.max(rho_init_1d)) * 1.05)
    ax_prof.set_xlabel('r')
    ax_prof.set_ylabel(r'$\rho$')
    ax_prof.set_title('(a) Radial density profile')
    ax_prof.legend(loc='upper right', fontsize=9)

    log_path = os.path.join(input_dir, 'run.log')
    t_end = parse_time_from_log(log_path)
    title = fig.suptitle('')
    footer = fig.text(
        0.01, 0.01,
        f'input: {os.path.relpath(input_dir, repo_root)}',
        fontsize=7, color='#777777', family='monospace'
    )

    def update(frame_idx):
        path, density = frames[frame_idx]
        mesh.set_array(density.ravel())
        current_line.set_ydata(density[:, j_eq])
        rho_max = np.max(density)
        rho_min = np.min(density[density > 0]) if np.any(density > 0) else 0.0
        base = os.path.basename(path)
        if base == 'output_final.vtk' and t_end is not None:
            frame_title = f'Density evolution: final (t = {t_end:.3e})'
        else:
            frame_title = f'Density evolution: {base}'
        title.set_text(frame_title + f'    ρ_min+ = {rho_min:.3e}, ρ_max = {rho_max:.3e}')
        ax_map.set_title('(b) 2D density map')
        return mesh, current_line, analytic_line, init_line, title, footer

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / max(args.fps, 1), blit=False)

    quantity = 'density_evolution'
    tend_for_name = t_end if t_end is not None else 0.0
    out_name = build_filename('lane_emden', quantity, nr, nt, tend_for_name, ext='gif')
    out_dir = os.path.join(repo_root, 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    anim.save(out_path, writer=PillowWriter(fps=args.fps))
    plt.close(fig)

    print(f'Saved: {out_path}')
    print(f'Input dir: {input_dir}')
    print(f'Frames: {len(frames)}')
    print(f'Color scale: {"log" if use_log else "linear"} [{vmin:.3e}, {vmax:.3e}]')


if __name__ == '__main__':
    main()
