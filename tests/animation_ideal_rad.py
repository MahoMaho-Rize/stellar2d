"""
Ideal-radiation EOS focused density animation.

Adds a per-frame north-south asymmetry summary to help diagnose z-axis
dipole-like symmetry breaking.
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
from animate_density import find_default_input_dir, parse_time_from_log, read_vtk, solve_lane_emden


def natural_key(path):
    base = os.path.basename(path)
    m = re.search(r'output_(\d+)\.vtk$', base)
    if m:
        return (0, int(m.group(1)))
    if base == 'output_final.vtk':
        return (1, 10**9)
    return (2, base)


def main():
    parser = argparse.ArgumentParser(description='Animate ideal+radiation EOS diagnostics.')
    parser.add_argument('--input-dir', default=None, help='Directory containing output_*.vtk and run.log')
    parser.add_argument('--fps', type=int, default=3, help='GIF frame rate')
    parser.add_argument('--percentile-max', type=float, default=99.5, help='Upper clipping percentile')
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
    vmin = max(np.min(positive), np.percentile(positive, 1.0))
    vmax = np.percentile(positive, args.percentile_max)
    if vmax <= vmin:
        vmax = np.max(positive)

    fig, (ax_prof, ax_map) = plt.subplots(1, 2, figsize=(12, 5.8))
    first = frames[0][1]
    mesh = ax_map.pcolormesh(
        x_n, z_n, first, shading='flat', cmap='inferno',
        norm=LogNorm(vmin=vmin, vmax=vmax)
    )
    cbar = fig.colorbar(mesh, ax=ax_map)
    cbar.set_label(r'$\rho$ (log scale, clipped)')
    ax_map.set_xlabel('x (= r sin θ)')
    ax_map.set_ylabel('z (= r cos θ)')
    ax_map.set_aspect('equal')
    ax_map.set_title('(b) 2D density map')

    xi_le, theta_le, xi1 = solve_lane_emden(1.5)
    alpha = np.sqrt((1.5 + 1.0) * 1.0 * 1.0**(1.0 / 1.5 - 1.0) / (4 * np.pi * 1.0))
    r_analytic = xi_le * alpha
    rho_analytic = np.where(theta_le > 0, theta_le**1.5, 0.0)
    j_eq = nt // 2
    r_1d = r_cell[:, j_eq]
    rho_init_1d = first[:, j_eq]
    ax_prof.plot(r_analytic, rho_analytic, 'k-', lw=2, label='analytic')
    ax_prof.plot(r_1d, rho_init_1d, 'b--', lw=1.4, label='t = 0')
    current_line, = ax_prof.plot(r_1d, rho_init_1d, 'r.', ms=4, label='current')
    ax_prof.set_xlim(0.0, np.max(r_analytic) * 1.1)
    ax_prof.set_ylim(0.0, max(np.max(rho_analytic), np.max(rho_init_1d)) * 1.05)
    ax_prof.set_xlabel('r')
    ax_prof.set_ylabel(r'$\rho$')
    ax_prof.set_title('(a) Equatorial radial density profile')
    ax_prof.legend(loc='upper right', fontsize=9)

    log_path = os.path.join(input_dir, 'run.log')
    t_end = parse_time_from_log(log_path)
    title = fig.suptitle('')
    asym_text = fig.text(
        0.56, 0.04, '', fontsize=8, family='monospace',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='none')
    )
    footer = fig.text(
        0.01, 0.01,
        f'input: {os.path.relpath(input_dir, repo_root)}',
        fontsize=7, color='#777777', family='monospace'
    )

    def update(frame_idx):
        path, density = frames[frame_idx]
        mesh.set_array(density.ravel())
        current_line.set_ydata(density[:, j_eq])

        pole_mean = 0.5 * (density[:, 0] + density[:, -1])
        equator = density[:, j_eq]
        pole_to_eq = pole_mean / np.maximum(equator, 1e-30)
        north_south = np.abs(density[:, 0] - density[:, -1]) / np.maximum(pole_mean, 1e-30)

        rho_max = np.max(density)
        rho_min = np.min(density[density > 0]) if np.any(density > 0) else 0.0
        base = os.path.basename(path)
        if base == 'output_final.vtk' and t_end is not None:
            frame_title = f'Ideal+Radiation EOS: final (t = {t_end:.3e})'
        else:
            frame_title = f'Ideal+Radiation EOS: {base}'
        title.set_text(frame_title + f'    ρ_min+ = {rho_min:.3e}, ρ_max = {rho_max:.3e}')
        asym_text.set_text(
            f'max pole/equator = {np.max(pole_to_eq):.2e}\n'
            f'max north/south = {np.max(north_south):.2e}'
        )
        return mesh, current_line, title, asym_text, footer

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / max(args.fps, 1), blit=False)

    quantity = 'ideal_rad_density_evolution'
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
    print(f'Color scale: log [{vmin:.3e}, {vmax:.3e}]')


if __name__ == '__main__':
    main()
