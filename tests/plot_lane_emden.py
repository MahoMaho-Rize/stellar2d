"""
Lane-Emden hydrostatic equilibrium verification plot.
Reads VTK output from stellar2d and compares against the analytic solution.
Naming convention: see docs/provenance.md
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import re, glob, os
from provenance import build_filename, add_provenance_footer

# ── Parse a single VTK structured-grid file ──────────────────────────
def read_vtk(path):
    with open(path) as f:
        lines = f.readlines()

    for l in lines:
        if l.startswith('DIMENSIONS'):
            dims = list(map(int, l.split()[1:]))
            nt = dims[0] - 1
            nr = dims[1] - 1
            break

    idx = next(i for i, l in enumerate(lines) if l.startswith('CELL_DATA'))
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
            name = lines[i].split()[1]
            i += 1
            vals = []
            while len(vals) < ncells * 3:
                vals.extend(map(float, lines[i].split()))
                i += 1
            data[name] = np.array(vals).reshape(nr, nt, 3)
        else:
            i += 1

    # node coordinates
    idx_pts = next(i for i, l in enumerate(lines) if l.startswith('POINTS'))
    npts = int(lines[idx_pts].split()[1])
    coords = []
    j = idx_pts + 1
    while len(coords) < npts * 3:
        coords.extend(map(float, lines[j].split()))
        j += 1
    coords = np.array(coords).reshape(nr + 1, nt + 1, 3)

    x_n = coords[:, :, 0]
    z_n = coords[:, :, 2]
    r_nodes = np.sqrt(x_n**2 + z_n**2)
    r_cell = 0.25 * (r_nodes[:-1, :-1] + r_nodes[1:, :-1] +
                      r_nodes[:-1, 1:] + r_nodes[1:, 1:])

    return nr, nt, r_cell, data, coords

# ── Lane-Emden analytic solution ─────────────────────────────────────
def solve_lane_emden(n_poly, dxi=0.001):
    xi_arr, theta_arr = [0.0], [1.0]
    xi, theta, dtheta = 1e-10, 1.0, 0.0
    while theta > 0 and xi < 100:
        def f2(x, t, dt):
            if x < 1e-10: return -t / 3.0
            return -(t**n_poly if t > 0 else 0.0) - 2 * dt / x
        k1y1, k1y2 = dtheta, f2(xi, theta, dtheta)
        k2y1 = dtheta + 0.5*dxi*k1y2
        k2y2 = f2(xi+0.5*dxi, theta+0.5*dxi*k1y1, dtheta+0.5*dxi*k1y2)
        k3y1 = dtheta + 0.5*dxi*k2y2
        k3y2 = f2(xi+0.5*dxi, theta+0.5*dxi*k2y1, dtheta+0.5*dxi*k2y2)
        k4y1 = dtheta + dxi*k3y2
        k4y2 = f2(xi+dxi, theta+dxi*k3y1, dtheta+dxi*k3y2)
        theta += dxi/6*(k1y1+2*k2y1+2*k3y1+k4y1)
        dtheta += dxi/6*(k1y2+2*k2y2+2*k3y2+k4y2)
        xi += dxi
        xi_arr.append(xi); theta_arr.append(max(theta, 0.0))
        if theta <= 0: break
    return np.array(xi_arr), np.array(theta_arr), xi

# ── Main ─────────────────────────────────────────────────────────────
repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

def find_output_dir():
    candidates = [
        os.path.join(repo_root, 'build'),
        os.path.join(repo_root, 'build-mac'),
        repo_root,
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, 'output_0000.vtk')):
            return candidate
    return candidates[0]

parser = argparse.ArgumentParser(description='Plot Lane-Emden verification diagnostics.')
parser.add_argument('--input-dir', default=None,
                    help='Directory containing output_*.vtk and optional run.log')
args = parser.parse_args()

build_dir = args.input_dir or find_output_dir()

vtk_init = os.path.join(build_dir, 'output_0000.vtk')
vtk_files = sorted(glob.glob(os.path.join(build_dir, 'output_*.vtk')))
vtk_last = max(vtk_files, key=os.path.getmtime) if vtk_files else vtk_init

nr, nt, r_init, data_init, coords_init = read_vtk(vtk_init)
nr_final, nt_final, r_final, data_final, coords_final = read_vtk(vtk_last)
if nt_final != nt:
    raise RuntimeError(
        f"Mismatched outputs: init has nt={nt}, latest file '{os.path.basename(vtk_last)}' has nt={nt_final}. "
        "Clean old VTK files or rerun in a fresh output directory."
    )

# analytic
n_poly, rho_c, K, G = 1.5, 1.0, 1.0, 1.0
xi_le, theta_le, xi1 = solve_lane_emden(n_poly)
alpha = np.sqrt((n_poly + 1) * K * rho_c**(1.0/n_poly - 1) / (4 * np.pi * G))
R_star = alpha * xi1
r_analytic = xi_le * alpha
rho_analytic = rho_c * np.where(theta_le > 0, theta_le**n_poly, 0.0)

j_eq = nt // 2
rho_init_1d = data_init['density'][:, j_eq]
rho_final_1d = data_final['density'][:, j_eq]
r_1d = r_init[:, j_eq]

density_final = data_final['density']
pole_mean = 0.5 * (density_final[:, 0] + density_final[:, -1])
equator = density_final[:, j_eq]
pole_to_eq = pole_mean / np.maximum(equator, 1e-30)
theta_cv = np.std(density_final, axis=1) / np.maximum(np.mean(density_final, axis=1), 1e-30)

flat_max = np.argmax(density_final)
i_max, j_max = np.unravel_index(flat_max, density_final.shape)
rho_max = density_final[i_max, j_max]
r_spike = r_final[i_max, j_max]
pole_ratio_max = np.max(pole_to_eq)
pole_ratio_r = r_1d[np.argmax(pole_to_eq)]
theta_cv_max = np.max(theta_cv)
theta_cv_r = r_1d[np.argmax(theta_cv)]

# ── Figure ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Lane\u2013Emden Hydrostatic Equilibrium Verification (n = 1.5)', fontsize=14)

# (a) density profile
ax = axes[0, 0]
ax.plot(r_analytic, rho_analytic, 'k-', lw=2, label='Analytic (Lane\u2013Emden)')
ax.plot(r_1d, rho_init_1d, 'b--', lw=1.5, label='t = 0 (initial)')
ax.plot(r_1d, rho_final_1d, 'r.', ms=5, label='t = final')
ax.set_xlabel('r'); ax.set_ylabel(r'$\rho$')
ax.set_title('(a) Radial density profile')
ax.legend(); ax.set_xlim(0, R_star * 1.1)

# (b) relative density error
ax = axes[0, 1]
rho_ref = np.interp(r_1d, r_analytic, rho_analytic)
mask = rho_ref > 1e-6
delta_init = np.abs(rho_init_1d[mask] - rho_ref[mask]) / rho_ref[mask]
delta_final = np.abs(rho_final_1d[mask] - rho_ref[mask]) / rho_ref[mask]
ax.semilogy(r_1d[mask], delta_init, 'b--', lw=1.5, label='t = 0')
ax.semilogy(r_1d[mask], delta_final, 'r-', lw=1.5, label='t = final')
ax.set_xlabel('r'); ax.set_ylabel(r'$|\delta\rho / \rho|$')
ax.set_title('(b) Relative density error vs. analytic')
ax.legend(); ax.set_xlim(0, R_star * 1.1)

# (c) 2D density map — use node coordinates directly for pcolormesh
ax = axes[1, 0]
x_n = coords_final[:, :, 0]
z_n = coords_final[:, :, 2]
positive = density_final[density_final > 0]
vmin = max(np.min(positive), np.percentile(positive, 1.0))
vmax = np.percentile(positive, 99.5)
if vmax <= vmin:
    vmax = np.max(positive)
pcm = ax.pcolormesh(
    x_n, z_n, density_final, shading='flat', cmap='inferno',
    norm=LogNorm(vmin=vmin, vmax=vmax)
)
fig.colorbar(pcm, ax=ax, label=r'$\rho$ (log scale, clipped)')
spike_theta = 0.5 * (np.arctan2(x_n[i_max, j_max], z_n[i_max, j_max]) + np.arctan2(x_n[i_max + 1, j_max + 1], z_n[i_max + 1, j_max + 1]))
ax.plot(
    r_spike * np.sin(spike_theta),
    r_spike * np.cos(spike_theta),
    marker='o', ms=5, mec='cyan', mfc='none', mew=1.2
)
ax.set_xlabel('x (= r sin \u03b8)'); ax.set_ylabel('z (= r cos \u03b8)')
ax.set_title(f'(c) 2D density map (final, true max = {rho_max:.2e})')
ax.set_aspect('equal')

# (d) polar spike metrics
ax = axes[1, 1]
log_path = os.path.join(build_dir, 'run.log')
times, masses = [], []
if os.path.exists(log_path):
    with open(log_path) as f:
        for line in f:
            m = re.search(r't\s*=\s*([\d.e+-]+).*M\s*=\s*([\d.e+-]+)', line)
            if m:
                times.append(float(m.group(1)))
                masses.append(float(m.group(2)))
ax.semilogy(r_1d, pole_to_eq, color='tab:purple', lw=1.6, label='pole/equator')
ax.semilogy(r_1d, theta_cv, color='tab:green', lw=1.6, ls='--', label=r'$\sigma_\theta/\bar{\rho}$')
ax.set_xlabel('r')
ax.set_ylabel('asymmetry metric')
ax.set_title('(d) Polar spike metrics (final)')
ax.axhline(1.0, color='k', ls=':', lw=0.5)
ax.legend(fontsize=9)
if len(masses) >= 2:
    M0 = masses[0]
    dM = np.array([(m - M0) / M0 for m in masses])
    ax.text(
        0.03, 0.03,
        f'max |ΔM/M0| = {np.max(np.abs(dM)):.2e}',
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none')
    )

# ── Provenance ───────────────────────────────────────────────────────
tend = times[-1] if times else 0.05
run_cmd = f'./stellar2d --test lane_emden --nr {nr} --ntheta {nt} --tend {tend}'
add_provenance_footer(fig, __file__, run_cmd=run_cmd)

plt.tight_layout(rect=[0, 0.03, 1, 1])
fname = build_filename('lane_emden', 'verification', nr, nt, tend)
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, fname)
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
print(f'Output dir: {build_dir}')
print(f'Latest file: {os.path.basename(vtk_last)}')
print(f'Max density: {rho_max:.6e} at (i={i_max}, j={j_max}, r={r_spike:.6e})')
print(f'Max pole/equator ratio: {pole_ratio_max:.6e} at r={pole_ratio_r:.6e}')
print(f'Max theta std/mean: {theta_cv_max:.6e} at r={theta_cv_r:.6e}')
