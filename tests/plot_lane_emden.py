"""
Lane-Emden hydrostatic equilibrium verification plot.
Reads VTK output from stellar2d and compares against the analytic solution.
Naming convention: see docs/provenance.md
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build')

vtk_init = os.path.join(build_dir, 'output_0000.vtk')
vtk_files = sorted(glob.glob(os.path.join(build_dir, 'output_*.vtk')))
vtk_final = [f for f in vtk_files if 'final' in f]
vtk_last = vtk_final[0] if vtk_final else (vtk_files[-1] if vtk_files else vtk_init)

nr, nt, r_init, data_init, coords_init = read_vtk(vtk_init)
_, _, r_final, data_final, coords_final = read_vtk(vtk_last)

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
pcm = ax.pcolormesh(x_n, z_n, data_final['density'], shading='flat', cmap='inferno')
fig.colorbar(pcm, ax=ax, label=r'$\rho$')
ax.set_xlabel('x (= r sin \u03b8)'); ax.set_ylabel('z (= r cos \u03b8)')
ax.set_title('(c) 2D density map (final)')
ax.set_aspect('equal')

# (d) mass conservation from log
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
if len(masses) >= 2:
    M0 = masses[0]
    dM = np.array([(m - M0) / M0 for m in masses])
    ax.plot(times, dM, 'b-', lw=1.5)
    ax.set_xlabel('t'); ax.set_ylabel(r'$\Delta M / M_0$')
    ax.set_title(f'(d) Relative mass drift (M\u2080 = {M0:.6e})')
    ax.axhline(0, color='k', ls=':', lw=0.5)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-6, -6))
else:
    ax.text(0.5, 0.5, 'Re-run with: ./stellar2d ... 2>&1 | tee run.log',
            ha='center', va='center', transform=ax.transAxes, fontsize=10)
    ax.set_title('(d) Mass conservation (no log data)')

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
