#!/usr/bin/env python3
"""Paper-ready figure: scheme × resolution scan, both codes, 5 observables.

Panels:
    (a) v_rms(resolution) for each scheme, colored by code
    (b) Ṁ_e/Lx(resolution)
    (c) δ_int(resolution) — log scale
    (d) L/Lx(resolution)
    (e) y_ub(resolution)
    (f) Ṁ_e × δ_int / L_iso vs resolution — "D_eff" proxy

3D reference overplotted as horizontal band where applicable.
"""
from __future__ import annotations

import glob
import numpy as np
import matplotlib.pyplot as plt


def load(pat):
    fs = sorted(glob.glob(pat))
    return np.loadtxt(fs[-1], delimiter=',', skiprows=1) if fs else None


def avg_col(d, c, lo, hi):
    if d is None: return None
    t = d[:, 0]; v = d[:, c]
    m = (t >= lo) & (t <= hi) & ~np.isnan(v)
    return float(np.nanmean(v[m])) if m.sum() >= 3 else None


def fit_md(d, lo, hi):
    if d is None: return None
    t = d[:, 0]; Me = d[:, 2]
    ok = ~np.isnan(Me); t = t[ok]; Me = Me[ok]
    m = (t >= lo) & (t <= hi)
    return float(np.polyfit(t[m], Me[m], 1)[0]) if m.sum() >= 3 else None


Lx = 2.0

# (code, res, scheme, label, color, marker, run_glob, t_lo, t_hi)
RUNS = [
    # Eulerian Godunov (vl2)
    ('vl2', 128, 'VL',  'Athena PLM-VL', 'C0', 'o', 'runs/athena_vl2_scan_v2/128_vanleer/andrassy2022_128x128_*', 1000, 2000),
    ('vl2', 256, 'VL',  'Athena PLM-VL', 'C0', 'o', 'runs/athena_vl2_scan_v2/256_vanleer/andrassy2022_256x256_*', 1000, 2000),
    ('vl2', 512, 'VL',  'Athena PLM-VL', 'C0', 'o', 'runs/athena_vl2_scan_v2/512_vanleer/andrassy2022_512x512_*', 1000, 2000),
    ('vl2', 128, 'MM',  'Athena PLM-MM', 'C0', 's', 'runs/athena_vl2_scan_v2/128_minmod/andrassy2022_128x128_*', 1000, 2000),
    ('vl2', 256, 'MM',  'Athena PLM-MM', 'C0', 's', 'runs/athena_vl2_scan_v2/256_minmod/andrassy2022_256x256_*', 1000, 2000),
    ('vl2', 512, 'MM',  'Athena PLM-MM', 'C0', 's', 'runs/athena_vl2_scan_v2/512_minmod/andrassy2022_512x512_*', 1000, 2000),
    # Lagrangian ALE
    ('ale2', 256, 'MC',  'ALE MUSCL-MC',     'C3', '^', 'runs/scheme_scan_256/muscl_mc',       1000, 2000),
    ('ale2', 256, 'VL',  'ALE MUSCL-VL',     'C3', 'o', 'runs/scheme_scan_256/muscl_vanleer',  1000, 2000),
    ('ale2', 256, 'MM',  'ALE MUSCL-MM',     'C3', 's', 'runs/scheme_scan_256/muscl_minmod',   1000, 2000),
    ('ale2', 256, 'PPM', 'ALE PPM-CS',       'C3', 'D', 'runs/scheme_scan_256/ppm_cs_prim',    1000, 2000),
    ('ale2', 512, 'VL',  'ALE MUSCL-VL',     'C3', 'o', 'runs/scheme_scan_512/muscl_vanleer/andrassy2022_512x512_*', 1000, 2000),
    ('ale2', 512, 'MM',  'ALE MUSCL-MM',     'C3', 's', 'runs/scheme_scan_512/muscl_minmod/andrassy2022_512x512_*', 1000, 2000),
    ('ale2', 512, 'PPM', 'ALE PPM-CS',       'C3', 'D', 'runs/scheme_scan_512/ppm_cs_char/andrassy2022_512x512_*',  1000, 2000),
    # 128 ale2 — interface collapses, use early-time window
    ('ale2', 128, 'VL',  'ALE MUSCL-VL',     'C3', 'o', 'runs/scheme_scan_128/muscl_vanleer/andrassy2022_128x128_*', 100, 400),
    ('ale2', 128, 'MM',  'ALE MUSCL-MM',     'C3', 's', 'runs/scheme_scan_128/muscl_minmod/andrassy2022_128x128_*', 100, 400),
    ('ale2', 128, 'PPM', 'ALE PPM-CS',       'C3', 'D', 'runs/scheme_scan_128/ppm_cs_char/andrassy2022_128x128_*',  100, 400),
]

rows = []
for (code, res, sch, lab, col, mkr, glob_pat, lo, hi) in RUNS:
    ts = load(f'{glob_pat}/analysis/timeseries.csv') if 'analysis' not in glob_pat else None
    if ts is None:
        ts = load(f'{glob_pat}/timeseries.csv')
    ent = load(f'{glob_pat}/entrainment.csv')
    intf = load(f'{glob_pat}/interface.csv')
    vr = avg_col(ts, 3, lo, hi)
    yu = avg_col(ts, 2, lo, hi)
    md = fit_md(ent, lo, hi)
    di = avg_col(intf, 2, lo, hi)
    Ll = avg_col(intf, 4, lo, hi)
    rows.append(dict(code=code, res=res, sch=sch, label=lab, color=col, marker=mkr,
                     v_rms=vr, y_ub=yu, Mdot=md, delta=di, Ll=Ll))

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
(ax_v, ax_md, ax_d), (ax_L, ax_y, ax_deff) = axes

def plot_obs(ax, key, ylabel, log=False, ref=None, ref_label='Andrassy 3D'):
    for r in rows:
        if r[key] is None: continue
        y = r[key]
        # Normalize Ṁ by Lx
        if key == 'Mdot':
            y = y / Lx
        ax.plot(r['res'], y, marker=r['marker'], color=r['color'],
                ls='', markersize=8, label=None)
    if ref is not None:
        ax.axhline(ref, color='gray', ls='--', lw=1.2, label=ref_label)
    ax.set_xlabel('resolution N')
    ax.set_ylabel(ylabel)
    ax.set_xscale('log', base=2)
    ax.set_xticks([128, 256, 512])
    ax.set_xticklabels(['128', '256', '512'])
    if log:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

plot_obs(ax_v,   'v_rms', r'$v_{\rm rms,conv}$', ref=0.088)
plot_obs(ax_md,  'Mdot',  r'$\dot M_e / L_x$', log=True, ref=2.37e-5)
plot_obs(ax_d,   'delta', r'$\delta_{\rm int}$', log=True)
plot_obs(ax_L,   'Ll',    r'$L_{\rm iso} / L_x$')
plot_obs(ax_y,   'y_ub',  r'$y_{\rm ub}$', ref=2.25)

# Panel (f): effective diffusivity proxy D_eff ~ Ṁ δ / (ρ L_iso)
# Taking ρ≈1 (slab avg).
for r in rows:
    if r['Mdot'] is None or r['delta'] is None or r['Ll'] is None: continue
    D = r['Mdot'] * r['delta'] / (1.0 * r['Ll'] * Lx)
    if D <= 0: continue
    ax_deff.plot(r['res'], D, marker=r['marker'], color=r['color'], ls='', markersize=8)
ax_deff.set_xlabel('resolution N')
ax_deff.set_ylabel(r'$D_{\rm eff} \sim \dot M_e \delta / (\rho A)$')
ax_deff.set_xscale('log', base=2)
ax_deff.set_xticks([128, 256, 512])
ax_deff.set_xticklabels(['128', '256', '512'])
ax_deff.set_yscale('log')
ax_deff.grid(True, alpha=0.3)

# Legend — unique label per family × scheme
seen = set()
handles = []
for r in rows:
    key = (r['color'], r['marker'])
    if key in seen: continue
    seen.add(key)
    h = plt.Line2D([], [], marker=r['marker'], color=r['color'], ls='', markersize=8, label=r['label'])
    handles.append(h)
handles.append(plt.Line2D([], [], color='gray', ls='--', label='Andrassy 3D ref'))
fig.legend(handles=handles, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.98), fontsize=9)

fig.suptitle('Scheme × Resolution Scan — 2D Andrassy 2022 Benchmark',
             y=1.00, fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = 'paper/figures/scheme_resolution_scan.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Wrote {out}')