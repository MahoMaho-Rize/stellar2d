"""
Plot a one-zone effective pp-chain burn history from CSV output.

Usage:
    python plot_onezone_pp.py --csv ../build-mac/onezone_pp_test.csv
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from provenance import add_provenance_footer, build_filename


def read_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items() if k})
    if not rows:
        raise RuntimeError(f'No data rows found in {path}')
    return rows


def main():
    parser = argparse.ArgumentParser(description='Plot one-zone pp-chain CSV output.')
    parser.add_argument('--csv', required=True, help='Path to onezone_pp CSV file')
    args = parser.parse_args()

    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    csv_path = os.path.abspath(args.csv)
    rows = read_csv(csv_path)

    t = np.array([row['t'] for row in rows])
    T = np.array([row['T'] for row in rows])
    P = np.array([row['P'] for row in rows])
    e_int = np.array([row['e_int'] for row in rows])
    X_h = np.array([row['X_h'] for row in rows])
    X_he = np.array([row['X_he'] for row in rows])
    eps = np.array([row['eps_nuc'] for row in rows])
    rate = np.array([row['rate'] for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('One-Zone Effective pp-Chain Evolution', fontsize=14)

    ax = axes[0, 0]
    ax.plot(t, T, color='tab:red', lw=1.8)
    ax.set_xlabel('t')
    ax.set_ylabel('T')
    ax.set_title('(a) Temperature')

    ax = axes[0, 1]
    ax.plot(t, X_h, color='tab:blue', lw=1.8, label='X_H')
    ax.plot(t, X_he, color='tab:orange', lw=1.8, label='X_He')
    ax.set_xlabel('t')
    ax.set_ylabel('mass fraction')
    ax.set_title('(b) Composition')
    ax.legend()

    ax = axes[1, 0]
    ax.plot(t, e_int, color='tab:green', lw=1.8, label='e_int')
    ax.plot(t, P, color='tab:purple', lw=1.4, ls='--', label='P')
    ax.set_xlabel('t')
    ax.set_ylabel('state variable')
    ax.set_title('(c) Thermodynamic response')
    ax.legend()

    ax = axes[1, 1]
    ax.semilogy(t, np.maximum(eps, 1e-30), color='tab:brown', lw=1.8, label='eps_nuc')
    ax.semilogy(t, np.maximum(rate, 1e-30), color='tab:gray', lw=1.4, ls='--', label='rate')
    ax.set_xlabel('t')
    ax.set_ylabel('burning metric')
    ax.set_title('(d) Burn rate / heating')
    ax.legend()

    run_cmd = f'./build-mac/onezone_pp --output-csv {os.path.relpath(csv_path, repo_root)}'
    add_provenance_footer(fig, __file__, run_cmd=run_cmd)
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    out_dir = os.path.join(repo_root, 'results')
    os.makedirs(out_dir, exist_ok=True)
    tend = t[-1] if len(t) else 0.0
    out_name = build_filename('onezone_pp', 'history', 1, 1, tend, ext='png')
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved: {out_path}')
    print(f'CSV: {csv_path}')
    print(f'Final T = {T[-1]:.6e}, X_H = {X_h[-1]:.6e}, X_He = {X_he[-1]:.6e}, eps = {eps[-1]:.6e}')


if __name__ == '__main__':
    main()
