# Figure and Data Provenance Convention

## Objective

Every figure and data file produced by this project must be traceable to the
exact code revision, simulation parameters, and generating script that produced
it.  A reader should be able to reproduce any figure from its filename alone.

---

## Filename Format

```
{test}_{quantity}_{nr}x{nt}_t{tend}_{commit7}_{date}.{ext}
```

| Field      | Description                                  | Example            |
|------------|----------------------------------------------|--------------------|
| `test`     | Test case name                               | `lane_emden`       |
| `quantity` | What the figure/data shows                   | `verification`     |
| `nr`x`nt`  | Grid resolution                              | `64x32`            |
| `tend`     | Simulation end time, 3 decimal places        | `t0.050`           |
| `commit7`  | First 7 characters of the git commit hash    | `9d6bdbe`          |
| `date`     | Generation date (YYYYMMDD)                   | `20260427`         |
| `ext`      | File extension                               | `png`, `vtk`, `h5` |

### Examples

```
lane_emden_verification_64x32_t0.050_9d6bdbe_20260427.png
sedov_density_2d_256x128_t0.100_a3f01cc_20260502.png
evrard_radial_profile_128x64_t1.000_b7e229a_20260510.png
```

---

## Figure Footer

Every figure includes a provenance footer (bottom-left, grey monospace, 6 pt)
containing:

```
commit: 9d6bdbe (9d6bdbee31a04f2c8bda…)
script: tests/plot_lane_emden.py
date:   2026-04-27 18:30:42
cmd:    ./stellar2d --test lane_emden --nr 64 --ntheta 32 --tend 0.05
```

This is produced by `tests/provenance.py:add_provenance_footer()`.

---

## Simulation Log

Each simulation run should capture stdout to a log file alongside the VTK
output.  The log filename follows the same convention:

```
lane_emden_run_64x32_t0.050_9d6bdbe_20260427.log
```

---

## Data Directory Layout

```
results/
└── lane_emden_64x32_t0.050_9d6bdbe_20260427/
    ├── output_0000.vtk
    ├── output_0001.vtk
    ├── ...
    ├── output_final.vtk
    └── run.log
```

Each run lives in its own directory.  The directory name follows the filename
convention (minus `quantity` and extension).

---

## Implementation

The Python module `tests/provenance.py` provides:

| Function                  | Purpose                                         |
|---------------------------|-------------------------------------------------|
| `build_filename()`        | Construct a traceable filename from parameters   |
| `provenance_string()`     | Build a multi-line provenance block              |
| `add_provenance_footer()` | Stamp a matplotlib figure with provenance        |
| `git_commit_short()`      | Return 7-char commit hash                        |
| `git_dirty()`             | Check for uncommitted changes                    |

### Usage in plotting scripts

```python
from provenance import build_filename, add_provenance_footer

# ... generate figure ...

run_cmd = './stellar2d --test lane_emden --nr 64 --ntheta 32 --tend 0.05'
add_provenance_footer(fig, __file__, run_cmd=run_cmd)

fname = build_filename('lane_emden', 'verification', 64, 32, 0.05)
fig.savefig(fname, dpi=150, bbox_inches='tight')
```

---

## Rules

1. **Never rename or move** a generated figure.  Its filename is its identity.
2. If the code changes, re-run and generate a **new** figure with the new
   commit hash.  Old figures remain valid references to old code.
3. Figures committed to the repository go in `results/`.  The `.gitignore`
   excludes `build/output_*.vtk` but not `results/`.
4. If a figure is produced from a dirty working tree, the footer says
   `(dirty)` — such figures should not be used in publications.
