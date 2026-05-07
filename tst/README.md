# stellar2d end-to-end pytest suite

Framework for GPU end-to-end validation, launched 2026-05-07 per
`docs/design/testing_infrastructure_plan_2026-05-07.md` Phase 2.

## Running

```bash
# Build first
cd build && make -j stellar2d

# Via pytest directly
cd tst
STELLAR2D_BIN=../build/stellar2d pytest -m fast -v

# Via ctest (from build/)
ctest -L fast -R pytest_gpu
```

`pytest_gpu_fast` is registered in `CMakeLists.txt`; it sets
`STELLAR2D_BIN=$CMAKE_BINARY_DIR/stellar2d` and runs `pytest -m fast` in
`tst/`.

## Layout

| file | role |
|---|---|
| `conftest.py` | register fast/slow/scan markers + binary-missing guard |
| `testutils.py` | `run_stellar2d()`, `find_error_dat()`, `read_error_dat()` |
| `test_ale2/` | cart_ale2 e2e tests |
| `test_vl2/` | athena_vl2 e2e tests |
| `bin/` | ephemeral per-test run directories (gitignored) |

## Markers

| marker | target time | triggered by |
|---|---|---|
| `fast` | < 30 s each | default `ctest -L fast` / CI |
| `slow` | 1-5 min | opt-in `pytest -m slow` |
| `scan` | > 30 min | always manual |

## compute_error pattern

The core pattern is **"C++ computes error, Python asserts"** (Athena++
§compute_error). Each test function:

1. Calls `stellar2d ... --compute-error` inside a fresh `tst/bin/<name>/`.
2. The solver runs to `--tend` then calls
   `CartAle2Solver::compute_<test>_error()` (or `AthenaVL2Solver::...`).
3. That method scores L1 / Linf vs. analytic and appends a line to
   `<run_dir>/<test>-errors.dat` with a schema header.
4. `pytest` reads the `.dat` via `read_error_dat()` and asserts thresholds.

No analytic solvers in Python. All analytic math (Toro Riemann, Gresho
piecewise `vφ(r)`, Yee isentropic vortex, entropy wave phase fit) lives in
the solver .cu files or `src/gpu/common/*.h` (e.g. `sod_exact.h`).

## Current coverage (2026-05-07)

| benchmark | cart_ale2 | athena_vl2 |
|---|---|---|
| entropy_wave | ✅ 2 tests (N=64, N=128) | ✅ 2 tests (N=64, N=128) |
| sod          | ✅ 2 tests (N=128, convergence) | ✅ 2 tests |
| gresho       | ✅ 2 tests (N=128, convergence) | — |
| yee          | ✅ 2 tests (N=128 short-t, smoke) | — |
| linwave      | ❌ (Phase 2 residual, task #57) | ❌ |
| sedov        | — (no C++ analytic) — python `scripts/tests_ale2/sedov_compare.py` | — |
| noh          | — (inflow BC not supported) — python `scripts/tests_ale2/noh_compare.py` | — |

## Adding a new compute_error test

1. Implement `XSolver::compute_mytest_error(t, ncycle, ..., run_dir)` in
   the solver `.cu`. Schema line on first write:
   `# schema: Nx Ny Ncycle t_end <your cols>`.
2. Add CLI flag if needed (usually `--compute-error` is sufficient).
3. Invoke from the driver when `cfg.compute_error && is_mytest`.
4. Write `tst/test_X/test_X_mytest_gpu.py` — call `run_stellar2d(...)`,
   then `rows = read_error_dat(find_error_dat(run_base, "mytest"))`.
5. Use `@pytest.mark.fast` unless the run is > 30 s.
