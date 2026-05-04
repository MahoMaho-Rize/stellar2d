# MESA OPAL / Ferguson Rosseland-mean opacity tables (KAPv1 binary cache)

Produced by `scripts/convert_mesa_kap.py` from the ASCII Type-1 files that
ship with MESA (`$MESA_DIR/data/kap_data/`). One `.kapbin` per `(family, Z)`
group; each file stacks all X-slices into a single `(n_X, n_logT, n_logR)`
float64 tensor of log₁₀ κ_R [cm²/g], with a 128-byte self-describing header
(magic `KAPv1`) — see the module docstring in `scripts/convert_mesa_kap.py`.

## Regenerating

```bash
# default: /home/kiriko/mesa-ref/data/kap_data -> third_party/mesa_kap/
python3 scripts/convert_mesa_kap.py

# one family only
python3 scripts/convert_mesa_kap.py \
    /home/kiriko/mesa-ref/data/kap_data third_party/mesa_kap gs98

# verify round-trip against ASCII
python3 scripts/verify_mesa_kapbin.py
```

Full run produces 467 files, ~410 MB, log κ bit-exact round-trip. `.kapbin`
binaries are git-ignored.

## Coverage

| family                         | n_Z | notes |
|---|---|---|
| gs98, a09, gn93                | 13  | standard solar OPAL Type-1 |
| gs98_aFe_m2…p8                 | 13 each | α-enhanced variants |
| OP_gs98, OP_a09…, OP_Fe         | 1–13 | OP project opacities |
| oplib_gs98/agss09/aag21/mb22   | 41 each | OPLIB 2024 |
| lowT_af94_gn93, lowT_fa05_*    | 13 each | Alexander-Ferguson + Ferguson-Alexander low-T |

`_co` (C/O-enhanced, form=2) tables and `kR_*` conductivity tables use
different layouts and are not handled here.

## How to pick the right family at runtime

- Matches your MESA inlist `kap_file_prefix`: point radial1d at the same
  family so both runs share opacity and the PK reflects EOS + solver only.
- For generic solar MS / pre-MS work: `gs98_z0.02.kapbin` (high-T) plus
  `lowT_fa05_gs98_z0.02.kapbin` (photosphere). Stitch at the overlap region
  logT ≈ 4.0 by taking the min of the two (MESA's own default scheme).
