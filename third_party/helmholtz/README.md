# Helmholtz EOS table drop-in

Timmes & Swesty (2000) Helmholtz free-energy EOS — standard community tool,
public domain. Covers:
  * T ∈ [10⁴, 10¹¹] K
  * ρ ∈ [10⁻¹², 10¹¹] g/cc
  * partial electron degeneracy → relativistic limit
  * pure ideal gas + radiation + electrons/positrons

## Why we use it

Full analytic EOS (ideal + radiation) breaks down for:
  * dense cores (electron degeneracy pressure)
  * white dwarfs / neutron-star precursors
  * late-stage pre-MS where degenerate core starts forming

Helmholtz tabulates only the **electron free energy** (the messy part) and
keeps radiation + ideal ion pressure as analytic terms. Table is ~5 MB.

## Getting the table

Download `helm_table.dat` from Frank Timmes's cococubed site:
  http://cococubed.com/code_pages/eos.shtml
  (direct: http://cococubed.com/codes/eos/helm_table.dat)

File is ASCII, ~8 MB. Place into this directory:
  third_party/helmholtz/helm_table.dat

License: cococubed pages explicitly state code + tables are free to use
("you may use, modify, and redistribute ..."). We treat as public-domain.

## Binary conversion

Because the ASCII file is slow to parse (nested loops of Fortran
formatted reads), we convert once to little-endian binary on first run:
  third_party/helmholtz/helm_table.bin   (~5 MB, float64)

The conversion utility lives in `src/physics/helmholtz_eos.cu` under
`helm_convert_ascii_to_binary()`.

## GPU placement strategy

On Ada/Hopper (sm_89+) the 5 MB binary is small enough to pin into
L2 persisting cache via `cudaAccessPolicyWindow` with
`cudaAccessPropertyPersisting`. This gives us ~50× bandwidth vs. DRAM
for every EOS table lookup inside JFNK GMRES iterations.

RTX 4090: L2 = 72 MB
H100:     L2 = 60 MB
Our table: 5 MB — fits comfortably alongside other persisting state.
