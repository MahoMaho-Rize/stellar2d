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

The ASCII file is ~60 MB and slow to parse (Fortran free-format nested
loops). We preprocess once with a standalone C++ tool into a clean
little-endian binary with a 64-byte magic+metadata header:

```
g++ -O2 tools/helm_convert.cpp -o tools/helm_convert
tools/helm_convert \
    third_party/helmholtz/helm_table.dat \
    third_party/helmholtz/helm_table.bin
```

Resulting `helm_table.bin` is ~17 MB (twice-dense resolution,
imax=541, jmax=201, 21 tabulated fields × float64). The CUDA loader
in `src/physics/helmholtz_eos.cu` **only reads the binary** — no ASCII
parsing in the build. The ASCII path argument to `HelmholtzTable::load`
is unused (retained for API compatibility).

## GPU placement strategy

On Ada/Hopper (sm_89+) the 5 MB binary is small enough to pin into
L2 persisting cache via `cudaAccessPolicyWindow` with
`cudaAccessPropertyPersisting`. This gives us ~50× bandwidth vs. DRAM
for every EOS table lookup inside JFNK GMRES iterations.

RTX 4090:       L2 = 72 MB,  persisting cap ~48 MB
RTX 4080 Super: L2 = 64 MB,  persisting cap 44 MB   ← our dev box
H100:           L2 = 60 MB,  persisting cap ~40 MB
Our table:      17 MB — fits comfortably.

Verified working on 4080 Super (sm_89):
```
helmholtz: pinned 17.42 MB to L2 persisting cache (cap 17.4 MB, device max 44.0 MB)
```
