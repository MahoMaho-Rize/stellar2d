#!/usr/bin/env bash
# Download all benchmark datasets for the stellar2d spectral pipeline.
# Idempotent: skips files that already exist.  Total ~35 MB.
# Usage: bash data/benchmarks/download_all.sh
set -euo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE"

mkdir -p bowman2019 dedalus_examples fgong_archive gyre_examples \
         jones2011 kepler_legacy mesa_models posc

echo "==> 1. FGONG solar models (Christensen-Dalsgaard, users-phys.au.dk)"
cd "$BASE/fgong_archive"
for f in fgong.l5bi.d.15 fgong.l5bi.d.15c cptrho.l5bi.d.15c file-format.pdf; do
    [[ -f "$f" ]] || wget -q "https://users-phys.au.dk/jcd/solar_models/$f"
done

echo "==> 2. Kepler LEGACY (Lund+ 2017, VizieR J/ApJ/835/172)"
cd "$BASE/kepler_legacy"
for f in ReadMe table1.dat table6.dat table7.dat table8.dat; do
    [[ -f "$f" ]] || wget -q "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/835/172/$f"
done

echo "==> 3. Bowman+ 2019 Nat Ast SI (22 MB)"
cd "$BASE/bowman2019"
if [[ ! -f Bowman2019_SI.pdf ]]; then
    curl -sL -o Bowman2019_SI.pdf \
        "https://static-content.springer.com/esm/art%3A10.1038%2Fs41550-019-0768-1/MediaObjects/41550_2019_768_MOESM1_ESM.pdf"
fi

echo "==> 4. Rayleigh input decks (Jones 2011 / Christensen 2001 / Boussinesq 2010)"
cd "$BASE/jones2011"
if [[ ! -d rayleigh_src ]]; then
    git clone --depth=1 --filter=blob:none --sparse \
        https://github.com/geodynamics/Rayleigh.git rayleigh_src
    (cd rayleigh_src && git sparse-checkout set input_examples)
    rm -rf rayleigh_src/.git
fi

echo "==> 5. Dedalus examples (Rayleigh-Benard / Lane-Emden ball / rotating convection)"
cd "$BASE/dedalus_examples"
if [[ ! -d dedalus_src ]]; then
    git clone --depth=1 --filter=blob:none --sparse \
        https://github.com/DedalusProject/dedalus.git dedalus_src
    (cd dedalus_src && git sparse-checkout set examples)
    rm -rf dedalus_src/.git
fi

echo "==> 6. GYRE bundled models (local copy)"
cd "$BASE/gyre_examples"
if [[ -d "$HOME/gyre/models" ]] && [[ -z "$(ls -A . 2>/dev/null)" ]]; then
    cp -r "$HOME/gyre/models/"* .
else
    echo "  (skipped: either ~/gyre/models missing or target already populated)"
fi

echo "==> 7. MESA test_suite (symlink)"
cd "$BASE/mesa_models"
if [[ -d "$HOME/mesa-ref/star/test_suite" ]] && [[ ! -e test_suite ]]; then
    ln -s "$HOME/mesa-ref/star/test_suite" test_suite
else
    echo "  (skipped: either MESA not at ~/mesa-ref or symlink already exists)"
fi

echo "==> 8. POSC / ESTA task2 (indexing suppressed — only roadmap + landing)"
cd "$BASE/posc"
[[ -f ESTA_Task1_Roadmap.pdf ]] || \
    wget -q "https://www.astro.up.pt/corot/compmod/task1/ESTA_Task1_Roadmap.pdf"
[[ -f posc_task2_index.html ]] || \
    wget -q -O posc_task2_index.html "https://www.astro.up.pt/corot/compfreqs/task2/"

echo ""
echo "All done. Sizes:"
du -sh "$BASE"/*/ | sort -hr
