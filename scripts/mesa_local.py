"""Helpers for finding a local MESA installation on this machine."""
from __future__ import annotations

import os
from pathlib import Path


def _existing_dirs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        try:
            rp = p.expanduser().resolve()
        except FileNotFoundError:
            rp = p.expanduser()
        if rp in seen:
            continue
        seen.add(rp)
        if rp.is_dir():
            out.append(rp)
    return out


def mesa_root_candidates() -> list[Path]:
    """Return plausible local MESA roots, best guesses first."""
    home = Path.home()
    env_root = os.environ.get("MESA_DIR", "").strip()
    cands: list[Path] = []
    if env_root:
        cands.append(Path(env_root))

    cands.extend([
        home / "MESA" / "mesa",
        home / "MESA",
        home / "mesa",
        home / "mesa-ref",
    ])
    cands.extend(sorted((home / "MESA").glob("mesa-*"), reverse=True))
    cands.extend(sorted(home.glob("mesa-*"), reverse=True))
    return _existing_dirs(cands)


def detect_mesa_dir() -> Path | None:
    """Return the first candidate that looks like a MESA tree."""
    for root in mesa_root_candidates():
        if (root / "data").is_dir() and (root / "star").is_dir():
            return root
    return None


def detect_kap_data_dir() -> Path | None:
    """Return `<mesa>/data/kap_data` if a local MESA install is found."""
    mesa = detect_mesa_dir()
    if mesa is None:
        return None
    kap = mesa / "data" / "kap_data"
    return kap if kap.is_dir() else None
