#pragma once

// Tier B-3 CLI — resolved-configuration dump (TOML form).
//
// After parse_cli() + setup_simulation() finish, call dump_resolved_cli()
// to write ``<run_dir>/config.toml``, a flat TOML transcript of every
// SimConfig field active for this run.
//
// The output is directly reloadable via `stellar2d run --config
// <run_dir>/config.toml` — that's the reproducibility contract:
// identical config in, identical run out.
//
// Tier A shipped a key=value text dump (config.dump.txt); Tier B-3 promotes
// it to proper TOML (string values quoted, empty strings as "", etc.) so
// that config_loader.cpp can round-trip it.  The schema is the same flat
// namespace; Tier C will switch to grouped tables ([grid], [solver.<name>]).
//
// Failure to open the output file emits a warning to stderr and is
// non-fatal: the simulation still runs, we just lose the dump.

#include <string>

struct SimConfig;

void dump_resolved_cli(const SimConfig& cfg, const std::string& run_dir);
