#pragma once

// Tier A CLI — resolved-configuration dump.
//
// After parse_cli() + setup_simulation() finish, call dump_resolved_cli()
// to write ``<run_dir>/config.dump.txt``, a plain key=value transcript of
// every SimConfig field active for this run.
//
// Format is intentionally not TOML in Tier A — we want a zero-dependency
// reproducibility paper-trail first, and will upgrade to a TOML form that
// round-trips with ``stellar2d --config`` in Tier B (see
// docs/design/cli_unification_plan_2026-05-09.md §2d / §6).
//
// Failure to open the output file emits a warning to stderr and is
// non-fatal: the simulation still runs, we just lose the dump.

#include <string>

struct SimConfig;

void dump_resolved_cli(const SimConfig& cfg, const std::string& run_dir);
