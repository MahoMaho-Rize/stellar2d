#pragma once

// Tier B-1 CLI — TOML config loader.
//
// `load_toml_into_cfg(path, cfg)` reads a flat TOML file produced either
// by the Tier B-3 `run_dir/config.toml` dump or hand-written, and applies
// every recognised key to the given SimConfig.  Unknown top-level keys
// are a HARD ERROR (with Levenshtein suggestion, reusing src/cli/suggest).
//
// Keys not present in the file are left at their SimConfig defaults, so
// partial TOML files are fine — they only override what they mention.
//
// Priority ordering (see docs/design/cli_unification_plan_2026-05-09.md §3f):
//     CLI flags > --profile NAME > --config FILE > built-in defaults
//
// This means main() calls load_toml_into_cfg BEFORE parse_cli, letting
// subsequent CLI flags override any TOML-loaded values.  Tier B-2 will add
// the --profile layer.
//
// Returns 0 on success, 1 on any error (parse error, unknown key,
// type mismatch).  In the error path, a human-readable diagnostic is
// written to stderr.

#include <string>

struct SimConfig;

int load_toml_into_cfg(const std::string& path, SimConfig& cfg);
