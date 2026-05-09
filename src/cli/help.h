#pragma once

// Tier A CLI — user-facing help and version printers.
//
// `print_help()` dumps a grouped, indented reference covering every flag
// recognised by parse_cli().  Content is hard-coded as a static block in
// help.cpp; Tier C (see docs/design/cli_unification_plan_2026-05-09.md §3e)
// will drive this from a SolverSpec registry instead.
//
// `print_version()` emits the compile-time git hash + build date.  The two
// macros STELLAR2D_GIT_HASH and STELLAR2D_BUILD_DATE are supplied by
// CMakeLists.txt via target_compile_definitions.  If absent (ad-hoc
// compilation outside cmake), "unknown" is printed instead.

void print_help();
void print_version();
