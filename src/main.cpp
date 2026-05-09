// stellar2d — 2D Axisymmetric Euler + Self-Gravity (+ MHD, low-Mach, spectral)
//
// Top-level entry point.  Tier B-1 introduced the subcommand layer: users
// MUST now write `stellar2d run [FLAGS]` instead of the bare flag form.
// See docs/design/cli_unification_plan_2026-05-09.md §2b / §4.
//
// Responsibilities:
//   1. install SIGINT/SIGTERM handler so the time loop can bail cleanly
//   2. scan for --help / --version anywhere in argv (global)
//   3. require argv[1] == "run" (other subcommands planned for Tier B-3)
//   4. pre-scan --config FILE → load_toml_into_cfg so TOML values get
//      overridden by subsequent CLI flags (priority: CLI > TOML > defaults)
//   5. parse argv into SimConfig                       (cli/options.cpp)
//   6. build grid / state / EOS / tables / IC          (sim/setup.cpp)
//   7. dispatch to the selected solver driver          (drivers/<name>.cpp)
//   8. dump output_final.vtk + destroy tables          (below)
//
// Each solver owns its own time loop, diagnostics, VTK cadence, and exit
// code in src/drivers/<name>.cpp; add a new solver by writing one driver
// file + adding a `} else if (cfg.solver_type == "newname") { ... }`
// branch here + a prototype in drivers/drivers.h.

#include "grid.h"
#include "state.h"
#include "eos.h"
#include "io/output.h"

#include "cli/options.h"
#include "cli/config_dump.h"
#include "cli/config_loader.h"
#include "cli/help.h"
#include "sim/helpers.h"
#include "sim/setup.h"
#include "sim/run_loop.h"
#include "drivers/drivers.h"

#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// g_interrupted is defined in sim/run_loop.cpp (GPU) or drivers/cpu.cpp (CPU).
static void handle_sigint(int) { g_interrupted = 1; }

namespace {

int run_simulation(int argc, char** argv) {
    // argv[0] is the "run" subcommand token; parse_cli's for-loop starts at
    // index 1, so it effectively ignores it.

    SimConfig cfg;

    // Pre-scan --config FILE so TOML-loaded values are written into cfg
    // BEFORE parse_cli applies CLI overrides (priority order §3f).
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            if (int rc = load_toml_into_cfg(argv[i + 1], cfg); rc != 0)
                return rc;
            ++i;   // skip the path token we just consumed
        }
    }

    // Pre-scan --profile NAME after --config so profiles overlay on top
    // of raw configs (§3f: CLI > --profile > --config > defaults).
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--profile") == 0 && i + 1 < argc) {
            if (int rc = load_profile_into_cfg(argv[i + 1], cfg); rc != 0)
                return rc;
            ++i;
        }
    }

    if (int rc = parse_cli(argc, argv, cfg); rc != 0)
        return (rc == 2) ? 0 : rc;

    SimContext ctx;
    if (int rc = setup_simulation(cfg, ctx); rc != 0) return rc;

    // Reproducibility paper trail (Tier A; Tier B-3 will upgrade to TOML).
    dump_resolved_cli(cfg, ctx.run_dir);

    double t = 0.0;
    int step = 0;
    std::printf("Starting time integration...\n");

#ifdef USE_GPU
    if      (cfg.solver_type == "radial1d")        { if (int rc = run_radial1d       (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "projection")      { if (int rc = run_projection     (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "simple")          { if (int rc = run_simple         (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "fas2")            { if (int rc = run_fas2           (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "fas" ||
             cfg.solver_type == "explicit")        { if (int rc = run_fas            (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "cart_lag")        { if (int rc = run_cart_lag       (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "cart_ale")        { if (int rc = run_cart_ale       (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "cart_impl")       { if (int rc = run_cart_impl      (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "cart_ale2")       { if (int rc = run_cart_ale2      (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "athena_vl2")      { if (int rc = run_athena_vl2     (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "athena_mhd")      { if (int rc = run_athena_mhd     (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "pseudo_spectral") { if (int rc = run_pseudo_spectral(cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "anelastic_sl")    { if (int rc = run_anelastic_sl   (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "sph2d_spectral")  { if (int rc = run_sph2d_spectral (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "ale2d")           { if (int rc = run_ale2d          (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "wb2d")            { if (int rc = run_wb2d           (cfg, ctx, t, step); rc != 0) return rc; }
    else if (cfg.solver_type == "lowmach")         { if (int rc = run_lowmach        (cfg, ctx, t, step); rc != 0) return rc; }
    else {
#ifdef USE_AMGX
        if (int rc = run_compressible(cfg, ctx, t, step); rc != 0) return rc;
#else
        std::fprintf(stderr, "ERROR: --solver compressible requires AmgX. "
                     "Rebuild with -DAMGX_DIR=/path/to/amgx, or use --solver lowmach.\n");
        return 1;
#endif
    }
#else
    if (int rc = run_cpu(cfg, ctx, t, step); rc != 0) return rc;
#endif

    Diagnostics diag = compute_diagnostics(ctx.grid, ctx.state, cfg.gamma);
    std::printf("Final: step %d  t = %.6e  M = %.10e  E = %.10e\n",
                step, t, diag.total_mass, diag.total_energy);
    {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/output_final.vtk", ctx.run_dir.c_str());
        write_vtk(path, ctx.grid, ctx.state, cfg.gamma);
    }

#ifdef USE_GPU
    ctx.destroy_tables();
#endif

    std::printf("Done.\n");
    return 0;
}

void print_missing_subcommand_error() {
    std::fprintf(stderr,
        "ERROR: stellar2d requires a subcommand.\n"
        "  Usage:\n"
        "    stellar2d run [FLAGS]                Run a simulation\n"
        "    stellar2d run --config FILE          Load TOML preset, optionally override\n"
        "    stellar2d run --profile NAME         Run a named preset from config/profiles/\n"
        "    stellar2d list profiles              List available profiles\n"
        "    stellar2d list solvers               List available solvers\n"
        "    stellar2d describe profile NAME      Print a profile's TOML to stdout\n"
        "    stellar2d validate PATH              Parse a TOML without executing\n"
        "    stellar2d --help                     Full flag reference\n"
        "    stellar2d --version                  Print version\n"
        "\n"
        "  The bare `stellar2d --flag ...` form is no longer accepted\n"
        "  (CLI unification Tier B — see docs/design/cli_unification_plan_2026-05-09.md).\n");
}

void print_unknown_subcommand_error(const char* cmd) {
    std::fprintf(stderr, "ERROR: unknown subcommand \"%s\"\n", cmd);
    if (cmd[0] == '-' && cmd[1] == '-') {
        std::fprintf(stderr,
            "  Tier B requires 'stellar2d run ...'; flags alone are no longer accepted.\n"
            "  Retry as:   stellar2d run %s ...\n", cmd);
    } else {
        std::fprintf(stderr,
            "  Available subcommands: run | list | describe | validate\n"
            "  (doctor / help planned for later milestones).\n");
    }
    std::fprintf(stderr, "  Run 'stellar2d --help' for the flag reference.\n");
}

// ---------------------------------------------------------------------------
//  Tier B-3 subcommands: list / describe / validate.
//
//  These are plain CPP functions that live next to the run_simulation()
//  dispatch.  They intentionally don't go through parse_cli because they
//  don't need a SimConfig / SimContext — they only need filesystem access
//  and a few string comparisons.
// ---------------------------------------------------------------------------

// Keep in sync with the solver_type dispatch block in run_simulation().
// Tier C will replace this with a SolverSpec-registry autogeneration.
const std::vector<std::pair<const char*, const char*>>& known_solvers() {
    static const std::vector<std::pair<const char*, const char*>> v = {
        {"compressible",     "AmgX-based compressible Euler (requires USE_AMGX)"},
        {"strang",           "2D explicit Strang split HLLC+MUSCL (baseline)"},
        {"wb2d",             "Well-balanced 2D Euler (perturbed t~2 unstable)"},
        {"lowmach",          "JFNK low-Mach with HLLC variants"},
        {"fas",              "FAS multigrid low-Mach"},
        {"fas2",             "FAS2 multigrid variant"},
        {"simple",           "SIMPLE-family incompressible"},
        {"projection",       "Chorin projection low-Mach"},
        {"cart_lag",         "Cartesian Lagrangian Caramana"},
        {"cart_ale",         "Cartesian ALE (Caramana Lag + Eulerian rezone)"},
        {"cart_impl",        "Cartesian implicit ALE (JFNK)"},
        {"cart_ale2",        "Cart ALE + periodic BC + PPM (convection workhorse)"},
        {"athena_vl2",       "Athena++ vl2 port (PLM + HLLC)"},
        {"athena_mhd",       "Athena++ MHD port (§A-§F derivations)"},
        {"radial1d",         "1D Lagrangian JFNK hydro testbed"},
        {"pseudo_spectral",  "2D incompressible NS (cuFFT + IFRK3)"},
        {"anelastic_sl",     "2D Boussinesq / anelastic spectral"},
        {"sph2d_spectral",   "2D thin-shell spectral"},
        {"ale2d",            "Axisymmetric ALE (hoop-stress bug — see design doc)"},
    };
    return v;
}

int run_list(int argc, char** argv) {
    // argv[0] == "list", argv[1] == category
    if (argc < 2) {
        std::fprintf(stderr,
            "ERROR: `stellar2d list` requires a category.\n"
            "  Usage:\n"
            "    stellar2d list profiles\n"
            "    stellar2d list solvers\n");
        return 1;
    }
    const std::string cat = argv[1];
    if (cat == "profiles") {
        auto names = available_profile_names();
        if (names.empty()) {
            std::printf("(no profiles found under config/profiles/; "
                        "are you running from repo root?)\n");
            return 0;
        }
        std::printf("Profiles (config/profiles/):\n");
        for (const auto& n : names) {
            std::printf("  %s\n", n.c_str());
        }
        std::printf("\nLoad with:  stellar2d run --profile <name>\n");
        return 0;
    }
    if (cat == "solvers") {
        std::printf("Solvers (see docs/design/*_design.md and CLAUDE.md §immutable assets):\n");
        for (const auto& p : known_solvers()) {
            std::printf("  %-18s %s\n", p.first, p.second);
        }
        return 0;
    }
    if (cat == "tests") {
        std::fprintf(stderr,
            "ERROR: `stellar2d list tests` is planned for Tier C "
            "(needs per-solver SolverSpec registry).\n");
        return 1;
    }
    std::fprintf(stderr,
        "ERROR: unknown list category \"%s\".\n"
        "  Available: profiles | solvers  (tests planned for Tier C).\n",
        cat.c_str());
    return 1;
}

int run_describe(int argc, char** argv) {
    // argv[0] == "describe", argv[1] == "profile" | "solver" | <name>
    if (argc < 2) {
        std::fprintf(stderr,
            "ERROR: `stellar2d describe` requires a target.\n"
            "  Usage:\n"
            "    stellar2d describe profile <name>\n"
            "    (`stellar2d describe <solver>` planned for Tier C)\n");
        return 1;
    }
    const std::string what = argv[1];
    if (what == "profile") {
        if (argc < 3) {
            std::fprintf(stderr,
                "ERROR: `describe profile` needs a profile name.\n"
                "  Usage:  stellar2d describe profile <name>\n");
            return 1;
        }
        const std::string name = argv[2];
        const std::string path = "config/profiles/" + name + ".toml";
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            std::fprintf(stderr,
                "ERROR: profile \"%s\" not found (looked for %s).\n",
                name.c_str(), path.c_str());
            auto avail = available_profile_names();
            if (!avail.empty()) {
                std::fprintf(stderr, "  Available profiles:\n");
                for (const auto& n : avail) {
                    std::fprintf(stderr, "    %s\n", n.c_str());
                }
            }
            return 1;
        }
        // Cat the file to stdout verbatim; keeps comments intact so users
        // can see the profile's own provenance header.
        std::ifstream in(path);
        if (!in) {
            std::fprintf(stderr, "ERROR: cannot open %s for reading.\n", path.c_str());
            return 1;
        }
        std::cout << in.rdbuf();
        return 0;
    }
    std::fprintf(stderr,
        "ERROR: `describe %s` not implemented.  Supported: `describe profile <name>`.\n"
        "  Tier C will add `describe <solver>` from the SolverSpec registry.\n",
        what.c_str());
    return 1;
}

int run_validate(int argc, char** argv) {
    // argv[0] == "validate", argv[1] == path
    if (argc < 2) {
        std::fprintf(stderr,
            "ERROR: `stellar2d validate` requires a TOML path.\n"
            "  Usage:  stellar2d validate <path-to-config.toml>\n");
        return 1;
    }
    return validate_toml_file(argv[1]);
}

} // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);
    std::signal(SIGTERM, handle_sigint);

    // Global scan for --help / --version: they work as subcommand-free
    // meta flags so `stellar2d --help` and `stellar2d --version` remain
    // idiomatic (compare: git --help).
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 ||
            std::strcmp(argv[i], "-h")     == 0) { print_help();    return 0; }
        if (std::strcmp(argv[i], "--version") == 0) { print_version(); return 0; }
    }

    if (argc < 2) { print_missing_subcommand_error(); return 1; }

    const std::string cmd = argv[1];

    if (cmd == "run") {
        // Shift so callee sees argv[0] == "run", argv[1..] == flags.
        return run_simulation(argc - 1, argv + 1);
    }

    if (cmd == "list")     return run_list    (argc - 1, argv + 1);
    if (cmd == "describe") return run_describe(argc - 1, argv + 1);
    if (cmd == "validate") return run_validate(argc - 1, argv + 1);

    // Tier C will flesh out these stubs.
    if (cmd == "doctor" || cmd == "help") {
        std::fprintf(stderr,
            "ERROR: subcommand \"%s\" is planned for Tier C and not yet "
            "implemented.  See docs/design/cli_unification_plan_2026-05-09.md.\n",
            cmd.c_str());
        return 1;
    }

    print_unknown_subcommand_error(argv[1]);
    return 1;
}
