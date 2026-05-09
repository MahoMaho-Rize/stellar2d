# stellar2d profile — README
#
# Profile files here are consumed by
#
#     stellar2d run --profile <name> [FLAGS]
#
# which is sugar for
#
#     stellar2d run --config config/profiles/<name>.toml [FLAGS]
#
# (see docs/design/cli_unification_plan_2026-05-09.md §2a, §3e).
#
# ## What belongs here
#
# Each profile is a *flat* TOML file in the same key=value layout as
# `runs/<name>/config.dump.txt`; only list keys that differ from the
# built-in SimConfig defaults.  Unknown keys are a HARD ERROR with a
# Levenshtein did-you-mean hint.
#
# A profile should reproduce a canonical / published / benchmark
# configuration with no user thought required.  Tier B-3 (planned) will
# also make every `runs/<name>/config.toml` directly usable as a profile
# for bit-identical re-runs.
#
# ## Priority ordering
#
# When both --config and --profile are passed:
#
#     CLI flags > --profile NAME > --config FILE > built-in defaults
#
# Profiles overlay on top of any --config file, and CLI flags override
# everything.  The exact ordering is documented in §3f of the plan.
#
# ## Current profiles (Tier B-2)
#
# | Profile              | Solver        | Test             | Notes                  |
# |----------------------|---------------|------------------|------------------------|
# | lane_emden_n15       | fas           | lane_emden       | Polytropic static eq.  |
# | sod_cart_ale2        | cart_ale2     | sod              | Quasi-1D Sod shocktube |
# | kh_lecoanet          | cart_ale2     | kh_lecoanet      | Lecoanet 2015 KH       |
# | andrassy2022         | cart_ale2     | andrassy2022     | Andrassy+ O-shell IC   |
# | linwave_vl2          | athena_vl2    | entropy_wave     | vl2+PLM conv. probe    |
# | mhd_brio_wu          | athena_mhd    | brio_wu          | Brio-Wu 1D MHD shock   |
#
# ## Extending
#
# When adding a new profile:
#   1. Only set keys that differ from SimConfig defaults;
#   2. Add a one-line entry to the table above;
#   3. Verify the profile parses: `stellar2d run --profile <name>
#      --tend 0.001 --nr 16 --ntheta 8` should proceed to
#      `[config_dump] wrote ...` and run a few steps.
#
# Tier C will introduce grouped TOML ([grid], [solver.cart_ale2], ...)
# and namespaced CLI (`--solver.cart_ale2.ppm`) — profiles written today
# will be automatically migrated by a one-shot script, not by hand edits.
