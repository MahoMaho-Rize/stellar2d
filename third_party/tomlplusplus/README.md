# toml++ single-header drop-in (v3.4.0)

[toml++](https://github.com/marzer/tomlplusplus) — C++17 TOML parser +
generator, MIT-licensed, header-only.  We use the upstream single-header
release (`toml.hpp`, ~17 k lines, ~460 KB) vendored here directly to keep
the build self-contained — no git submodule, no conda dependency, no
pixi-task to fetch.

## Why we use it

Commit 2 of the CLI unification roadmap
(`docs/design/cli_unification_plan_2026-05-09.md` §3-§6) introduces
`stellar2d run --config FILE` and `--profile NAME` backed by TOML files.
toml++ was chosen over the alternatives because:

1. **Single-header drop-in** — no cmake find_package / submodule /
   conda channel footprint; a `git clone` of stellar2d is still buildable
   offline with zero fetch step.
2. **Mature parser** — handles the full TOML 1.0.0 spec including dotted
   keys, inline tables, nested arrays and datetimes.  We rely on dotted
   access (`tbl["solver"]["cart_ale2"]["ppm"]`) for the namespaced form
   planned in Tier C.
3. **Exception-aware** — opt-in via `TOML_EXCEPTIONS`, so the parser can
   return `toml::parse_result` with `.failed()` / `.error()` without
   throwing.  We run with exceptions enabled (GCC default), get the
   convenient try/catch form.
4. **Preserves comments on re-serialise** — important for Tier B-3's
   `runs/<name>/config.toml` paper trail.

## Version / hash

- Tag:     `v3.4.0` (released 2023-11-22)
- Source:  `https://github.com/marzer/tomlplusplus/releases/download/v3.4.0/toml.hpp`
- SHA-256: `6b5172ad4dd6519aec67b919181fa7a38a2234131e5b2afa232dfe444819783e`

## Refreshing

```bash
# Bump to a newer release
TAG=v3.4.0
curl -sSfL "https://raw.githubusercontent.com/marzer/tomlplusplus/${TAG}/toml.hpp" \
    -o third_party/tomlplusplus/toml.hpp
sha256sum third_party/tomlplusplus/toml.hpp   # update this README
```

No per-release CMake changes are needed — CMakeLists.txt just adds the
directory to the `stellar2d` target's `target_include_directories`.
