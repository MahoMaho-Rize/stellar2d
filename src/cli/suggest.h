#pragma once

// Tier A CLI hardening — did-you-mean suggestions for unknown flags.
//
// See docs/design/cli_unification_plan_2026-05-09.md §3.  The Levenshtein
// edit distance is a conservative heuristic; the default ``max_distance``
// threshold of 3 roughly means "one typo or transposition away".
//
// The table of known flags lives in src/cli/options.cpp (static
// KNOWN_FLAGS vector) — this header only exposes the distance primitives.

#include <string>
#include <vector>

// Standard Levenshtein edit distance (insertions, deletions, substitutions
// all cost 1).  O(|a|·|b|) time, O(min(|a|,|b|)) space.
int levenshtein(const std::string& a, const std::string& b);

// Return the element of ``known`` with the smallest edit distance to
// ``typo``, provided that distance is <= ``max_distance``.  Returns an
// empty string if no candidate is close enough.
std::string suggest_closest(const std::string& typo,
                            const std::vector<std::string>& known,
                            int max_distance = 3);
