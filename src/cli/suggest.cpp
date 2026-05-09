#include "cli/suggest.h"

#include <algorithm>

int levenshtein(const std::string& a, const std::string& b) {
    const int m = static_cast<int>(a.size());
    const int n = static_cast<int>(b.size());
    if (m == 0) return n;
    if (n == 0) return m;

    // Roll two rows; prev[j] = D(a[..i-1], b[..j]), curr[j] = D(a[..i], b[..j]).
    std::vector<int> prev(static_cast<size_t>(n) + 1);
    std::vector<int> curr(static_cast<size_t>(n) + 1);
    for (int j = 0; j <= n; ++j) prev[static_cast<size_t>(j)] = j;

    for (int i = 1; i <= m; ++i) {
        curr[0] = i;
        for (int j = 1; j <= n; ++j) {
            const int cost = (a[static_cast<size_t>(i - 1)] ==
                              b[static_cast<size_t>(j - 1)]) ? 0 : 1;
            curr[static_cast<size_t>(j)] = std::min({
                prev[static_cast<size_t>(j)]     + 1,          // deletion
                curr[static_cast<size_t>(j - 1)] + 1,          // insertion
                prev[static_cast<size_t>(j - 1)] + cost,       // substitution
            });
        }
        std::swap(prev, curr);
    }
    return prev[static_cast<size_t>(n)];
}

std::string suggest_closest(const std::string& typo,
                            const std::vector<std::string>& known,
                            int max_distance) {
    int best_dist = max_distance + 1;
    std::string best;
    for (const std::string& k : known) {
        // Quick length-difference short-circuit: if |len(a) - len(b)| already
        // exceeds max_distance the DP cannot produce a smaller result.
        const int la = static_cast<int>(typo.size());
        const int lb = static_cast<int>(k.size());
        if (std::abs(la - lb) > max_distance) continue;

        const int d = levenshtein(typo, k);
        if (d < best_dist) {
            best_dist = d;
            best = k;
        }
    }
    return (best_dist <= max_distance) ? best : std::string{};
}
