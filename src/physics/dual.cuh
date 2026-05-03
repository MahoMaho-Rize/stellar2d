#pragma once

// Forward-mode automatic differentiation.
//
// Dual<N> carries a value + N directional derivatives. All arithmetic is
// dimension-N SIMD-style (plain loops, compiler unrolls for small N). Host +
// device compatible. No allocations.
//
// Two canonical use cases in radial1d:
//   1. Exact J·v matvec (N=1): seed d = (v_i · e_i, direction i), evaluate
//      F(U + d) once; out[i].g[0] = (J·v)_i. Replaces 2 FD evals with 1 AD.
//   2. Block-tridiag PC build (N=9): seed d_ff(zone, field) = 1 for nine
//      (color, field) probes, evaluate F twice per zone (colors 0/1/2 or
//      just per-zone Dual<3>). Replaces 9 FD matvecs with 1 AD pass.
//
// Reference: Hogan 2014 "Fast reverse-mode autodiff", Griewank &
// Walther 2008 "Evaluating Derivatives".

#include <cmath>

#ifdef __CUDACC__
#define DUAL_HD __host__ __device__ inline
#else
#define DUAL_HD inline
#endif

namespace dual {

template <int N>
struct Dual {
    double v;        // value
    double g[N];     // gradient (N directional derivatives)

    DUAL_HD Dual() : v(0.0) { for (int k = 0; k < N; ++k) g[k] = 0.0; }
    DUAL_HD Dual(double x) : v(x) { for (int k = 0; k < N; ++k) g[k] = 0.0; }

    // Static constructor: constant (all derivatives zero)
    DUAL_HD static Dual constant(double x) { return Dual(x); }

    // Static constructor: seed direction i with magnitude 1 in component
    DUAL_HD static Dual seed(double x, int i) {
        Dual r(x);
        if (i >= 0 && i < N) r.g[i] = 1.0;
        return r;
    }
};

// ---- Arithmetic (Dual ⊕ Dual, Dual ⊕ double, double ⊕ Dual) ----
template <int N> DUAL_HD Dual<N> operator+(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> r; r.v = a.v + b.v;
    for (int k = 0; k < N; ++k) r.g[k] = a.g[k] + b.g[k];
    return r;
}
template <int N> DUAL_HD Dual<N> operator+(const Dual<N>& a, double b) {
    Dual<N> r; r.v = a.v + b;
    for (int k = 0; k < N; ++k) r.g[k] = a.g[k];
    return r;
}
template <int N> DUAL_HD Dual<N> operator+(double a, const Dual<N>& b) { return b + a; }

template <int N> DUAL_HD Dual<N> operator-(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> r; r.v = a.v - b.v;
    for (int k = 0; k < N; ++k) r.g[k] = a.g[k] - b.g[k];
    return r;
}
template <int N> DUAL_HD Dual<N> operator-(const Dual<N>& a, double b) {
    Dual<N> r; r.v = a.v - b;
    for (int k = 0; k < N; ++k) r.g[k] = a.g[k];
    return r;
}
template <int N> DUAL_HD Dual<N> operator-(double a, const Dual<N>& b) {
    Dual<N> r; r.v = a - b.v;
    for (int k = 0; k < N; ++k) r.g[k] = -b.g[k];
    return r;
}
template <int N> DUAL_HD Dual<N> operator-(const Dual<N>& a) {
    Dual<N> r; r.v = -a.v;
    for (int k = 0; k < N; ++k) r.g[k] = -a.g[k];
    return r;
}

template <int N> DUAL_HD Dual<N> operator*(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> r; r.v = a.v * b.v;
    for (int k = 0; k < N; ++k) r.g[k] = a.g[k] * b.v + a.v * b.g[k];
    return r;
}
template <int N> DUAL_HD Dual<N> operator*(const Dual<N>& a, double b) {
    Dual<N> r; r.v = a.v * b;
    for (int k = 0; k < N; ++k) r.g[k] = a.g[k] * b;
    return r;
}
template <int N> DUAL_HD Dual<N> operator*(double a, const Dual<N>& b) { return b * a; }

template <int N> DUAL_HD Dual<N> operator/(const Dual<N>& a, const Dual<N>& b) {
    Dual<N> r; r.v = a.v / b.v;
    double inv = 1.0 / b.v;
    double inv2 = inv * inv;
    for (int k = 0; k < N; ++k) r.g[k] = (a.g[k] * b.v - a.v * b.g[k]) * inv2;
    return r;
}
template <int N> DUAL_HD Dual<N> operator/(const Dual<N>& a, double b) {
    Dual<N> r; double inv = 1.0 / b; r.v = a.v * inv;
    for (int k = 0; k < N; ++k) r.g[k] = a.g[k] * inv;
    return r;
}
template <int N> DUAL_HD Dual<N> operator/(double a, const Dual<N>& b) {
    Dual<N> r; r.v = a / b.v;
    double inv2 = 1.0 / (b.v * b.v);
    for (int k = 0; k < N; ++k) r.g[k] = -a * b.g[k] * inv2;
    return r;
}

// ---- Compound assignment ----
template <int N> DUAL_HD Dual<N>& operator+=(Dual<N>& a, const Dual<N>& b) { a = a + b; return a; }
template <int N> DUAL_HD Dual<N>& operator+=(Dual<N>& a, double b)          { a = a + b; return a; }
template <int N> DUAL_HD Dual<N>& operator-=(Dual<N>& a, const Dual<N>& b) { a = a - b; return a; }
template <int N> DUAL_HD Dual<N>& operator-=(Dual<N>& a, double b)          { a = a - b; return a; }
template <int N> DUAL_HD Dual<N>& operator*=(Dual<N>& a, const Dual<N>& b) { a = a * b; return a; }
template <int N> DUAL_HD Dual<N>& operator*=(Dual<N>& a, double b)          { a = a * b; return a; }
template <int N> DUAL_HD Dual<N>& operator/=(Dual<N>& a, const Dual<N>& b) { a = a / b; return a; }
template <int N> DUAL_HD Dual<N>& operator/=(Dual<N>& a, double b)          { a = a / b; return a; }

// ---- Comparisons: act on value only. Derivatives of ordering are zero ----
template <int N> DUAL_HD bool operator< (const Dual<N>& a, const Dual<N>& b) { return a.v <  b.v; }
template <int N> DUAL_HD bool operator<=(const Dual<N>& a, const Dual<N>& b) { return a.v <= b.v; }
template <int N> DUAL_HD bool operator> (const Dual<N>& a, const Dual<N>& b) { return a.v >  b.v; }
template <int N> DUAL_HD bool operator>=(const Dual<N>& a, const Dual<N>& b) { return a.v >= b.v; }
template <int N> DUAL_HD bool operator==(const Dual<N>& a, const Dual<N>& b) { return a.v == b.v; }
template <int N> DUAL_HD bool operator!=(const Dual<N>& a, const Dual<N>& b) { return a.v != b.v; }

template <int N> DUAL_HD bool operator< (const Dual<N>& a, double b) { return a.v <  b; }
template <int N> DUAL_HD bool operator<=(const Dual<N>& a, double b) { return a.v <= b; }
template <int N> DUAL_HD bool operator> (const Dual<N>& a, double b) { return a.v >  b; }
template <int N> DUAL_HD bool operator>=(const Dual<N>& a, double b) { return a.v >= b; }
template <int N> DUAL_HD bool operator==(const Dual<N>& a, double b) { return a.v == b; }
template <int N> DUAL_HD bool operator!=(const Dual<N>& a, double b) { return a.v != b; }

template <int N> DUAL_HD bool operator< (double a, const Dual<N>& b) { return a <  b.v; }
template <int N> DUAL_HD bool operator<=(double a, const Dual<N>& b) { return a <= b.v; }
template <int N> DUAL_HD bool operator> (double a, const Dual<N>& b) { return a >  b.v; }
template <int N> DUAL_HD bool operator>=(double a, const Dual<N>& b) { return a >= b.v; }

// ---- Transcendental functions (chain rule: d f(x) / dx_i = f'(x) · dx/dx_i) ----

template <int N> DUAL_HD Dual<N> sqrt(const Dual<N>& a) {
    Dual<N> r; r.v = ::sqrt(a.v);
    double inv = (r.v > 1e-300) ? 0.5 / r.v : 0.0;
    for (int k = 0; k < N; ++k) r.g[k] = a.g[k] * inv;
    return r;
}

template <int N> DUAL_HD Dual<N> exp(const Dual<N>& a) {
    Dual<N> r; r.v = ::exp(a.v);
    for (int k = 0; k < N; ++k) r.g[k] = a.g[k] * r.v;
    return r;
}

template <int N> DUAL_HD Dual<N> log(const Dual<N>& a) {
    Dual<N> r; r.v = ::log(a.v);
    double inv = (a.v != 0.0) ? 1.0 / a.v : 0.0;
    for (int k = 0; k < N; ++k) r.g[k] = a.g[k] * inv;
    return r;
}

template <int N> DUAL_HD Dual<N> log10(const Dual<N>& a) {
    constexpr double inv_ln10 = 0.43429448190325176;  // 1/ln(10)
    Dual<N> r; r.v = ::log10(a.v);
    double inv = (a.v != 0.0) ? inv_ln10 / a.v : 0.0;
    for (int k = 0; k < N; ++k) r.g[k] = a.g[k] * inv;
    return r;
}

// pow(Dual, double): d/dx x^p = p · x^(p-1). Handles p=0, p=1, and negative x=0.
template <int N> DUAL_HD Dual<N> pow(const Dual<N>& a, double p) {
    Dual<N> r; r.v = ::pow(a.v, p);
    if (p == 0.0 || a.v == 0.0) {
        for (int k = 0; k < N; ++k) r.g[k] = 0.0;
        return r;
    }
    double deriv = p * ::pow(a.v, p - 1.0);
    for (int k = 0; k < N; ++k) r.g[k] = a.g[k] * deriv;
    return r;
}

// pow(Dual, Dual): x^y = exp(y log x) → derivatives from that identity
template <int N> DUAL_HD Dual<N> pow(const Dual<N>& a, const Dual<N>& b) {
    return exp(b * log(a));
}

template <int N> DUAL_HD Dual<N> fabs(const Dual<N>& a) {
    Dual<N> r;
    if (a.v >= 0.0) {
        r.v = a.v;
        for (int k = 0; k < N; ++k) r.g[k] = a.g[k];
    } else {
        r.v = -a.v;
        for (int k = 0; k < N; ++k) r.g[k] = -a.g[k];
    }
    return r;
}

template <int N> DUAL_HD Dual<N> fmax(const Dual<N>& a, const Dual<N>& b) {
    return (a.v >= b.v) ? a : b;
}
template <int N> DUAL_HD Dual<N> fmax(const Dual<N>& a, double b) {
    return (a.v >= b) ? a : Dual<N>(b);
}
template <int N> DUAL_HD Dual<N> fmax(double a, const Dual<N>& b) {
    return (a >= b.v) ? Dual<N>(a) : b;
}
template <int N> DUAL_HD Dual<N> fmin(const Dual<N>& a, const Dual<N>& b) {
    return (a.v <= b.v) ? a : b;
}
template <int N> DUAL_HD Dual<N> fmin(const Dual<N>& a, double b) {
    return (a.v <= b) ? a : Dual<N>(b);
}
template <int N> DUAL_HD Dual<N> fmin(double a, const Dual<N>& b) {
    return (a <= b.v) ? Dual<N>(a) : b;
}

}  // namespace dual

#undef DUAL_HD
