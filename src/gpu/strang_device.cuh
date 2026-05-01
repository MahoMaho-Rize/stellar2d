#pragma once

// ============================================================
// strang_device.cuh — Device helper functions for Strang solver
//
// Shared between strang_solver.cu and test files.
// ============================================================

#ifdef __CUDACC__

// Eq. (13.3): MC (Monotonized Central) limiter.
__device__ __forceinline__
double d_mc_limit(double a, double b)
{
    if (a * b <= 0.0) return 0.0;
    double s  = copysign(1.0, a);
    double av = 0.5 * fabs(a + b);
    double ta = 2.0 * fabs(a);
    double tb = 2.0 * fabs(b);
    return s * fmin(av, fmin(ta, tb));
}

// Eq. (13.2): isentropic HSE background evaluated at arbitrary y.
__device__ __forceinline__
double d_hse_rho(double y, double rho0_gm1, double coeff, double inv_gm1)
{
    double arg = rho0_gm1 - coeff * y;
    return pow(fmax(arg, 1e-20), inv_gm1);
}

// Eq. (13.2): P_HSE = K · ρ_HSE^γ.
__device__ __forceinline__
double d_hse_p(double rho, double K, double gamma)
{
    return K * pow(rho, gamma);
}

// Eq. (1.1): conserved → primitive (total values, not perturbation).
__device__ __forceinline__
void d_cons2prim(double rho, double mx, double my, double E_tot,
                 double gm1,
                 double& u, double& v, double& P)
{
    double inv_rho = 1.0 / fmax(rho, 1e-30);
    u = mx * inv_rho;
    v = my * inv_rho;
    P = gm1 * (E_tot - 0.5 * rho * (u*u + v*v));   // Eq. (1.2)
    P = fmax(P, 1e-30);
}

// ---- Euler flux in x-direction ----
__device__ __forceinline__
void d_euler_flux_x(double rho, double u, double v, double P, double gm1,
                    double& f0, double& f1, double& f2, double& f3)
{
    double E = P / gm1 + 0.5 * rho * (u*u + v*v);
    f0 = rho * u;
    f1 = rho * u * u + P;
    f2 = rho * u * v;
    f3 = (E + P) * u;
}

// ---- Euler flux in y-direction ----
__device__ __forceinline__
void d_euler_flux_y(double rho, double u, double v, double P, double gm1,
                    double& g0, double& g1, double& g2, double& g3)
{
    double E = P / gm1 + 0.5 * rho * (u*u + v*v);
    g0 = rho * v;
    g1 = rho * u * v;
    g2 = rho * v * v + P;
    g3 = (E + P) * v;
}

// ============================================================
//  LM-HLLC: Low-Mach corrected HLLC Riemann solver
//
//  Generic 1D solver in normal/tangential frame.
//  Input:  (ρ, u_n, u_t, P) for left and right states
//  Output: flux (f_rho, f_mn, f_mt, f_E) in normal direction
//
//  Low-Mach correction:
//    M_local = (|un_L| + |un_R|) / (c_L + c_R)
//    f(M) = clamp(M_local, M_cutoff, 1.0)
//    In S* formula: (pR - pL) → f(M) * (pR - pL)
//    Effect: at M→0, f→0, suppresses excess pressure dissipation
//            at M≈1, f=1, recovers standard HLLC for shocks
// ============================================================

__device__ __forceinline__
void d_lmhllc(double rhoL, double unL, double utL, double PL,
              double rhoR, double unR, double utR, double PR,
              double gamma,
              double& f_rho, double& f_mn, double& f_mt, double& f_E)
{
    double gm1 = gamma - 1.0;

    // Sound speeds
    double cL = sqrt(gamma * PL / fmax(rhoL, 1e-30));
    double cR = sqrt(gamma * PR / fmax(rhoR, 1e-30));

    // Eq. (4.1): Davis wave-speed estimates.
    double SL = fmin(unL - cL, unR - cR);
    double SR = fmax(unL + cL, unR + cR);

    // Eq. (14.1-14.2): local Mach number + blending factor f(M).
    double M_cutoff = 1e-3;
    double M_local  = (fabs(unL) + fabs(unR)) / fmax(cL + cR, 1e-30);
    double fM       = fmin(1.0, fmax(M_local, M_cutoff));

    // Eq. (14.3): contact wave S* with LM-corrected pressure jump.
    double denom = rhoL * (SL - unL) - rhoR * (SR - unR);
    if (fabs(denom) < 1e-300) denom = copysign(1e-300, denom);
    double S_star = (fM * (PR - PL) + rhoL * unL * (SL - unL)
                     - rhoR * unR * (SR - unR)) / denom;

    // Total energies
    double EL = PL / gm1 + 0.5 * rhoL * (unL*unL + utL*utL);
    double ER = PR / gm1 + 0.5 * rhoR * (unR*unR + utR*utR);

    // Physical fluxes
    //   F(U) = (ρ un, ρ un² + P, ρ un ut, (E+P) un)
    double FL0 = rhoL * unL;
    double FL1 = rhoL * unL * unL + PL;
    double FL2 = rhoL * unL * utL;
    double FL3 = (EL + PL) * unL;

    double FR0 = rhoR * unR;
    double FR1 = rhoR * unR * unR + PR;
    double FR2 = rhoR * unR * utR;
    double FR3 = (ER + PR) * unR;

    if (SL >= 0.0) {
        // Supersonic from left
        f_rho = FL0;  f_mn = FL1;  f_mt = FL2;  f_E = FL3;
        return;
    }
    if (SR <= 0.0) {
        // Supersonic from right
        f_rho = FR0;  f_mn = FR1;  f_mt = FR2;  f_E = FR3;
        return;
    }

    // Star-region pressures  (Eq. 10.26 in Toro)
    double pstar = PL + rhoL * (SL - unL) * (S_star - unL);
    pstar = fmax(pstar, 0.0);

    if (S_star >= 0.0) {
        // Left star state
        double inv = 1.0 / fmax(fabs(SL - S_star), 1e-300);
        double coeff = SL - unL;
        double rhoSL = rhoL * coeff / (SL - S_star);
        rhoSL = fmax(rhoSL, 1e-30);

        // U*L
        double U0 = rhoSL;
        double U1 = rhoSL * S_star;
        double U2 = rhoSL * utL;
        double U3 = rhoSL * (EL / rhoL + (S_star - unL)
                    * (S_star + PL / (rhoL * coeff)));

        f_rho = FL0 + SL * (U0 - rhoL);
        f_mn  = FL1 + SL * (U1 - rhoL * unL);
        f_mt  = FL2 + SL * (U2 - rhoL * utL);
        f_E   = FL3 + SL * (U3 - EL);
    } else {
        // Right star state
        double coeff = SR - unR;
        double rhoSR = rhoR * coeff / (SR - S_star);
        rhoSR = fmax(rhoSR, 1e-30);

        double U0 = rhoSR;
        double U1 = rhoSR * S_star;
        double U2 = rhoSR * utR;
        double U3 = rhoSR * (ER / rhoR + (S_star - unR)
                    * (S_star + PR / (rhoR * coeff)));

        f_rho = FR0 + SR * (U0 - rhoR);
        f_mn  = FR1 + SR * (U1 - rhoR * unR);
        f_mt  = FR2 + SR * (U2 - rhoR * utR);
        f_E   = FR3 + SR * (U3 - ER);
    }
}

#endif // __CUDACC__
