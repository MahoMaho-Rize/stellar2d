// FAS multigrid: restriction, prolongation, V-cycle, solve

#include "fas_solver.cuh"
#include "gpu_linalg.cuh"
#include <cmath>
#include <vector>

// ========================= 4-DOF Restriction ========================
// Volume-weighted restriction of conservative variables.

__global__
void k_fas_restrict_state(
    const double* f_rho, const double* f_mr, const double* f_mt, const double* f_rhoE,
    const double* f_vol,
    double* c_rho, double* c_mr, double* c_mt, double* c_rhoE,
    int cnr, int cnt, int fnt, int fng, int cng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= cnr*cnt) return;
    int ic = flat/cnt, jc = flat%cnt;
    int if0 = 2*ic, jf0 = 2*jc;
    int ck = (ic+cng)*(cnt+2*cng) + (jc+cng);

    double s_rho=0, s_mr=0, s_mt=0, s_rhoE=0, s_vol=0;
    for (int di = 0; di < 2; ++di) {
        int fi = if0 + di;
        for (int dj = 0; dj < 2; ++dj) {
            int fj = jf0 + dj;
            int fk = (fi+fng)*(fnt+2*fng) + (fj+fng);
            int ff = fi*fnt + fj;
            double v = f_vol[ff];
            s_rho += f_rho[fk]*v; s_mr += f_mr[fk]*v;
            s_mt  += f_mt[fk]*v;  s_rhoE += f_rhoE[fk]*v;
            s_vol += v;
        }
    }
    double inv_v = 1.0 / fmax(s_vol, 1e-300);
    c_rho[ck]  = s_rho * inv_v;
    c_mr[ck]   = s_mr  * inv_v;
    c_mt[ck]   = s_mt  * inv_v;
    c_rhoE[ck] = s_rhoE* inv_v;
}

void FasSolver::restrict_state(int fine, int coarse) {
    FasLevel& fl = levels[fine], &cl = levels[coarse];
    int cn = cl.nr * cl.nt, B = 256;
    k_fas_restrict_state<<<(cn+B-1)/B,B>>>(
        fl.d_rho, fl.d_mr, fl.d_mt, fl.d_rhoE, fl.d_cell_volume,
        cl.d_rho, cl.d_mr, cl.d_mt, cl.d_rhoE,
        cl.nr, cl.nt, fl.nt, fl.ng, cl.ng);
}

// ========================= FAS defect restriction ========================
// f_H = R(Î·u_h) + Î·(f_h - R(u_h))
// = R(u_H) + Î·(Uⁿ_h/dt + τ_h - R(u_h)) - Î·(Uⁿ_h/dt + τ_h) + R(u_H)
// Simpler: f_H = fas_rhs_H = Uⁿ_H/dt + τ_H
// where τ_H = R(u_H) - Î·R(u_h) + Î·(Uⁿ_h/dt + old_τ_h) - Uⁿ_H/dt
//
// Implementation:
//   1. Compute R(u_h) on fine level → stored in d_res
//   2. Compute F(u_h) = R(u_h) - U_h/dt + fas_rhs_h → d_res  (fine level defect)
//   3. Restrict defect: d_H = Î·F(u_h)
//   4. Restrict state: u_H = Î·u_h
//   5. Compute R(u_H) on coarse level
//   6. fas_rhs_H = U_H/dt + R(u_H) - d_H (so that F(u_H)=R(u_H)-U_H/dt+fas_rhs_H = -d_H ≈ -Î·F(u_h))

__global__
void k_fas_restrict_defect(
    const double* f_res, const double* f_vol,
    double* c_defect,
    int cnr, int cnt, int fnr, int fnt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= cnr*cnt) return;
    int ic = flat/cnt, jc = flat%cnt;
    int if0 = 2*ic, jf0 = 2*jc;
    int fn = fnr*fnt;
    int cn = cnr*cnt;

    for (int eq = 0; eq < 4; ++eq) {
        double ws=0, vs=0;
        for (int di = 0; di < 2; ++di) {
            int fi = if0 + di;
            if (fi >= fnr) continue;
            for (int dj = 0; dj < 2; ++dj) {
                int fj = jf0 + dj;
                if (fj >= fnt) continue;
                int ff = fi*fnt + fj;
                double v = f_vol[ff];
                ws += f_res[eq*fn + ff] * v;
                vs += v;
            }
        }
        c_defect[eq*cn + flat] = (vs > 0) ? ws/vs : 0.0;
    }
}

// Assemble FAS RHS on coarse level:
// fas_rhs_H = U_H/dt - R(U_H) + restricted_defect
// This ensures F(U_H) = R(U_H) - U_H/dt + fas_rhs_H = restricted_defect
__global__
void k_fas_assemble_coarse_rhs(
    const double* R_H,     // R(U_H) after restrict + compute_residual
    const double* rho_H, const double* mr_H, const double* mt_H, const double* rhoE_H,
    const double* defect,  // restricted fine-level defect
    double* gpu_rhs,
    double inv_dt, int nr, int nt, int ng) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int n = nr*nt;
    int k = gpu_idx(flat/nt, flat%nt, nt, ng);

    // gpu_rhs = U_H/dt - R(U_H) + defect
    // So F = R(U_H) - U_H/dt + gpu_rhs = defect (initially)
    gpu_rhs[flat]       = inv_dt*rho_H[k]  - R_H[flat]       + defect[flat];
    gpu_rhs[n + flat]   = inv_dt*mr_H[k]   - R_H[n + flat]   + defect[n + flat];
    gpu_rhs[2*n + flat] = inv_dt*mt_H[k]   - R_H[2*n + flat] + defect[2*n + flat];
    gpu_rhs[3*n + flat] = inv_dt*rhoE_H[k] - R_H[3*n + flat] + defect[3*n + flat];
}

void FasSolver::restrict_defect(int fine, int coarse, double g0_over_dt) {
    FasLevel& fl = levels[fine], &cl = levels[coarse];
    int fn = fl.nr * fl.nt, cn = cl.nr * cl.nt, B = 256;

    compute_F(fine, g0_over_dt);

    k_fas_restrict_defect<<<(cn+B-1)/B,B>>>(
        fl.d_res, fl.d_cell_volume,
        cl.d_Un,
        cl.nr, cl.nt, fl.nr, fl.nt);

    restrict_state(fine, coarse);
    apply_floor(coarse);

    compute_residual(coarse);

    k_fas_assemble_coarse_rhs<<<(cn+B-1)/B,B>>>(
        cl.d_res,
        cl.d_rho, cl.d_mr, cl.d_mt, cl.d_rhoE,
        cl.d_Un,
        cl.d_fas_rhs,
        g0_over_dt, cl.nr, cl.nt, cl.ng);

}

// ========================= Prolongation + correction ========================
// u_h += P · (u_H - Î·u_h)
// Piecewise-constant prolongation (injection) + floor protection.

__global__
void k_fas_prolongate_correct(
    double* f_rho, double* f_mr, double* f_mt, double* f_rhoE,
    const double* c_rho, const double* c_mr, const double* c_mt, const double* c_rhoE,
    const double* save_rho, const double* save_mr, const double* save_mt, const double* save_rhoE,
    const double* f_rho0, double atm_thresh,
    int cnr, int cnt, int fnt, int fng, int cng, double gam) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= cnr*cnt) return;
    int ic = flat/cnt, jc = flat%cnt;
    int ck = (ic+cng)*(cnt+2*cng) + (jc+cng);

    double d_rho  = c_rho[ck]  - save_rho[flat];
    double d_mr   = c_mr[ck]   - save_mr[flat];
    double d_mt   = c_mt[ck]   - save_mt[flat];
    double d_rhoE = c_rhoE[ck] - save_rhoE[flat];

    for (int di = 0; di < 2; ++di) {
        int fi = 2*ic + di;
        for (int dj = 0; dj < 2; ++dj) {
            int fj = 2*jc + dj;
            int ff = fi*fnt + fj;

            // Skip atmosphere cells — coarse correction is meaningless there
            if (f_rho0[ff] < atm_thresh) continue;

            int fk = (fi+fng)*(fnt+2*fng) + (fj+fng);

            double rho_f = fmax(f_rho[fk], 1e-20);
            double KE_f = 0.5*(f_mr[fk]*f_mr[fk] + f_mt[fk]*f_mt[fk]) / rho_f;
            double P_f = fmax((gam - 1.0) * (f_rhoE[fk] - KE_f), 1e-30);
            double cs = sqrt(gam * P_f / rho_f);

            const double beta = 0.5;
            double lim_rho  = beta * rho_f;
            double lim_mom  = beta * rho_f * cs;
            double lim_rhoE = beta * fmax(f_rhoE[fk], 1e-20);

            double cr  = fmax(-lim_rho,  fmin(d_rho,  lim_rho));
            double cmr = fmax(-lim_mom,  fmin(d_mr,   lim_mom));
            double cmt = fmax(-lim_mom,  fmin(d_mt,   lim_mom));
            double crE = fmax(-lim_rhoE, fmin(d_rhoE, lim_rhoE));

            f_rho[fk]  += cr;
            f_mr[fk]   += cmr;
            f_mt[fk]   += cmt;
            f_rhoE[fk] += crE;
        }
    }
}

void FasSolver::prolongate_correct(int coarse, int fine) {
    FasLevel& cl = levels[coarse], &fl = levels[fine];
    int cn = cl.nr * cl.nt, B = 256;

    // cl.d_save contains the pre-solve restricted state
    k_fas_prolongate_correct<<<(cn+B-1)/B,B>>>(
        fl.d_rho, fl.d_mr, fl.d_mt, fl.d_rhoE,
        cl.d_rho, cl.d_mr, cl.d_mt, cl.d_rhoE,
        cl.d_save, cl.d_save + cn, cl.d_save + 2*cn, cl.d_save + 3*cn,
        fl.d_rho0, atm_rho_thresh,
        cl.nr, cl.nt, fl.nt, fl.ng, cl.ng, gamma);
    apply_floor(fine);
}

// ========================= FAS V-cycle ========================

void FasSolver::fas_vcycle(int l, double dt, double g0_over_dt) {
    // Recurse to coarser levels if available (full V-cycle, not just two-grid)
    if (l >= n_levels - 1) {
        assemble_smoother(l, g0_over_dt);
        smooth(l, dt, g0_over_dt, NU1 + NU2);
        return;
    }

    assemble_smoother(l, g0_over_dt);
    smooth(l, dt, g0_over_dt, NU1);

    FasLevel& cl = levels[l + 1];
    int cn = cl.nr * cl.nt, B = 256;

    restrict_defect(l, l + 1, g0_over_dt);

    k_fas_pack_flat<<<(cn+B-1)/B,B>>>(
        cl.d_rho, cl.d_mr, cl.d_mt, cl.d_rhoE,
        cl.d_save, cl.nr, cl.nt, cl.ng);

    fas_vcycle(l + 1, dt, g0_over_dt);

    prolongate_correct(l + 1, l);

    assemble_smoother(l, g0_over_dt);
    smooth(l, dt, g0_over_dt, NU2);
}
