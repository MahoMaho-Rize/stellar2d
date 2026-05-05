// ALE2D kernels: axisymmetric Lagrangian Caramana-compatible hydro.
//
// Conventions:
//   (R, Z) cylindrical coordinates. Z = symmetry axis.
//   Spherical coords map: R = r·sin θ,  Z = r·cos θ.
//   θ = 0 → north pole (+Z axis), θ = π → south pole (-Z axis).
//
// Topology:
//   cell  (ic, jc)  ∈ [0, nr) × [0, nt)
//   node  (in, jn)  ∈ [0, nr] × [0, nt]
//   cell (ic, jc) has 4 corners  k = 0..3 (CCW):
//     0: (ic,   jc)     inner-south
//     1: (ic+1, jc)     outer-south
//     2: (ic+1, jc+1)   outer-north
//     3: (ic,   jc+1)   inner-north
//
// Revolved volume (Pappus; valid for CCW polygon with Z being the symmetry axis):
//   V = (π/3) · Σ_k (R_k² + R_k R_{k+1} + R_{k+1}²) · (Z_{k+1} − Z_k)
//
// Outward-normal-weighted annular area on edge (k, k+1):
//   A_edge = 2π R_mid · (dZ, −dR)  with  dR = R_{k+1}−R_k, dZ = Z_{k+1}−Z_k
//
// Subcell force (corner k of cell c):
//   F_k = ½ · (P+Q) · [A_edge(k−1→k) + A_edge(k→k+1)]
// Node total force is the sum over the ≤4 incident cells.

#include "ale2d_solver.cuh"
#include "fas_common.cuh"  // CUDA_CHECK
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ __forceinline__ int ale_n(int in, int jn, int nnode_t) { return in * nnode_t + jn; }
__device__ __forceinline__ int ale_c(int ic, int jc, int nt) { return ic * nt + jc; }

// ============================================================
// Geometry: volume, cross-sectional area, min-height, edge stats
// ============================================================
__global__
void k_ale_geometry(const double* R, const double* Z,
                    double* Vol, double* Area, double* minheight,
                    int nr, int nt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int ic = flat / nt, jc = flat % nt;
    int nnt = nt + 1;
    int i0 = ale_n(ic,   jc,   nnt);
    int i1 = ale_n(ic+1, jc,   nnt);
    int i2 = ale_n(ic+1, jc+1, nnt);
    int i3 = ale_n(ic,   jc+1, nnt);
    double R0=R[i0], R1=R[i1], R2=R[i2], R3=R[i3];
    double Z0=Z[i0], Z1=Z[i1], Z2=Z[i2], Z3=Z[i3];

    // Signed 2D polygon area (positive if CCW)
    double A2 = 0.5 * ((R0*Z1 - R1*Z0) + (R1*Z2 - R2*Z1) + (R2*Z3 - R3*Z2) + (R3*Z0 - R0*Z3));
    Area[flat] = fabs(A2);

    // Revolved volume (Pappus, sum over edges)
    double V =  (R0*R0 + R0*R1 + R1*R1) * (Z1 - Z0);
    V       += (R1*R1 + R1*R2 + R2*R2) * (Z2 - Z1);
    V       += (R2*R2 + R2*R3 + R3*R3) * (Z3 - Z2);
    V       += (R3*R3 + R3*R0 + R0*R0) * (Z0 - Z3);
    V *= M_PI / 3.0;
    Vol[flat] = fabs(V);

    // Min-height: minimum distance from a node to the opposite edge
    auto dist_pt_edge = [](double Px, double Py,
                           double Ax, double Ay, double Bx, double By) -> double {
        double dx = Bx - Ax, dy = By - Ay;
        double len2 = dx*dx + dy*dy;
        if (len2 < 1e-30) return sqrt((Px-Ax)*(Px-Ax) + (Py-Ay)*(Py-Ay));
        // perpendicular distance
        double cross = (Bx - Ax) * (Py - Ay) - (By - Ay) * (Px - Ax);
        return fabs(cross) / sqrt(len2);
    };
    double h0 = dist_pt_edge(R0, Z0, R1, Z1, R2, Z2);
    double h1 = dist_pt_edge(R1, Z1, R2, Z2, R3, Z3);
    double h2 = dist_pt_edge(R2, Z2, R3, Z3, R0, Z0);
    double h3 = dist_pt_edge(R3, Z3, R0, Z0, R1, Z1);
    double mh = fmin(fmin(h0, h1), fmin(h2, h3));
    // Guard against degenerate inner row where two corners coincide
    if (mh < 1e-12) {
        // Fall back to sqrt(Area) as a crude length
        double Aabs = fabs(A2);
        mh = sqrt(fmax(Aabs, 1e-30));
    }
    minheight[flat] = mh;
}

// ============================================================
// EOS + artificial viscosity (von Neumann-Richtmyer, Matterflow style)
//   ρ     = dm / V
//   e_int = cell energy / dm   (stored as specific already)
//   P     = (γ−1) · ρ · e_int
//   c_s   = √(γ · P / ρ)
//
//   strain_rate = -(1/V) · dV/dt  (positive under compression)
//   Use dV/dt = Σ_edges (v_mid · A_edge_outward)
//
//   Q = ρ · (CQ_quad · (s·L)² + CQ_lin · c_s · s·L)  when s > 0
//     where L = √Area0  (constant reference)
// ============================================================
__global__
void k_ale_eos_and_q(const double* R, const double* Z,
                     const double* vR, const double* vZ,
                     const double* dm, const double* Vol, const double* Area0,
                     const double* e_int,
                     double* rho, double* P, double* Q, double* cs,
                     double* strain_rate,
                     int nr, int nt, double gam,
                     double CQ_lin, double CQ_quad) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int ic = flat / nt, jc = flat % nt;
    int nnt = nt + 1;

    double V = fmax(Vol[flat], 1e-30);
    double r_ = dm[flat] / V;
    rho[flat] = r_;

    double e = e_int[flat];
    double p = fmax((gam - 1.0) * r_ * e, 1e-30);
    P[flat] = p;
    cs[flat] = sqrt(gam * p / r_);

    // dV/dt  from nodal velocities via Pappus-flux
    int i0 = ale_n(ic,   jc,   nnt);
    int i1 = ale_n(ic+1, jc,   nnt);
    int i2 = ale_n(ic+1, jc+1, nnt);
    int i3 = ale_n(ic,   jc+1, nnt);
    double R0=R[i0], R1=R[i1], R2=R[i2], R3=R[i3];
    double Z0=Z[i0], Z1=Z[i1], Z2=Z[i2], Z3=Z[i3];
    double vR0=vR[i0], vR1=vR[i1], vR2=vR[i2], vR3=vR[i3];
    double vZ0=vZ[i0], vZ1=vZ[i1], vZ2=vZ[i2], vZ3=vZ[i3];

    // Edge contribution to dV/dt:
    //   dV/dt_edge ≈ 2π R_mid · (v_mid · n_outward · |edge|)
    //              = 2π R_mid · (v_mid_R · dZ − v_mid_Z · dR)
    auto edge_dVdt = [](double Ra, double Za, double Rb, double Zb,
                        double vRa, double vZa, double vRb, double vZb) -> double {
        double Rm = 0.5*(Ra + Rb), dR = Rb - Ra, dZ = Zb - Za;
        double vRm = 0.5*(vRa + vRb), vZm = 0.5*(vZa + vZb);
        return 2.0 * M_PI * Rm * (vRm * dZ - vZm * dR);
    };
    double dVdt = edge_dVdt(R0, Z0, R1, Z1, vR0, vZ0, vR1, vZ1)
                + edge_dVdt(R1, Z1, R2, Z2, vR1, vZ1, vR2, vZ2)
                + edge_dVdt(R2, Z2, R3, Z3, vR2, vZ2, vR3, vZ3)
                + edge_dVdt(R3, Z3, R0, Z0, vR3, vZ3, vR0, vZ0);
    double s = -dVdt / V;  // strain rate (+ in compression)
    strain_rate[flat] = s;

    double L = sqrt(fmax(Area0[flat], 1e-30));
    double q = 0.0;
    if (s > 0.0) {
        double q_quad = CQ_quad * s * L * s * L;         // ∝ (sL)²
        double q_lin  = CQ_lin  * cs[flat] * s * L;       // ∝ cs·sL
        q = r_ * (q_quad + q_lin);
    }
    Q[flat] = q;
}

// ============================================================
// Accumulate node forces and per-subcell force vectors.
// Each cell contributes 4 edge forces; each edge puts half on each end node.
// Subcell force per corner = sum of two half-edges touching that corner.
// ============================================================
__global__
void k_ale_node_forces(const double* R, const double* Z,
                       const double* P, const double* Q,
                       double* FR, double* FZ,
                       double* FSR, double* FSZ,  // per-subcell (ncell*4)
                       int nr, int nt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int ic = flat / nt, jc = flat % nt;
    int nnt = nt + 1;
    int I[4] = { ale_n(ic,   jc,   nnt),
                 ale_n(ic+1, jc,   nnt),
                 ale_n(ic+1, jc+1, nnt),
                 ale_n(ic,   jc+1, nnt) };
    double Rk[4] = {R[I[0]], R[I[1]], R[I[2]], R[I[3]]};
    double Zk[4] = {Z[I[0]], Z[I[1]], Z[I[2]], Z[I[3]]};
    double PQ = P[flat] + Q[flat];

    // Per-edge area vector (dZ, -dR) · 2πR_mid
    double aR[4], aZ[4];  // edge k: from corner k to corner (k+1)%4
    for (int k = 0; k < 4; ++k) {
        int kp = (k + 1) & 3;
        double Rm = 0.5 * (Rk[k] + Rk[kp]);
        double dR = Rk[kp] - Rk[k];
        double dZ = Zk[kp] - Zk[k];
        aR[k] =  PQ * 2.0 * M_PI * Rm * dZ;
        aZ[k] = -PQ * 2.0 * M_PI * Rm * dR;
    }
    // Half of each edge's force goes to each of its two end corners.
    // Subcell force at corner k = ½·(edge(k-1→k) + edge(k→k+1))
    for (int k = 0; k < 4; ++k) {
        int km = (k + 3) & 3;
        double sx = 0.5 * (aR[km] + aR[k]);
        double sz = 0.5 * (aZ[km] + aZ[k]);
        FSR[flat * 4 + k] = sx;
        FSZ[flat * 4 + k] = sz;
        // Atomic add to node force
        atomicAdd(&FR[I[k]], sx);
        atomicAdd(&FZ[I[k]], sz);
    }
}

__global__
void k_ale_zero_nodes(double* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 0.0;
}

// ============================================================
// Add gravity on nodes: F_g = -G M_enc(node_r)/node_r² · r_hat
// Using shell-based M_enc: we map node to its radial index (in from 0..nr)
// and interpolate r from the node's (R,Z).
// For A1 we use: g_mag = G · M_enc[in] / r_node²; direction = -(R,Z)/r.
// ============================================================
__global__
void k_ale_add_gravity(const double* R, const double* Z,
                       const double* mnode,
                       const double* M_enc,  // size nnode_r (= nr+1)
                       double* FR, double* FZ,
                       int nnode_r, int nnode_t, double G_const) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int nnode = nnode_r * nnode_t;
    if (flat >= nnode) return;
    int in = flat / nnode_t;
    double Rn = R[flat], Zn = Z[flat];
    double r = sqrt(Rn*Rn + Zn*Zn);
    if (r < 1e-14) return;  // origin
    double Menc = M_enc[in];
    double g_mag = -G_const * Menc / (r * r);
    FR[flat] += mnode[flat] * g_mag * (Rn / r);
    FZ[flat] += mnode[flat] * g_mag * (Zn / r);
}

// ============================================================
// Kick-drift-kick: v_{n+1/2} = v_n + dt/2 · F/m,
//                  pos_{n+1} = pos_n + dt · v_{n+1/2},
//                  v_{n+1}   = v_{n+1/2} + dt/2 · F_{n+1}/m
//   We store v at half-step between force evaluations.
// For this A1 we use the simpler Matterflow update:
//   Δv = F/m · dt
//   Δpos = (v + 0.5·Δv) · dt     [midpoint drift]
//   pos += Δpos
//   v   += Δv
// which is equivalent and gives a natural "Δpos per node" that we feed
// into the Caramana energy update in the same step.
// ============================================================
__global__
void k_ale_node_update(double* R, double* Z,
                       double* vR, double* vZ,
                       const double* FR, const double* FZ,
                       const double* mnode,
                       double* dR_out, double* dZ_out,  // displacement (for energy)
                       double dt, int nnode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnode) return;
    double mm = fmax(mnode[i], 1e-30);
    double dvR = FR[i] / mm * dt;
    double dvZ = FZ[i] / mm * dt;
    double vRh = vR[i] + 0.5 * dvR;
    double vZh = vZ[i] + 0.5 * dvZ;
    double dR = vRh * dt;
    double dZ = vZh * dt;
    R[i] += dR;
    Z[i] += dZ;
    vR[i] += dvR;
    vZ[i] += dvZ;
    dR_out[i] = dR;
    dZ_out[i] = dZ;
}

// ============================================================
// Caramana compatible energy update:
//   e_int_cell -= Σ_corners (F_subcell · dPos_node) / dm_cell
// ============================================================
__global__
void k_ale_energy_update(const int nr, const int nt,
                         const double* FSR, const double* FSZ,
                         const double* dR_node, const double* dZ_node,
                         const double* dm,
                         double* e_int) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int ic = flat / nt, jc = flat % nt;
    int nnt = nt + 1;
    int I[4] = { ale_n(ic,   jc,   nnt),
                 ale_n(ic+1, jc,   nnt),
                 ale_n(ic+1, jc+1, nnt),
                 ale_n(ic,   jc+1, nnt) };
    double W = 0.0;
    for (int k = 0; k < 4; ++k) {
        W += FSR[flat*4 + k] * dR_node[I[k]] + FSZ[flat*4 + k] * dZ_node[I[k]];
    }
    double m = fmax(dm[flat], 1e-30);
    e_int[flat] -= W / m;
}

// ============================================================
// Boundary conditions:
//   (a) Axis (jn == 0 or jn == nt):   R = 0, vR = 0, FR = 0
//   (b) Origin (R² + Z² == 0):        pinned, everything 0
//   (c) Outer surface (in == nr):     free — nothing to do (P_surf floor in force kernel TBD)
// For now origin pinning handled by noting r_face[0] = 0 puts all in=0 nodes at (0,0).
// ============================================================
__global__
void k_ale_bc_axis(double* R, double* vR, double* FR,
                   int nnode_r, int nnode_t) {
    int in = blockIdx.x * blockDim.x + threadIdx.x;
    if (in >= nnode_r) return;
    // jn = 0 (north pole) and jn = nt (south pole) lie on the symmetry axis
    int n_north = in * nnode_t + 0;
    int n_south = in * nnode_t + (nnode_t - 1);
    R[n_north] = 0.0;   vR[n_north] = 0.0;   FR[n_north] = 0.0;
    R[n_south] = 0.0;   vR[n_south] = 0.0;   FR[n_south] = 0.0;
}

__global__
void k_ale_bc_origin(double* R, double* Z, double* vR, double* vZ,
                     double* FR, double* FZ, int nnode_t) {
    int jn = blockIdx.x * blockDim.x + threadIdx.x;
    if (jn >= nnode_t) return;
    // in = 0 nodes are at the origin (r=0). Pin them.
    int k = jn;
    R[k] = 0.0; Z[k] = 0.0;
    vR[k] = 0.0; vZ[k] = 0.0;
    FR[k] = 0.0; FZ[k] = 0.0;
}

// ============================================================
// CFL: dt_sound = cfl · minheight / (cs + |v|)
//      dt_visc  = cfl · L² / (2 · Q/ρ / max(strain,eps))   -- bounded by quadratic term
//      dt_comp  = cfl · comp_dt_frac / strain_rate (if positive)
// ============================================================
__global__
void k_ale_cfl(const double* minheight, const double* cs,
               const double* strain_rate, const double* Area0,
               const double* vR, const double* vZ,
               int nr, int nt, double cfl, double comp_frac,
               double* dt_cell) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    int ic = flat / nt, jc = flat % nt;
    int nnt = nt + 1;
    int I[4] = { ale_n(ic,   jc,   nnt),
                 ale_n(ic+1, jc,   nnt),
                 ale_n(ic+1, jc+1, nnt),
                 ale_n(ic,   jc+1, nnt) };
    double vmax = 0.0;
    for (int k = 0; k < 4; ++k) {
        double vv = sqrt(vR[I[k]]*vR[I[k]] + vZ[I[k]]*vZ[I[k]]);
        vmax = fmax(vmax, vv);
    }
    double L = minheight[flat];
    double c = cs[flat];
    double dt_s = L / (c + vmax + 1e-30);
    double s = strain_rate[flat];
    double dt_c = (s > 0.0) ? (comp_frac / s) : 1e30;
    double dt = fmin(dt_s, dt_c);
    dt_cell[flat] = cfl * dt;
}

// ============================================================
// Apply P_floor / rho floor — unused in A1 (we trust compatible hydro)
// ============================================================
__global__
void k_ale_floor(double* e_int, const double* e0,
                 double P_floor_frac, int nr, int nt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nr*nt) return;
    if (e_int[flat] < P_floor_frac * e0[flat])
        e_int[flat] = P_floor_frac * e0[flat];
}

// ============================================================
// Shell mass: Σ_j dm[i, j]  — a per-radial-shell total, invariant after init.
// ============================================================
__global__
void k_ale_shell_mass(const double* dm, double* shell_mass,
                      int nr, int nt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nr) return;
    double s = 0.0;
    for (int j = 0; j < nt; ++j) s += dm[i*nt + j];
    shell_mass[i] = s;
}

// ============================================================
// Enclosed mass: prefix sum, stored at node-radial index in = 0..nr
//   M_enc[0] = 0;  M_enc[in] = Σ_{i<in} shell_mass[i]  (assuming node sits on shell face)
// For simplicity this is single-thread.
// ============================================================
__global__
void k_ale_enclosed_mass(const double* shell_mass, double* M_enc,
                         int nr) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    M_enc[0] = 0.0;
    double s = 0.0;
    for (int i = 0; i < nr; ++i) {
        s += shell_mass[i];
        M_enc[i + 1] = s;
    }
}

// ============================================================
// Helpers exposed to host via cudaMemcpy
// ============================================================
__global__
void k_ale_init_from_rth(const double* r_face, const double* theta_face,
                         double* R, double* Z,
                         int nnode_r, int nnode_t) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nnode_r * nnode_t;
    if (flat >= n) return;
    int in = flat / nnode_t, jn = flat % nnode_t;
    double rr = r_face[in];
    double th = theta_face[jn];
    R[flat] = rr * sin(th);
    Z[flat] = rr * cos(th);
}

// Compute node mass from 1/4 of each of the (up to 4) adjacent cell masses.
__global__
void k_ale_node_mass(const double* dm, double* mnode,
                     int nr, int nt) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int nnt = nt + 1, nnr = nr + 1;
    int n = nnr * nnt;
    if (flat >= n) return;
    int in = flat / nnt, jn = flat % nnt;
    double m = 0.0;
    // Up to 4 adjacent cells: (in-1, jn-1), (in-1, jn), (in, jn-1), (in, jn)
    for (int di = -1; di <= 0; ++di) {
        for (int dj = -1; dj <= 0; ++dj) {
            int ic = in + di, jc = jn + dj;
            if (ic < 0 || ic >= nr || jc < 0 || jc >= nt) continue;
            m += 0.25 * dm[ic*nt + jc];
        }
    }
    mnode[flat] = m;
}
