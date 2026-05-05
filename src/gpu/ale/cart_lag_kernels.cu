// Cartesian 2D Lagrangian kernels (Caramana compatible, quad mesh).
//
// Conventions:
//   cell (ic, jc)  ∈ [0, nx) × [0, ny)
//   node (in, jn)  ∈ [0, nx] × [0, ny]
//   cell corners (CCW, bottom-left start):
//     0: (ic,   jc)      SW
//     1: (ic+1, jc)      SE
//     2: (ic+1, jc+1)    NE
//     3: (ic,   jc+1)    NW
//
// Geometry (planar, no revolution):
//   Area = ½ |Σ (x_k · y_{k+1} − x_{k+1} · y_k)|   [signed shoelace]
//   Edge area-vector (outward normal × length) = (dy, −dx)
//     from corner k to corner (k+1), with  dx = x_{k+1}−x_k, dy = y_{k+1}−y_k
//
// Force per edge:
//   F_edge = (P + Q) · (dy, −dx)
// Each edge contributes half its force to each of its two end corners.
// Subcell force = ½ (F_edge(k-1→k) + F_edge(k→k+1)).

#include "cart_lag_solver.cuh"
#include "gpu_common.cuh"  // CUDA_CHECK
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ __forceinline__ int cln(int in, int jn, int nny) { return in * nny + jn; }
__device__ __forceinline__ int clc(int ic, int jc, int ny)  { return ic * ny  + jc; }

// ============================================================
// Geometry: area, min-height
// ============================================================
__global__
void k_clag_geometry(const double* X, const double* Y,
                     double* Vol, double* minheight,
                     int nx, int ny) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nx*ny) return;
    int ic = flat / ny, jc = flat % ny;
    int nny = ny + 1;
    int i0 = cln(ic,   jc,   nny);
    int i1 = cln(ic+1, jc,   nny);
    int i2 = cln(ic+1, jc+1, nny);
    int i3 = cln(ic,   jc+1, nny);
    double X0=X[i0], X1=X[i1], X2=X[i2], X3=X[i3];
    double Y0=Y[i0], Y1=Y[i1], Y2=Y[i2], Y3=Y[i3];

    // Eq. (15.2): shoelace formula for quadrilateral signed area.
    double A2 = 0.5 * ((X0*Y1 - X1*Y0) + (X1*Y2 - X2*Y1)
                     + (X2*Y3 - X3*Y2) + (X3*Y0 - X0*Y3));
    Vol[flat] = fabs(A2);

    // Min perpendicular distance from a corner to the opposite diagonal-spanning edge
    auto perp = [](double Px, double Py, double Ax, double Ay, double Bx, double By) {
        double dx = Bx - Ax, dy = By - Ay;
        double len2 = dx*dx + dy*dy;
        if (len2 < 1e-30) return sqrt((Px-Ax)*(Px-Ax) + (Py-Ay)*(Py-Ay));
        double cross = (Bx - Ax)*(Py - Ay) - (By - Ay)*(Px - Ax);
        return fabs(cross) / sqrt(len2);
    };
    // Distance from each corner to its opposite edge
    double h0 = perp(X0, Y0, X1, Y1, X2, Y2);
    double h1 = perp(X1, Y1, X2, Y2, X3, Y3);
    double h2 = perp(X2, Y2, X3, Y3, X0, Y0);
    double h3 = perp(X3, Y3, X0, Y0, X1, Y1);
    double mh = fmin(fmin(h0, h1), fmin(h2, h3));
    minheight[flat] = fmax(mh, 1e-12);
}

// ============================================================
// EOS + artificial viscosity (Matterflow-style von Neumann-Richtmyer)
// ============================================================
__global__
void k_clag_eos_and_q(const double* X, const double* Y,
                      const double* vX, const double* vY,
                      const double* dm, const double* Vol, const double* Area0,
                      const double* e_int,
                      double* rho, double* P, double* Q, double* cs,
                      double* strain_rate,
                      int nx, int ny, double gam,
                      double CQ_lin, double CQ_quad) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nx*ny) return;
    int ic = flat / ny, jc = flat % ny;
    int nny = ny + 1;

    double V = fmax(Vol[flat], 1e-30);
    double r_ = dm[flat] / V;
    rho[flat] = r_;
    double e = e_int[flat];
    double p = fmax((gam - 1.0) * r_ * e, 1e-30);   // Eq. (1.2)
    P[flat] = p;
    cs[flat] = sqrt(gam * p / r_);                   // Eq. (1.3)

    int i0 = cln(ic,   jc,   nny);
    int i1 = cln(ic+1, jc,   nny);
    int i2 = cln(ic+1, jc+1, nny);
    int i3 = cln(ic,   jc+1, nny);
    double X0=X[i0], X1=X[i1], X2=X[i2], X3=X[i3];
    double Y0=Y[i0], Y1=Y[i1], Y2=Y[i2], Y3=Y[i3];
    double vX0=vX[i0], vX1=vX[i1], vX2=vX[i2], vX3=vX[i3];
    double vY0=vY[i0], vY1=vY[i1], vY2=vY[i2], vY3=vY[i3];

    // dV/dt from nodal velocities: ∮ (v · n) dl
    //   edge (k→k+1): contrib = v_mid·(dy, -dx) = v_mid_x·dy − v_mid_y·dx
    auto edge_dAdt = [](double Xa, double Ya, double Xb, double Yb,
                        double vXa, double vYa, double vXb, double vYb) {
        double dx = Xb - Xa, dy = Yb - Ya;
        double vXm = 0.5*(vXa + vXb), vYm = 0.5*(vYa + vYb);
        return vXm * dy - vYm * dx;
    };
    // Eq. (15.3): divergence-consistent strain rate = -∮(v·n)dl / V.
    double dAdt = edge_dAdt(X0,Y0,X1,Y1,vX0,vY0,vX1,vY1)
                + edge_dAdt(X1,Y1,X2,Y2,vX1,vY1,vX2,vY2)
                + edge_dAdt(X2,Y2,X3,Y3,vX2,vY2,vX3,vY3)
                + edge_dAdt(X3,Y3,X0,Y0,vX3,vY3,vX0,vY0);
    double s = -dAdt / V;
    strain_rate[flat] = s;

    // Eq. (15.4): compression-only von Neumann-Richtmyer AV (scalar, no
    // shear weighting — that's a cart_ale2 extension).
    double L = sqrt(fmax(Area0[flat], 1e-30));
    double q = 0.0;
    if (s > 0.0) {
        double q_quad = CQ_quad * s * L * s * L;
        double q_lin  = CQ_lin  * cs[flat] * s * L;
        q = r_ * (q_quad + q_lin);
    }
    Q[flat] = q;
}

// ============================================================
// Accumulate node + subcell forces
// ============================================================
__global__
void k_clag_node_forces(const double* X, const double* Y,
                        const double* P, const double* Q,
                        double* FX, double* FY,
                        double* FSX, double* FSY,
                        int nx, int ny) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nx*ny) return;
    int ic = flat / ny, jc = flat % ny;
    int nny = ny + 1;
    int I[4] = { cln(ic,   jc,   nny),
                 cln(ic+1, jc,   nny),
                 cln(ic+1, jc+1, nny),
                 cln(ic,   jc+1, nny) };
    double Xk[4] = {X[I[0]], X[I[1]], X[I[2]], X[I[3]]};
    double Yk[4] = {Y[I[0]], Y[I[1]], Y[I[2]], Y[I[3]]};
    double PQ = P[flat] + Q[flat];

    // Eq. (15.5): edge-normal integrated force (P+Q)·(Δy, −Δx) per edge.
    double aX[4], aY[4];
    for (int k = 0; k < 4; ++k) {
        int kp = (k + 1) & 3;
        double dx = Xk[kp] - Xk[k];
        double dy = Yk[kp] - Yk[k];
        aX[k] =  PQ * dy;
        aY[k] = -PQ * dx;
    }
    // Eq. (15.6): corner subcell force = ½(edge_{k-1} + edge_k).
    for (int k = 0; k < 4; ++k) {
        int km = (k + 3) & 3;
        double sx = 0.5 * (aX[km] + aX[k]);
        double sy = 0.5 * (aY[km] + aY[k]);
        FSX[flat * 4 + k] = sx;
        FSY[flat * 4 + k] = sy;
        atomicAdd(&FX[I[k]], sx);
        atomicAdd(&FY[I[k]], sy);
    }
}

__global__
void k_clag_zero(double* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 0.0;
}

// ============================================================
// Add constant downward gravity to node forces: F_y += -g · m_node
// ============================================================
__global__
void k_clag_add_gravity(const double* mnode, double* FY,
                        double g_val, int nnode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnode) return;
    FY[i] += -g_val * mnode[i];
}

// ============================================================
// BCs: reflective walls on all 4 box sides.
//   vX = 0 on left/right edges, vY = 0 on top/bottom edges.
//   Correspondingly, FX on left/right = 0, FY on top/bottom = 0.
// Also pin node positions on those edges.
// ============================================================
__global__
void k_clag_bc_reflective(double* X, double* Y,
                          double* vX, double* vY,
                          double* FX, double* FY,
                          double Lx, double Ly,
                          int nnx, int nny) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nnx * nny;
    if (flat >= n) return;
    int in = flat / nny, jn = flat % nny;
    if (in == 0) {
        X[flat] = 0.0;  vX[flat] = 0.0;  FX[flat] = 0.0;
    }
    if (in == nnx - 1) {
        X[flat] = Lx;   vX[flat] = 0.0;  FX[flat] = 0.0;
    }
    if (jn == 0) {
        Y[flat] = 0.0;  vY[flat] = 0.0;  FY[flat] = 0.0;
    }
    if (jn == nny - 1) {
        Y[flat] = Ly;   vY[flat] = 0.0;  FY[flat] = 0.0;
    }
}

// ============================================================
// Kick-drift-kick node update (Matterflow formulation):
//   Δv = F/m · dt
//   ΔPos = (v + ½·Δv) · dt
//   Pos  += ΔPos
//   v    += Δv
// Compatible energy update later consumes ΔPos.
// ============================================================
__global__
void k_clag_node_update(double* X, double* Y,
                        double* vX, double* vY,
                        const double* FX, const double* FY,
                        const double* mnode,
                        double* dX_out, double* dY_out,
                        double dt, int nnode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnode) return;
    // Eq. (15.7): kick-drift-kick node update.
    double mm = fmax(mnode[i], 1e-30);
    double dvX = FX[i] / mm * dt;
    double dvY = FY[i] / mm * dt;
    double vXh = vX[i] + 0.5 * dvX;
    double vYh = vY[i] + 0.5 * dvY;
    double dX = vXh * dt;
    double dY = vYh * dt;
    X[i]  += dX;
    Y[i]  += dY;
    vX[i] += dvX;
    vY[i] += dvY;
    dX_out[i] = dX;
    dY_out[i] = dY;
}

// ============================================================
// Caramana compatible energy update:
//   e_int_cell -= Σ_corners (F_subcell · ΔPos_node) / dm_cell
// ============================================================
__global__
void k_clag_energy_update(const int nx, const int ny,
                          const double* FSX, const double* FSY,
                          const double* dX_node, const double* dY_node,
                          const double* dm,
                          double* e_int) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nx*ny) return;
    int ic = flat / ny, jc = flat % ny;
    int nny = ny + 1;
    int I[4] = { cln(ic,   jc,   nny),
                 cln(ic+1, jc,   nny),
                 cln(ic+1, jc+1, nny),
                 cln(ic,   jc+1, nny) };
    // Eq. (15.8): compatible energy update — Δe · m = −Σ F_sub · ΔX_node.
    double W = 0.0;
    for (int k = 0; k < 4; ++k) {
        W += FSX[flat*4 + k] * dX_node[I[k]] + FSY[flat*4 + k] * dY_node[I[k]];
    }
    double m = fmax(dm[flat], 1e-30);
    e_int[flat] -= W / m;
}

// ============================================================
// CFL: dt = cfl · min( L / (cs + |v|),  comp_frac / strain_rate )
// ============================================================
__global__
void k_clag_cfl(const double* minheight, const double* cs,
                const double* strain_rate,
                const double* vX, const double* vY,
                int nx, int ny, double cfl, double comp_frac,
                double* dt_cell) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nx*ny) return;
    int ic = flat / ny, jc = flat % ny;
    int nny = ny + 1;
    int I[4] = { cln(ic,   jc,   nny),
                 cln(ic+1, jc,   nny),
                 cln(ic+1, jc+1, nny),
                 cln(ic,   jc+1, nny) };
    double vmax = 0.0;
    for (int k = 0; k < 4; ++k) {
        double vv = sqrt(vX[I[k]]*vX[I[k]] + vY[I[k]]*vY[I[k]]);
        vmax = fmax(vmax, vv);
    }
    // Eq. (15.9): minimum-height CFL.
    double L = minheight[flat];
    double dt_s = L / (cs[flat] + vmax + 1e-30);
    double s = strain_rate[flat];
    double dt_c = (s > 0.0) ? (comp_frac / s) : 1e30;
    dt_cell[flat] = cfl * fmin(dt_s, dt_c);
}

// ============================================================
// Init helpers: uniform node layout; node mass = ¼·Σ adjacent cell dm
// ============================================================
__global__
void k_clag_init_nodes(double* X, double* Y,
                       double Lx, double Ly, int nnx, int nny) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nnx * nny;
    if (flat >= n) return;
    int in = flat / nny, jn = flat % nny;
    X[flat] = Lx * (double)in / (double)(nnx - 1);
    Y[flat] = Ly * (double)jn / (double)(nny - 1);
}

__global__
void k_clag_node_mass(const double* dm, double* mnode,
                      int nx, int ny) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int nnx = nx + 1, nny = ny + 1;
    int n = nnx * nny;
    if (flat >= n) return;
    int in = flat / nny, jn = flat % nny;
    // Eq. (15.1): node mass = ¼ Σ adjacent cell masses (no periodic wrap here —
    // cart_lag is reflective only; see cart_ale2 for periodic version).
    double m = 0.0;
    for (int di = -1; di <= 0; ++di) {
        for (int dj = -1; dj <= 0; ++dj) {
            int ic = in + di, jc = jn + dj;
            if (ic < 0 || ic >= nx || jc < 0 || jc >= ny) continue;
            m += 0.25 * dm[ic*ny + jc];
        }
    }
    mnode[flat] = m;
}
