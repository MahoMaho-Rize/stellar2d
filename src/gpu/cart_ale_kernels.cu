// Cartesian 2D ALE kernels.
//
// Lagrangian phase is identical in form to cart_lag; the new work here
// is the rezone+remap phase (swept-edge donor-cell flux).
//
// Conventions:
//   cell (ic, jc)  ∈ [0, nx) × [0, ny)   → flat = ic*ny + jc
//   node (in, jn)  ∈ [0, nx] × [0, ny]   → flat = in*(ny+1) + jn
//   corners CCW: 0=SW, 1=SE, 2=NE, 3=NW
//
// Edges of a cell (identified by cell + local edge index e ∈ [0..3]):
//   e=0 south (n0→n1),   e=1 east  (n1→n2),
//   e=2 north (n2→n3),   e=3 west  (n3→n0).
// An interior east edge of cell (ic,jc) is shared with the west edge
// of cell (ic+1, jc). Likewise north↔south. The swept flux through
// an edge is signed by its donor side.
//
// Conserved quantities carried on cells: dm, dm·e_int, px, py.
// (Node velocities are converted to cell-centered momentum before
// remap and redistributed back afterwards.)

#include "cart_ale_solver.cuh"
#include "fas_common.cuh"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ __forceinline__ int caln(int in, int jn, int nny) { return in * nny + jn; }
__device__ __forceinline__ int calc(int ic, int jc, int ny)  { return ic * ny  + jc; }

// ============================================================
// Geometry
// ============================================================
__global__
void k_cale_geometry(const double* X, const double* Y,
                     double* Vol, double* minheight,
                     int nx, int ny) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nx*ny) return;
    int ic = flat / ny, jc = flat % ny;
    int nny = ny + 1;
    int i0 = caln(ic,   jc,   nny);
    int i1 = caln(ic+1, jc,   nny);
    int i2 = caln(ic+1, jc+1, nny);
    int i3 = caln(ic,   jc+1, nny);
    double X0=X[i0], X1=X[i1], X2=X[i2], X3=X[i3];
    double Y0=Y[i0], Y1=Y[i1], Y2=Y[i2], Y3=Y[i3];
    double A2 = 0.5 * ((X0*Y1 - X1*Y0) + (X1*Y2 - X2*Y1)
                     + (X2*Y3 - X3*Y2) + (X3*Y0 - X0*Y3));
    Vol[flat] = fabs(A2);

    auto perp = [](double Px, double Py, double Ax, double Ay, double Bx, double By) {
        double dx = Bx - Ax, dy = By - Ay;
        double len2 = dx*dx + dy*dy;
        if (len2 < 1e-30) return sqrt((Px-Ax)*(Px-Ax) + (Py-Ay)*(Py-Ay));
        double cross = (Bx - Ax)*(Py - Ay) - (By - Ay)*(Px - Ax);
        return fabs(cross) / sqrt(len2);
    };
    double h0 = perp(X0, Y0, X1, Y1, X2, Y2);
    double h1 = perp(X1, Y1, X2, Y2, X3, Y3);
    double h2 = perp(X2, Y2, X3, Y3, X0, Y0);
    double h3 = perp(X3, Y3, X0, Y0, X1, Y1);
    double mh = fmin(fmin(h0, h1), fmin(h2, h3));
    minheight[flat] = fmax(mh, 1e-12);
}

// ============================================================
// EOS + artificial viscosity (von Neumann-Richtmyer, compression only)
// ============================================================
__global__
void k_cale_eos_and_q(const double* X, const double* Y,
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
    double p = fmax((gam - 1.0) * r_ * e, 1e-30);
    P[flat] = p;
    cs[flat] = sqrt(gam * p / r_);

    int i0 = caln(ic,   jc,   nny);
    int i1 = caln(ic+1, jc,   nny);
    int i2 = caln(ic+1, jc+1, nny);
    int i3 = caln(ic,   jc+1, nny);
    double X0=X[i0], X1=X[i1], X2=X[i2], X3=X[i3];
    double Y0=Y[i0], Y1=Y[i1], Y2=Y[i2], Y3=Y[i3];
    double vX0=vX[i0], vX1=vX[i1], vX2=vX[i2], vX3=vX[i3];
    double vY0=vY[i0], vY1=vY[i1], vY2=vY[i2], vY3=vY[i3];

    auto edge_dAdt = [](double Xa, double Ya, double Xb, double Yb,
                        double vXa, double vYa, double vXb, double vYb) {
        double dx = Xb - Xa, dy = Yb - Ya;
        double vXm = 0.5*(vXa + vXb), vYm = 0.5*(vYa + vYb);
        return vXm * dy - vYm * dx;
    };
    double dAdt = edge_dAdt(X0,Y0,X1,Y1,vX0,vY0,vX1,vY1)
                + edge_dAdt(X1,Y1,X2,Y2,vX1,vY1,vX2,vY2)
                + edge_dAdt(X2,Y2,X3,Y3,vX2,vY2,vX3,vY3)
                + edge_dAdt(X3,Y3,X0,Y0,vX3,vY3,vX0,vY0);
    double s = -dAdt / V;
    strain_rate[flat] = s;

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
// Node + subcell forces
// ============================================================
__global__
void k_cale_node_forces(const double* X, const double* Y,
                        const double* P, const double* Q,
                        double* FX, double* FY,
                        double* FSX, double* FSY,
                        int nx, int ny) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nx*ny) return;
    int ic = flat / ny, jc = flat % ny;
    int nny = ny + 1;
    int I[4] = { caln(ic,   jc,   nny),
                 caln(ic+1, jc,   nny),
                 caln(ic+1, jc+1, nny),
                 caln(ic,   jc+1, nny) };
    double Xk[4] = {X[I[0]], X[I[1]], X[I[2]], X[I[3]]};
    double Yk[4] = {Y[I[0]], Y[I[1]], Y[I[2]], Y[I[3]]};
    double PQ = P[flat] + Q[flat];

    double aX[4], aY[4];
    for (int k = 0; k < 4; ++k) {
        int kp = (k + 1) & 3;
        double dx = Xk[kp] - Xk[k];
        double dy = Yk[kp] - Yk[k];
        aX[k] =  PQ * dy;
        aY[k] = -PQ * dx;
    }
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
void k_cale_zero(double* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 0.0;
}

__global__
void k_cale_add_gravity(const double* mnode, double* FY,
                        double g_val, int nnode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnode) return;
    FY[i] += -g_val * mnode[i];
}

// ============================================================
// Reflective walls: pin positions at initial (X0, Y0) on boundary,
// and zero the wall-normal force/velocity components each step.
// Because the rezone phase will snap nodes back to (X0,Y0) anyway,
// pinning is almost redundant — but it keeps the Lagrangian
// substep honest.
// ============================================================
__global__
void k_cale_bc_reflective(const double* X0, const double* Y0,
                          double* X, double* Y,
                          double* vX, double* vY,
                          double* FX, double* FY,
                          int nnx, int nny) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nnx * nny;
    if (flat >= n) return;
    int in = flat / nny, jn = flat % nny;
    if (in == 0 || in == nnx - 1) {
        X[flat] = X0[flat];
        vX[flat] = 0.0;
        FX[flat] = 0.0;
    }
    if (jn == 0 || jn == nny - 1) {
        Y[flat] = Y0[flat];
        vY[flat] = 0.0;
        FY[flat] = 0.0;
    }
}

// ============================================================
// Kick-drift-kick node update (same as cart_lag)
// ============================================================
__global__
void k_cale_node_update(double* X, double* Y,
                        double* vX, double* vY,
                        const double* FX, const double* FY,
                        const double* mnode,
                        double* dX_out, double* dY_out,
                        double dt, int nnode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnode) return;
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
// Compatible energy update
// ============================================================
__global__
void k_cale_energy_update(int nx, int ny,
                          const double* FSX, const double* FSY,
                          const double* dX_node, const double* dY_node,
                          const double* dm,
                          double* e_int) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nx*ny) return;
    int ic = flat / ny, jc = flat % ny;
    int nny = ny + 1;
    int I[4] = { caln(ic,   jc,   nny),
                 caln(ic+1, jc,   nny),
                 caln(ic+1, jc+1, nny),
                 caln(ic,   jc+1, nny) };
    double W = 0.0;
    for (int k = 0; k < 4; ++k) {
        W += FSX[flat*4 + k] * dX_node[I[k]] + FSY[flat*4 + k] * dY_node[I[k]];
    }
    double m = fmax(dm[flat], 1e-30);
    e_int[flat] -= W / m;
}

// ============================================================
// CFL
// ============================================================
__global__
void k_cale_cfl(const double* minheight, const double* cs,
                const double* strain_rate,
                const double* vX, const double* vY,
                int nx, int ny, double cfl, double comp_frac,
                double* dt_cell) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nx*ny) return;
    int ic = flat / ny, jc = flat % ny;
    int nny = ny + 1;
    int I[4] = { caln(ic,   jc,   nny),
                 caln(ic+1, jc,   nny),
                 caln(ic+1, jc+1, nny),
                 caln(ic,   jc+1, nny) };
    double vmax = 0.0;
    for (int k = 0; k < 4; ++k) {
        double vv = sqrt(vX[I[k]]*vX[I[k]] + vY[I[k]]*vY[I[k]]);
        vmax = fmax(vmax, vv);
    }
    double L = minheight[flat];
    double dt_s = L / (cs[flat] + vmax + 1e-30);
    double s = strain_rate[flat];
    double dt_c = (s > 0.0) ? (comp_frac / s) : 1e30;
    dt_cell[flat] = cfl * fmin(dt_s, dt_c);
}

// ============================================================
// Node init: uniform lattice → stored in both X and X0
// ============================================================
__global__
void k_cale_init_nodes(double* X, double* Y,
                       double* X0, double* Y0,
                       double Lx, double Ly, int nnx, int nny) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nnx * nny;
    if (flat >= n) return;
    int in = flat / nny, jn = flat % nny;
    double x = Lx * (double)in / (double)(nnx - 1);
    double y = Ly * (double)jn / (double)(nny - 1);
    X[flat] = X0[flat] = x;
    Y[flat] = Y0[flat] = y;
}

// ============================================================
// Reset mesh: X ← X0, Y ← Y0
// ============================================================
__global__
void k_cale_reset_mesh(const double* X0, const double* Y0,
                       double* X, double* Y, int nnode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnode) return;
    X[i] = X0[i];
    Y[i] = Y0[i];
}

// ============================================================
// Node mass
// ============================================================
__global__
void k_cale_node_mass(const double* dm, double* mnode,
                      int nx, int ny) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int nnx = nx + 1, nny = ny + 1;
    int n = nnx * nny;
    if (flat >= n) return;
    int in = flat / nny, jn = flat % nny;
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

// ============================================================
// Cell-centered momentum from node velocities.
// Each cell gets the mass-weighted velocity contribution from its
// 4 corners. Using corner mass = 0.25·dm_cell keeps the sum
// Σ_cell p_cell = Σ_node m_node · v_node exactly (because corner
// shares sum back to node mass).
// ============================================================
__global__
void k_cale_cell_momentum(const double* vX, const double* vY,
                          const double* dm,
                          double* px, double* py,
                          int nx, int ny) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nx*ny) return;
    int ic = flat / ny, jc = flat % ny;
    int nny = ny + 1;
    int I[4] = { caln(ic,   jc,   nny),
                 caln(ic+1, jc,   nny),
                 caln(ic+1, jc+1, nny),
                 caln(ic,   jc+1, nny) };
    double qm = 0.25 * dm[flat];
    double sx = 0.0, sy = 0.0;
    for (int k = 0; k < 4; ++k) {
        sx += qm * vX[I[k]];
        sy += qm * vY[I[k]];
    }
    px[flat] = sx;
    py[flat] = sy;
}

// ============================================================
// Swept-edge remap — the heart of ALE.
//
// For each INTERIOR east edge between cell L = (ic, jc) and
// cell R = (ic+1, jc), the swept quadrilateral carries mass (and
// other conserved) in a direction determined by the signed swept
// area. We do both directions (east edges + north edges) as
// separate passes to avoid atomic contention.
//
// Swept area (signed) of an east edge whose nodes are A→A' (bottom)
// and B→B' (top):
//   A  = (ic+1, jc)        node at south end (old pos)
//   A' = same in (X,Y)     node at south end (new pos)
//   B  = (ic+1, jc+1)      node at north end (old pos)
//   B' = same in (X,Y)     new pos
// The quadrilateral A-A'-B'-B in CCW yields a signed area.
// If positive → swept volume moved from L into R; donor = L.
//
// We use first-order donor-cell upwind: transported scalar = ρ_donor·V_sweep.
// ============================================================

// Signed swept area of quad (A_old → A_new → B_new → B_old).
// Positive = quad oriented CCW = the sweep region is on the
// "forward" side of the edge.
__device__ __forceinline__
double swept_quad_signed(double Ax, double Ay,
                         double Anx, double Any,
                         double Bnx, double Bny,
                         double Bx, double By) {
    // Shoelace for the 4-node polygon A, A', B', B (this order).
    double s = (Ax*Any - Anx*Ay)
             + (Anx*Bny - Bnx*Any)
             + (Bnx*By - Bx*Bny)
             + (Bx*Ay - Ax*By);
    return 0.5 * s;
}

// East edges: edge between (ic, jc) and (ic+1, jc).
// Ownership nodes: south = (ic+1, jc), north = (ic+1, jc+1).
// Swept from OLD=X0 to NEW=X.
__global__
void k_cale_remap_east(const double* X0, const double* Y0,
                       const double* X,  const double* Y,
                       const double* dm, const double* e_int,
                       const double* px, const double* py,
                       const double* Vol0,             // old cell volume
                       double* dm_new, double* ie_new,
                       double* px_new, double* py_new,
                       int nx, int ny) {
    // Thread indexes one interior east edge: (ic, jc), ic in [0..nx-2], jc in [0..ny-1].
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int n_edges = (nx - 1) * ny;
    if (flat >= n_edges) return;
    int ic = flat / ny, jc = flat % ny;
    int nny = ny + 1;

    int nA = caln(ic+1, jc,   nny);
    int nB = caln(ic+1, jc+1, nny);
    double Ax = X0[nA], Ay = Y0[nA];
    double Bx = X0[nB], By = Y0[nB];
    double Anx = X[nA], Any = Y[nA];
    double Bnx = X[nB], Bny = Y[nB];
    double As = swept_quad_signed(Ax, Ay, Anx, Any, Bnx, Bny, Bx, By);
    if (As == 0.0) return;

    int cL = calc(ic,   jc, ny);
    int cR = calc(ic+1, jc, ny);
    int donor = (As > 0.0) ? cL : cR;
    double V_sweep = fabs(As);
    double V_donor = fmax(Vol0[donor], 1e-30);
    double frac = fmin(V_sweep / V_donor, 0.5);   // clamp for safety
    // Actual volume transferred:
    double V = frac * V_donor;

    double d_dm = (dm[donor] / V_donor) * V;
    double d_ie = (dm[donor] * e_int[donor] / V_donor) * V;
    double d_px = (px[donor] / V_donor) * V;
    double d_py = (py[donor] / V_donor) * V;

    if (As > 0.0) {
        // donor = L, receiver = R
        atomicAdd(&dm_new[cL], -d_dm);
        atomicAdd(&ie_new[cL], -d_ie);
        atomicAdd(&px_new[cL], -d_px);
        atomicAdd(&py_new[cL], -d_py);
        atomicAdd(&dm_new[cR],  d_dm);
        atomicAdd(&ie_new[cR],  d_ie);
        atomicAdd(&px_new[cR],  d_px);
        atomicAdd(&py_new[cR],  d_py);
    } else {
        atomicAdd(&dm_new[cR], -d_dm);
        atomicAdd(&ie_new[cR], -d_ie);
        atomicAdd(&px_new[cR], -d_px);
        atomicAdd(&py_new[cR], -d_py);
        atomicAdd(&dm_new[cL],  d_dm);
        atomicAdd(&ie_new[cL],  d_ie);
        atomicAdd(&px_new[cL],  d_px);
        atomicAdd(&py_new[cL],  d_py);
    }
}

// North edges: edge between (ic, jc) and (ic, jc+1).
// Ownership nodes: west = (ic, jc+1), east = (ic+1, jc+1).
__global__
void k_cale_remap_north(const double* X0, const double* Y0,
                        const double* X,  const double* Y,
                        const double* dm, const double* e_int,
                        const double* px, const double* py,
                        const double* Vol0,
                        double* dm_new, double* ie_new,
                        double* px_new, double* py_new,
                        int nx, int ny) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int n_edges = nx * (ny - 1);
    if (flat >= n_edges) return;
    int ic = flat / (ny - 1), jc = flat % (ny - 1);
    int nny = ny + 1;

    int nW = caln(ic,   jc+1, nny);
    int nE = caln(ic+1, jc+1, nny);
    double Ax = X0[nE], Ay = Y0[nE];          // east old
    double Anx = X[nE],  Any = Y[nE];
    double Bx = X0[nW], By = Y0[nW];          // west old
    double Bnx = X[nW],  Bny = Y[nW];
    // Quad (E_old → E_new → W_new → W_old) — oriented so that positive
    // signed area means the sweep moved into the cell ABOVE (jc+1).
    double As = swept_quad_signed(Ax, Ay, Anx, Any, Bnx, Bny, Bx, By);
    if (As == 0.0) return;

    int cD = calc(ic, jc,   ny);     // down
    int cU = calc(ic, jc+1, ny);     // up
    int donor = (As > 0.0) ? cD : cU;
    double V_sweep = fabs(As);
    double V_donor = fmax(Vol0[donor], 1e-30);
    double frac = fmin(V_sweep / V_donor, 0.5);
    double V = frac * V_donor;

    double d_dm = (dm[donor] / V_donor) * V;
    double d_ie = (dm[donor] * e_int[donor] / V_donor) * V;
    double d_px = (px[donor] / V_donor) * V;
    double d_py = (py[donor] / V_donor) * V;

    if (As > 0.0) {
        atomicAdd(&dm_new[cD], -d_dm);
        atomicAdd(&ie_new[cD], -d_ie);
        atomicAdd(&px_new[cD], -d_px);
        atomicAdd(&py_new[cD], -d_py);
        atomicAdd(&dm_new[cU],  d_dm);
        atomicAdd(&ie_new[cU],  d_ie);
        atomicAdd(&px_new[cU],  d_px);
        atomicAdd(&py_new[cU],  d_py);
    } else {
        atomicAdd(&dm_new[cU], -d_dm);
        atomicAdd(&ie_new[cU], -d_ie);
        atomicAdd(&px_new[cU], -d_px);
        atomicAdd(&py_new[cU], -d_py);
        atomicAdd(&dm_new[cD],  d_dm);
        atomicAdd(&ie_new[cD],  d_ie);
        atomicAdd(&px_new[cD],  d_px);
        atomicAdd(&py_new[cD],  d_py);
    }
}

// Initialize remap accumulators to the Lagrangian-state values
// (so that remap kernels *add* flux corrections).
__global__
void k_cale_remap_init(const double* dm, const double* e_int,
                       const double* px, const double* py,
                       double* dm_new, double* ie_new,
                       double* px_new, double* py_new,
                       int ncell) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncell) return;
    dm_new[c] = dm[c];
    ie_new[c] = dm[c] * e_int[c];
    px_new[c] = px[c];
    py_new[c] = py[c];
}

// Finalize: write dm_new → dm, ie_new/dm_new → e_int, p/m → velocity contribs.
__global__
void k_cale_remap_finalize_cells(const double* dm_new, const double* ie_new,
                                 double* dm, double* e_int,
                                 int ncell) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncell) return;
    double m = fmax(dm_new[c], 1e-30);
    dm[c] = dm_new[c];
    e_int[c] = ie_new[c] / m;
}

// Rebuild node velocity = Σ_adj_cells (p_cell / m_cell) · 0.25  /  (Σ_adj 0.25)
// i.e. area-weighted average of adjacent cell-center velocities.
// Using 1/4 weights per cell (all 4 corners contribute equally) gives
// simple arithmetic mean of the (up to 4) adjacent cell-center velocities.
__global__
void k_cale_rebuild_node_v(const double* px_new, const double* py_new,
                           const double* dm_new,
                           double* vX, double* vY,
                           int nx, int ny) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int nnx = nx + 1, nny = ny + 1;
    int n = nnx * nny;
    if (flat >= n) return;
    int in = flat / nny, jn = flat % nny;
    double sx = 0.0, sy = 0.0, w = 0.0;
    for (int di = -1; di <= 0; ++di) {
        for (int dj = -1; dj <= 0; ++dj) {
            int ic = in + di, jc = jn + dj;
            if (ic < 0 || ic >= nx || jc < 0 || jc >= ny) continue;
            int c = ic * ny + jc;
            double m = fmax(dm_new[c], 1e-30);
            sx += px_new[c] / m;
            sy += py_new[c] / m;
            w  += 1.0;
        }
    }
    if (w > 0.0) {
        vX[flat] = sx / w;
        vY[flat] = sy / w;
    }
}

// ============================================================
// 2nd-order MUSCL-in-remap (Kucharik & Shashkov 2012, JCP 231).
//
// Idea: instead of using the donor's cell-average as the swept value,
// fit a minmod-limited linear reconstruction inside each donor cell
//   f(x,y) = f_bar + s_x · (x − x_c) + s_y · (y − y_c)
// and evaluate it at the swept-region centroid. This raises the
// formal order of the remap from 1st to 2nd in smooth regions
// while minmod kills slopes near discontinuities — preserving
// monotonicity and TVD behavior.
//
// Because the rezone target is the uniform reference mesh,
//   donor cell volume V0 is constant Area0
//   donor cell centroid is (x_c, y_c) on the uniform lattice
// so both are trivially known from (ic, jc, dx_u, dy_u).
//
// The 4 conserved "densities" (per unit reference volume) are:
//   ρ     = dm / V0
//   ρE    = (dm·e_int) / V0
//   px_d  = px / V0
//   py_d  = py / V0
// Transported increment per swept region = f_donor(centroid) · V_sweep.
// ============================================================

// Build reference-volume densities from (dm, dm·e, px, py).
__global__
void k_cale_cell_densities(const double* dm, const double* e_int,
                           const double* px, const double* py,
                           const double* Area0,
                           double* rho_d, double* rhoE_d,
                           double* pxd, double* pyd,
                           int ncell) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncell) return;
    double V = fmax(Area0[c], 1e-30);
    rho_d[c]  = dm[c] / V;
    rhoE_d[c] = dm[c] * e_int[c] / V;
    pxd[c]    = px[c] / V;
    pyd[c]    = py[c] / V;
}

__device__ __forceinline__ double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    double aa = fabs(a), ab = fabs(b);
    return (aa < ab) ? a : b;
}

// van Leer harmonic limiter: smoother than minmod near smooth extrema.
// Returns 2ab/(a+b) when a*b > 0, else 0.
__device__ __forceinline__ double vanleer(double a, double b) {
    double ab = a * b;
    if (ab <= 0.0) return 0.0;
    return 2.0 * ab / (a + b);
}

// MC (monotonized central) limiter: limit[minmod(2a, 2b, (a+b)/2)].
// Slightly sharper than van Leer, still TVD.
__device__ __forceinline__ double mc_lim(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    double ctr = 0.5 * (a + b);
    double two_a = 2.0 * a, two_b = 2.0 * b;
    double sgn = (a > 0.0) ? 1.0 : -1.0;
    double amin = fmin(fmin(fabs(ctr), fabs(two_a)), fabs(two_b));
    return sgn * amin;
}

__device__ __forceinline__ double apply_limiter(double a, double b, int id) {
    if (id == 1) return vanleer(a, b);
    if (id == 2) return mc_lim(a, b);
    return minmod(a, b);
}

// Per-cell limited slopes on the uniform reference mesh.
// Boundary cells use a one-sided slope if only one neighbor exists,
// but we keep it simple and zero them (donor-cell limit near walls).
// limiter_id: 0=minmod (default), 1=van Leer, 2=MC.
__global__
void k_cale_slopes_minmod(const double* rho_d, const double* rhoE_d,
                          const double* pxd,   const double* pyd,
                          double* rho_sx,  double* rho_sy,
                          double* rhoE_sx, double* rhoE_sy,
                          double* pxd_sx,  double* pxd_sy,
                          double* pyd_sx,  double* pyd_sy,
                          int nx, int ny, double dx_u, double dy_u,
                          int limiter_id) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nx*ny) return;
    int ic = flat / ny, jc = flat % ny;

    auto slopes = [&](const double* f, double& sx, double& sy) {
        double fc = f[flat];
        if (ic > 0 && ic < nx - 1) {
            double dfL = (fc - f[(ic-1)*ny + jc]) / dx_u;
            double dfR = (f[(ic+1)*ny + jc] - fc) / dx_u;
            sx = apply_limiter(dfL, dfR, limiter_id);
        } else {
            sx = 0.0;
        }
        if (jc > 0 && jc < ny - 1) {
            double dfD = (fc - f[ic*ny + (jc-1)]) / dy_u;
            double dfU = (f[ic*ny + (jc+1)] - fc) / dy_u;
            sy = apply_limiter(dfD, dfU, limiter_id);
        } else {
            sy = 0.0;
        }
    };

    slopes(rho_d,  rho_sx[flat],  rho_sy[flat]);
    slopes(rhoE_d, rhoE_sx[flat], rhoE_sy[flat]);
    slopes(pxd,    pxd_sx[flat],  pxd_sy[flat]);
    slopes(pyd,    pyd_sx[flat],  pyd_sy[flat]);
}

// 2nd-order swept-edge remap — east edges.
//
// Geometry is identical to the 1st-order version: swept quad is
// (A_old, A_new, B_new, B_old). Difference: we evaluate each donor
// field at the *swept centroid* (4-point average of the quad vertices),
// using the minmod-limited linear reconstruction.
__global__
void k_cale_remap_east_2nd(const double* X0, const double* Y0,
                           const double* X,  const double* Y,
                           const double* rho_d,  const double* rhoE_d,
                           const double* pxd,    const double* pyd,
                           const double* rho_sx, const double* rho_sy,
                           const double* rhoE_sx,const double* rhoE_sy,
                           const double* pxd_sx, const double* pxd_sy,
                           const double* pyd_sx, const double* pyd_sy,
                           double* dm_new, double* ie_new,
                           double* px_new, double* py_new,
                           int nx, int ny, double dx_u, double dy_u) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int n_edges = (nx - 1) * ny;
    if (flat >= n_edges) return;
    int ic = flat / ny, jc = flat % ny;
    int nny = ny + 1;

    int nA = caln(ic+1, jc,   nny);
    int nB = caln(ic+1, jc+1, nny);
    double Ax = X0[nA], Ay = Y0[nA];
    double Bx = X0[nB], By = Y0[nB];
    double Anx = X[nA], Any = Y[nA];
    double Bnx = X[nB], Bny = Y[nB];
    double As = swept_quad_signed(Ax, Ay, Anx, Any, Bnx, Bny, Bx, By);
    if (As == 0.0) return;

    int cL = calc(ic,   jc, ny);
    int cR = calc(ic+1, jc, ny);
    int donor = (As > 0.0) ? cL : cR;
    double V_sweep = fabs(As);

    // Swept-region centroid (4-point average — OK for small skinny quads).
    double cx = 0.25 * (Ax + Anx + Bnx + Bx);
    double cy = 0.25 * (Ay + Any + Bny + By);

    // Donor cell centroid on the uniform reference mesh.
    int dic = donor / ny, djc = donor % ny;
    double xd = (dic + 0.5) * dx_u;
    double yd = (djc + 0.5) * dy_u;
    double ex = cx - xd, ey = cy - yd;

    double d_dm = (rho_d[donor]  + rho_sx[donor]*ex  + rho_sy[donor]*ey)  * V_sweep;
    double d_ie = (rhoE_d[donor] + rhoE_sx[donor]*ex + rhoE_sy[donor]*ey) * V_sweep;
    double d_px = (pxd[donor]    + pxd_sx[donor]*ex  + pxd_sy[donor]*ey)  * V_sweep;
    double d_py = (pyd[donor]    + pyd_sx[donor]*ex  + pyd_sy[donor]*ey)  * V_sweep;

    if (As > 0.0) {
        atomicAdd(&dm_new[cL], -d_dm);
        atomicAdd(&ie_new[cL], -d_ie);
        atomicAdd(&px_new[cL], -d_px);
        atomicAdd(&py_new[cL], -d_py);
        atomicAdd(&dm_new[cR],  d_dm);
        atomicAdd(&ie_new[cR],  d_ie);
        atomicAdd(&px_new[cR],  d_px);
        atomicAdd(&py_new[cR],  d_py);
    } else {
        atomicAdd(&dm_new[cR], -d_dm);
        atomicAdd(&ie_new[cR], -d_ie);
        atomicAdd(&px_new[cR], -d_px);
        atomicAdd(&py_new[cR], -d_py);
        atomicAdd(&dm_new[cL],  d_dm);
        atomicAdd(&ie_new[cL],  d_ie);
        atomicAdd(&px_new[cL],  d_px);
        atomicAdd(&py_new[cL],  d_py);
    }
}

__global__
void k_cale_remap_north_2nd(const double* X0, const double* Y0,
                            const double* X,  const double* Y,
                            const double* rho_d,  const double* rhoE_d,
                            const double* pxd,    const double* pyd,
                            const double* rho_sx, const double* rho_sy,
                            const double* rhoE_sx,const double* rhoE_sy,
                            const double* pxd_sx, const double* pxd_sy,
                            const double* pyd_sx, const double* pyd_sy,
                            double* dm_new, double* ie_new,
                            double* px_new, double* py_new,
                            int nx, int ny, double dx_u, double dy_u) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int n_edges = nx * (ny - 1);
    if (flat >= n_edges) return;
    int ic = flat / (ny - 1), jc = flat % (ny - 1);
    int nny = ny + 1;

    int nW = caln(ic,   jc+1, nny);
    int nE = caln(ic+1, jc+1, nny);
    double Ax = X0[nE], Ay = Y0[nE];
    double Anx = X[nE],  Any = Y[nE];
    double Bx = X0[nW], By = Y0[nW];
    double Bnx = X[nW],  Bny = Y[nW];
    double As = swept_quad_signed(Ax, Ay, Anx, Any, Bnx, Bny, Bx, By);
    if (As == 0.0) return;

    int cD = calc(ic, jc,   ny);
    int cU = calc(ic, jc+1, ny);
    int donor = (As > 0.0) ? cD : cU;
    double V_sweep = fabs(As);

    double cx = 0.25 * (Ax + Anx + Bnx + Bx);
    double cy = 0.25 * (Ay + Any + Bny + By);

    int dic = donor / ny, djc = donor % ny;
    double xd = (dic + 0.5) * dx_u;
    double yd = (djc + 0.5) * dy_u;
    double ex = cx - xd, ey = cy - yd;

    double d_dm = (rho_d[donor]  + rho_sx[donor]*ex  + rho_sy[donor]*ey)  * V_sweep;
    double d_ie = (rhoE_d[donor] + rhoE_sx[donor]*ex + rhoE_sy[donor]*ey) * V_sweep;
    double d_px = (pxd[donor]    + pxd_sx[donor]*ex  + pxd_sy[donor]*ey)  * V_sweep;
    double d_py = (pyd[donor]    + pyd_sx[donor]*ex  + pyd_sy[donor]*ey)  * V_sweep;

    if (As > 0.0) {
        atomicAdd(&dm_new[cD], -d_dm);
        atomicAdd(&ie_new[cD], -d_ie);
        atomicAdd(&px_new[cD], -d_px);
        atomicAdd(&py_new[cD], -d_py);
        atomicAdd(&dm_new[cU],  d_dm);
        atomicAdd(&ie_new[cU],  d_ie);
        atomicAdd(&px_new[cU],  d_px);
        atomicAdd(&py_new[cU],  d_py);
    } else {
        atomicAdd(&dm_new[cU], -d_dm);
        atomicAdd(&ie_new[cU], -d_ie);
        atomicAdd(&px_new[cU], -d_px);
        atomicAdd(&py_new[cU], -d_py);
        atomicAdd(&dm_new[cD],  d_dm);
        atomicAdd(&ie_new[cD],  d_ie);
        atomicAdd(&px_new[cD],  d_px);
        atomicAdd(&py_new[cD],  d_py);
    }
}

// Snapshot (ρ, P, e_int, vx_cc, vy_cc) into a contiguous per-frame
// slice of the frame pool. Layout per frame:
//   [0 .. ncell)         ρ
//   [ncell .. 2·ncell)   P
//   [2·ncell .. 3·ncell) e_int
//   [3·ncell .. 4·ncell) vx_cc (avg of 4 corner vx)
//   [4·ncell .. 5·ncell) vy_cc
__global__
void k_cale_snapshot(const double* rho, const double* P, const double* e_int,
                     const double* vX, const double* vY,
                     double* out, int nx, int ny) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nx*ny) return;
    int ic = flat / ny, jc = flat % ny;
    int nny = ny + 1;
    int I[4] = { caln(ic,   jc,   nny),
                 caln(ic+1, jc,   nny),
                 caln(ic+1, jc+1, nny),
                 caln(ic,   jc+1, nny) };
    double vx = 0.25 * (vX[I[0]] + vX[I[1]] + vX[I[2]] + vX[I[3]]);
    double vy = 0.25 * (vY[I[0]] + vY[I[1]] + vY[I[2]] + vY[I[3]]);
    int n = nx * ny;
    out[flat]         = rho[flat];
    out[flat + n]     = P[flat];
    out[flat + 2*n]   = e_int[flat];
    out[flat + 3*n]   = vx;
    out[flat + 4*n]   = vy;
}

// Clamp edge-aligned node velocities AFTER rebuild (reflective BC)
__global__
void k_cale_bc_velocity(double* vX, double* vY, int nnx, int nny) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nnx * nny;
    if (flat >= n) return;
    int in = flat / nny, jn = flat % nny;
    if (in == 0 || in == nnx - 1) vX[flat] = 0.0;
    if (jn == 0 || jn == nny - 1) vY[flat] = 0.0;
}
