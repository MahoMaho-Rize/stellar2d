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

#include "cart_ale2_solver.cuh"
#include "gpu_common.cuh"
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
void k_cale2_geometry(const double* X, const double* Y,
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
    // Eq. (15.2): shoelace formula for quadrilateral signed area.
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
void k_cale2_eos_and_q(const double* X, const double* Y,
                      const double* vX, const double* vY,
                      const double* dm, const double* Vol, const double* Area0,
                      const double* e_int,
                      double* rho, double* P, double* Q, double* cs,
                      double* strain_rate,
                      int nx, int ny, double gam,
                      double CQ_lin, double CQ_quad,
                      int shear_aware_av) {
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
    // Eq. (15.3): divergence-consistent strain rate from edge-averaged velocity
    // dotted with outward normal, summed over 4 edges; s = -div(v).
    double dAdt = edge_dAdt(X0,Y0,X1,Y1,vX0,vY0,vX1,vY1)
                + edge_dAdt(X1,Y1,X2,Y2,vX1,vY1,vX2,vY2)
                + edge_dAdt(X2,Y2,X3,Y3,vX2,vY2,vX3,vY3)
                + edge_dAdt(X3,Y3,X0,Y0,vX3,vY3,vX0,vY0);
    double s = -dAdt / V;
    strain_rate[flat] = s;

    double L = sqrt(fmax(Area0[flat], 1e-30));

    // Shear-aware AV (simplified tensor-AV): reduce Q when the cell is
    // dominated by pure shear (no compression normal to any edge) rather
    // than true compression. This is the single biggest factor distinguishing
    // our KH visual from Athena-class Godunov codes, because scalar Q would
    // otherwise fire on every discretisation error at a shear layer.
    //
    // Detect compression anisotropically by computing ∂v_n/∂n along both
    // local axes (uniform mesh makes this trivial). If BOTH directions are
    // dilating (∂u/∂x > 0 AND ∂v/∂y > 0), even a positive div·v is spurious
    // → kill Q. If only one is compressing, reduce Q by that fraction.
    double shear_weight = 1.0;
    if (shear_aware_av) {
        // Estimate ∂u/∂x and ∂v/∂y from the staggered node velocities.
        // Uniform mesh: east face vx avg − west face vx avg, divided by dx.
        double dx = fmax(X1 - X0, 1e-30);   // approx east-west spacing at bottom edge
        double dy = fmax(Y3 - Y0, 1e-30);   // approx north-south at west edge
        // du/dx: (avg vx on east edge) − (avg vx on west edge), per dx.
        //        east edge = nodes 1,2; west edge = nodes 0,3.
        double du_dx = 0.5 * ((vX1 + vX2) - (vX0 + vX3)) / dx;
        // dv/dy: (avg vy on north edge) − (avg vy on south edge), per dy.
        //        north edge = nodes 2,3; south edge = nodes 0,1.
        double dv_dy = 0.5 * ((vY2 + vY3) - (vY0 + vY1)) / dy;
        double comp_x = -du_dx;   // > 0 = compression
        double comp_y = -dv_dy;
        // Only count positive (compression) parts; normalised to their sum
        double cx = fmax(comp_x, 0.0);
        double cy = fmax(comp_y, 0.0);
        double csum = cx + cy;
        double tot  = fabs(comp_x) + fabs(comp_y) + 1e-30;
        // When both axes compress evenly, weight=1 (full shock); when only
        // one compresses, weight = that fraction of total |∂v_n/∂n|; when
        // neither, weight=0 (pure shear/dilation).
        shear_weight = csum / tot;
    }

    // Eq. (15.4): compression-only von Neumann-Richtmyer AV with optional
    // shear-aware tensor weighting.
    double q = 0.0;
    if (s > 0.0) {
        double q_quad = CQ_quad * s * L * s * L;
        double q_lin  = CQ_lin  * cs[flat] * s * L;
        q = shear_weight * r_ * (q_quad + q_lin);
    }
    Q[flat] = q;
}

// ============================================================
// Node + subcell forces
// ============================================================
__global__
void k_cale2_node_forces(const double* X, const double* Y,
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

    // Eq. (15.5): edge-normal integrated force (P+Q)·(Δy, −Δx) per cell edge.
    double aX[4], aY[4];
    for (int k = 0; k < 4; ++k) {
        int kp = (k + 1) & 3;
        double dx = Xk[kp] - Xk[k];
        double dy = Yk[kp] - Yk[k];
        aX[k] =  PQ * dy;
        aY[k] = -PQ * dx;
    }
    // Eq. (15.6): corner subcell force = ½(edge_{k−1} + edge_k); atomicAdd
    // to the 4 node forces. Also stash per-corner value in FSX/FSY for the
    // compatible energy update (Eq. 15.8).
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
void k_cale2_zero(double* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 0.0;
}

__global__
void k_cale2_add_gravity(const double* mnode, double* FY,
                        double g_val, int nnode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnode) return;
    FY[i] += -g_val * mnode[i];
}

// Variable gravity: g value per NODE-ROW (row index jn = i / nnode_x).
// FY layout: column-major over (ix, iy) flattened — see cart_ale2_solver.cu
// for exact: node index i = ix * nnode_y + iy  (see k_cale2_node_update).
__global__
void k_cale2_add_gravity_var(const double* mnode, double* FY,
                             const double* gy_per_row,
                             int nnode_x, int nnode_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nnode = nnode_x * nnode_y;
    if (i >= nnode) return;
    int iy = i % nnode_y;   // matches ix*nnode_y + iy flattening
    FY[i] += -gy_per_row[iy] * mnode[i];
}

// ============================================================
// Reflective walls: pin positions at initial (X0, Y0) on boundary,
// and zero the wall-normal force/velocity components each step.
// Because the rezone phase will snap nodes back to (X0,Y0) anyway,
// pinning is almost redundant — but it keeps the Lagrangian
// substep honest.
// ============================================================
// bc_mode encoding:
//   bit 0 (0x1): x-direction periodic (else reflective)
//   bit 1 (0x2): y-direction periodic (else reflective)
//   0 = both reflective (default, backward-compatible)
//   3 = both periodic
__global__
void k_cale2_bc_reflective(const double* X0, const double* Y0,
                          double* X, double* Y,
                          double* vX, double* vY,
                          double* FX, double* FY,
                          int nnx, int nny, int bc_mode) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nnx * nny;
    if (flat >= n) return;
    int in = flat / nny, jn = flat % nny;
    bool x_per = (bc_mode & 1) != 0;
    bool y_per = (bc_mode & 2) != 0;
    if (!x_per && (in == 0 || in == nnx - 1)) {
        X[flat] = X0[flat];
        vX[flat] = 0.0;
        FX[flat] = 0.0;
    }
    if (!y_per && (jn == 0 || jn == nny - 1)) {
        Y[flat] = Y0[flat];
        vY[flat] = 0.0;
        FY[flat] = 0.0;
    }
}

// ============================================================
// Kick-drift-kick node update (same as cart_lag)
// ============================================================
__global__
void k_cale2_node_update(double* X, double* Y,
                        double* vX, double* vY,
                        const double* FX, const double* FY,
                        const double* mnode,
                        double* dX_out, double* dY_out,
                        double dt, int nnode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnode) return;
    // Eq. (15.7): kick-drift-kick — v_half with ½Δv, drift X with v_half·Δt,
    // full Δv applied to v so it exits at t+Δt.
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
void k_cale2_energy_update(int nx, int ny,
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
    // Eq. (15.8): compatible energy update — cell internal energy decreases
    // by exactly the work the cell does on its 4 corner nodes (F_sub · dX).
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
void k_cale2_cfl(const double* minheight, const double* cs,
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
    // Eq. (15.9): minimum-height CFL — min(sound-wave crossing, compression
    // strain) on the possibly-deformed quadrilateral.
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
void k_cale2_init_nodes(double* X, double* Y,
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
void k_cale2_reset_mesh(const double* X0, const double* Y0,
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
void k_cale2_node_mass(const double* dm, double* mnode,
                      int nx, int ny, int bc_mode) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int nnx = nx + 1, nny = ny + 1;
    int n = nnx * nny;
    if (flat >= n) return;
    int in = flat / nny, jn = flat % nny;
    bool x_per = (bc_mode & 1) != 0;
    bool y_per = (bc_mode & 2) != 0;
    // Eq. (15.1): node mass = ¼ Σ of adjacent cell masses. Periodic wrap
    // makes in=0 and in=nnx-1 each independently see 4 cells (same physical
    // point) — diagnostics (compute_diagnostics) must skip one duplicate
    // when summing to avoid double counting (see P31).
    double m = 0.0;
    for (int di = -1; di <= 0; ++di) {
        for (int dj = -1; dj <= 0; ++dj) {
            int ic = in + di, jc = jn + dj;
            if (x_per) {
                if (ic < 0)  ic = nx - 1;
                if (ic >= nx) ic = 0;
            } else {
                if (ic < 0 || ic >= nx) continue;
            }
            if (y_per) {
                if (jc < 0)  jc = ny - 1;
                if (jc >= ny) jc = 0;
            } else {
                if (jc < 0 || jc >= ny) continue;
            }
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
void k_cale2_cell_momentum(const double* vX, const double* vY,
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

// Eq. (16.1): signed swept area of quad (A_old → A_new → B_new → B_old).
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
void k_cale2_remap_east(const double* X0, const double* Y0,
                       const double* X,  const double* Y,
                       const double* dm, const double* e_int,
                       const double* px, const double* py,
                       const double* Vol0,             // old cell volume
                       double* dm_new, double* ie_new,
                       double* px_new, double* py_new,
                       int nx, int ny, int bc_mode) {
    // Thread indexes one east edge: (ic, jc), ic in [0..nx-1], jc in [0..ny-1].
    // When bc_mode & 1 (x-periodic), edge ic=nx-1 connects cell nx-1 to cell 0
    // via wrap; otherwise that edge is skipped (reflective walls handle it).
    bool x_per = (bc_mode & 1) != 0;
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int n_edges = x_per ? nx * ny : (nx - 1) * ny;
    if (flat >= n_edges) return;
    int ic = flat / ny, jc = flat % ny;
    int nny = ny + 1;

    // Node indices on the east face of cell ic: nodes (ic+1, jc) and (ic+1, jc+1).
    // For the wrap edge (ic=nx-1 under periodic BC), those are nodes (nx, *) —
    // the right-edge duplicates of nodes (0, *). Because periodic_sync keeps
    // both copies bit-identical, using (ic+1) is correct either way.
    int nA = caln(ic+1, jc,   nny);
    int nB = caln(ic+1, jc+1, nny);
    double Ax = X0[nA], Ay = Y0[nA];
    double Bx = X0[nB], By = Y0[nB];
    double Anx = X[nA], Any = Y[nA];
    double Bnx = X[nB], Bny = Y[nB];
    double As = swept_quad_signed(Ax, Ay, Anx, Any, Bnx, Bny, Bx, By);
    if (As == 0.0) return;

    int cL = calc(ic,   jc, ny);
    int cR_idx = ic + 1;
    if (x_per && cR_idx >= nx) cR_idx = 0;
    int cR = calc(cR_idx, jc, ny);
    int donor = (As > 0.0) ? cL : cR;
    double V_sweep = fabs(As);
    double V_donor = fmax(Vol0[donor], 1e-30);
    double frac = fmin(V_sweep / V_donor, 0.5);   // clamp for safety
    // Actual volume transferred:
    double V = frac * V_donor;

    // Eq. (16.2): first-order donor-cell upwind flux — transported scalar
    // equals the donor-cell density times the swept volume.
    double d_dm = (dm[donor] / V_donor) * V;
    double d_ie = (dm[donor] * e_int[donor] / V_donor) * V;
    double d_px = (px[donor] / V_donor) * V;
    double d_py = (py[donor] / V_donor) * V;

    // Eq. (16.5): donor subtracts, acceptor adds the same value.
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
void k_cale2_remap_north(const double* X0, const double* Y0,
                        const double* X,  const double* Y,
                        const double* dm, const double* e_int,
                        const double* px, const double* py,
                        const double* Vol0,
                        double* dm_new, double* ie_new,
                        double* px_new, double* py_new,
                        int nx, int ny, int bc_mode) {
    bool y_per = (bc_mode & 2) != 0;
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int ny_eff = y_per ? ny : (ny - 1);
    int n_edges = nx * ny_eff;
    if (flat >= n_edges) return;
    int ic = flat / ny_eff, jc = flat % ny_eff;
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
    int cU_idx = jc + 1;
    if (y_per && cU_idx >= ny) cU_idx = 0;
    int cU = calc(ic, cU_idx, ny);     // up (wraps under y-periodic)
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
void k_cale2_remap_init(const double* dm, const double* e_int,
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
void k_cale2_remap_finalize_cells(const double* dm_new, const double* ie_new,
                                 double* dm, double* e_int,
                                 int ncell) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncell) return;
    double m = fmax(dm_new[c], 1e-30);
    dm[c] = dm_new[c];
    e_int[c] = ie_new[c] / m;
}

// Rebuild node velocity — mass-weighted average of adjacent cell-centered
// velocities. For a node with up to 4 adjacent cells:
//   v_node = Σ(0.25·m_c · v_c) / Σ(0.25·m_c) = Σ p_c/4 / Σ m_c/4
// Equivalently Σ px_c / Σ m_c (each cell contributes 1/4 of its mass to
// a node). This is exactly momentum-conservative: the kinetic energy
// loss from the averaging is proportional to ∇v variance within the
// stencil, which is physically meaningful (sub-grid diffusion), not
// the arithmetic-mean artifact of the old rebuild.
__global__
void k_cale2_rebuild_node_v(const double* px_new, const double* py_new,
                           const double* dm_new,
                           double* vX, double* vY,
                           int nx, int ny, int bc_mode) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int nnx = nx + 1, nny = ny + 1;
    int n = nnx * nny;
    if (flat >= n) return;
    int in = flat / nny, jn = flat % nny;
    bool x_per = (bc_mode & 1) != 0;
    bool y_per = (bc_mode & 2) != 0;
    double spx = 0.0, spy = 0.0, sm = 0.0;
    for (int di = -1; di <= 0; ++di) {
        for (int dj = -1; dj <= 0; ++dj) {
            int ic = in + di, jc = jn + dj;
            if (x_per) {
                if (ic < 0)  ic = nx - 1;
                if (ic >= nx) ic = 0;
            } else {
                if (ic < 0 || ic >= nx) continue;
            }
            if (y_per) {
                if (jc < 0)  jc = ny - 1;
                if (jc >= ny) jc = 0;
            } else {
                if (jc < 0 || jc >= ny) continue;
            }
            int c = ic * ny + jc;
            // each adjacent cell contributes 1/4 of its corner mass+momentum.
            spx += 0.25 * px_new[c];
            spy += 0.25 * py_new[c];
            sm  += 0.25 * dm_new[c];
        }
    }
    // Eq. (16.6): mass-weighted rebuild — momentum-conservative.
    if (sm > 1e-30) {
        vX[flat] = spx / sm;
        vY[flat] = spy / sm;
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
void k_cale2_cell_densities(const double* dm, const double* e_int,
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

// Build primitive variables (ρ, P, vx, vy) for primitive-space PPM.
// Reuses the 4 "density" buffers (rho_dens, rhoE_dens, pxd_dens, pyd_dens)
// as scratch — kernel output is:
//   rho_d  ← ρ        (unchanged semantics)
//   rhoE_d ← P        (pressure, γ-1)·ρ·e_int  [semantic reuse]
//   pxd    ← vx       [semantic reuse]
//   pyd    ← vy       [semantic reuse]
// PPM then reconstructs smooth primitives, avoiding the px=ρvx sign flip
// pathology that crashes conserved-variable PPM on tanh shear layers.
__global__
void k_cale2_cell_primitives(const double* dm, const double* e_int,
                            const double* px, const double* py,
                            const double* Area0,
                            double* rho_out, double* P_out,
                            double* vx_out, double* vy_out,
                            int ncell, double gam) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncell) return;
    double V   = fmax(Area0[c], 1e-30);
    double m   = dm[c];
    double rho = m / V;
    double inv_m = 1.0 / fmax(m, 1e-30);
    rho_out[c] = rho;
    P_out[c]   = (gam - 1.0) * rho * e_int[c];
    vx_out[c]  = px[c] * inv_m;
    vy_out[c]  = py[c] * inv_m;
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
void k_cale2_slopes_minmod(const double* rho_d, const double* rhoE_d,
                          const double* pxd,   const double* pyd,
                          double* rho_sx,  double* rho_sy,
                          double* rhoE_sx, double* rhoE_sy,
                          double* pxd_sx,  double* pxd_sy,
                          double* pyd_sx,  double* pyd_sy,
                          int nx, int ny, double dx_u, double dy_u,
                          int limiter_id, int bc_mode) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nx*ny) return;
    int ic = flat / ny, jc = flat % ny;
    bool x_per = (bc_mode & 1) != 0;
    bool y_per = (bc_mode & 2) != 0;

    // Eq. (16.3): limited left/right slope using minmod / van Leer / MC.
    auto slopes = [&](const double* f, double& sx, double& sy) {
        double fc = f[flat];
        bool has_left  = (ic > 0) || x_per;
        bool has_right = (ic < nx - 1) || x_per;
        if (has_left && has_right) {
            int icL = (ic == 0)        ? nx - 1 : ic - 1;
            int icR = (ic == nx - 1)   ? 0      : ic + 1;
            double dfL = (fc - f[icL*ny + jc]) / dx_u;
            double dfR = (f[icR*ny + jc] - fc) / dx_u;
            sx = apply_limiter(dfL, dfR, limiter_id);
        } else {
            sx = 0.0;
        }
        bool has_down = (jc > 0) || y_per;
        bool has_up   = (jc < ny - 1) || y_per;
        if (has_down && has_up) {
            int jcD = (jc == 0)      ? ny - 1 : jc - 1;
            int jcU = (jc == ny - 1) ? 0      : jc + 1;
            double dfD = (fc - f[ic*ny + jcD]) / dy_u;
            double dfU = (f[ic*ny + jcU] - fc) / dy_u;
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
void k_cale2_remap_east_2nd(const double* X0, const double* Y0,
                           const double* X,  const double* Y,
                           const double* rho_d,  const double* rhoE_d,
                           const double* pxd,    const double* pyd,
                           const double* rho_sx, const double* rho_sy,
                           const double* rhoE_sx,const double* rhoE_sy,
                           const double* pxd_sx, const double* pxd_sy,
                           const double* pyd_sx, const double* pyd_sy,
                           double* dm_new, double* ie_new,
                           double* px_new, double* py_new,
                           int nx, int ny, double dx_u, double dy_u,
                           int bc_mode) {
    bool x_per = (bc_mode & 1) != 0;
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int n_edges = x_per ? nx * ny : (nx - 1) * ny;
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
    int cR_idx = ic + 1;
    if (x_per && cR_idx >= nx) cR_idx = 0;
    int cR = calc(cR_idx, jc, ny);
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
    // Periodic wrap correction: if donor was folded via periodic wrap, the
    // swept centroid is on the opposite side of the domain. Subtract the
    // full box width to measure ex relative to the (virtual) unwrapped donor.
    if (x_per) {
        double Lx_full = nx * dx_u;
        if      (ex >  0.5 * Lx_full) ex -= Lx_full;
        else if (ex < -0.5 * Lx_full) ex += Lx_full;
    }

    // Eq. (16.4): MUSCL linear reconstruction at swept-region centroid.
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
void k_cale2_remap_north_2nd(const double* X0, const double* Y0,
                            const double* X,  const double* Y,
                            const double* rho_d,  const double* rhoE_d,
                            const double* pxd,    const double* pyd,
                            const double* rho_sx, const double* rho_sy,
                            const double* rhoE_sx,const double* rhoE_sy,
                            const double* pxd_sx, const double* pxd_sy,
                            const double* pyd_sx, const double* pyd_sy,
                            double* dm_new, double* ie_new,
                            double* px_new, double* py_new,
                            int nx, int ny, double dx_u, double dy_u,
                            int bc_mode) {
    bool y_per = (bc_mode & 2) != 0;
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int ny_eff = y_per ? ny : (ny - 1);
    int n_edges = nx * ny_eff;
    if (flat >= n_edges) return;
    int ic = flat / ny_eff, jc = flat % ny_eff;
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
    int cU_idx = jc + 1;
    if (y_per && cU_idx >= ny) cU_idx = 0;
    int cU = calc(ic, cU_idx, ny);
    int donor = (As > 0.0) ? cD : cU;
    double V_sweep = fabs(As);

    double cx = 0.25 * (Ax + Anx + Bnx + Bx);
    double cy = 0.25 * (Ay + Any + Bny + By);

    int dic = donor / ny, djc = donor % ny;
    double xd = (dic + 0.5) * dx_u;
    double yd = (djc + 0.5) * dy_u;
    double ex = cx - xd, ey = cy - yd;
    if (y_per) {
        double Ly_full = ny * dy_u;
        if      (ey >  0.5 * Ly_full) ey -= Ly_full;
        else if (ey < -0.5 * Ly_full) ey += Ly_full;
    }

    // Eq. (16.4): MUSCL linear reconstruction at swept-region centroid.
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

// ============================================================
// PPM (Colella-Woodward 1984) piecewise-parabolic reconstruction.
//
// Per cell, store left/right face values in x and bottom/top in y.
// The 4-cell centered interpolant:
//   f_{i+1/2} = (7(f_i + f_{i+1}) - (f_{i-1} + f_{i+2})) / 12
// For limiting we have two paths:
//   (a) ppm_monotonize — classical CW84 local-extremum clamp
//       (aggressive; flattens smooth extrema).
//   (b) ppm_cs_limit   — Colella-Sekora (2008) extremum-preserving
//       limiter, per Athena ppm.cpp:138-279. Two-stage: (Stage 1)
//       interface-value correction at smooth extrema using adjacent
//       second-derivative coincidence tests; (Stage 2) parabolic
//       coefficient limiting distinguishing smooth extrema, over-
//       shoot, and roundoff-sensitive cases.
// Direction-split: applied separately in x and y on uniform Cartesian
// mesh. C2=1.25 matches Athena's convention.
// ============================================================

// Eq. (17.2): classical Colella-Woodward 1984 monotonization.
__device__ __forceinline__
void ppm_monotonize(double f_c, double& f_L, double& f_R) {
    double d1 = (f_L - f_c) * (f_c - f_R);
    if (d1 <= 0.0) {
        f_L = f_c;
        f_R = f_c;
        return;
    }
    // Test for over/undershoot.
    double df = f_R - f_L;
    double d6 = 6.0 * (f_c - 0.5 * (f_L + f_R));
    if (df * d6 >  df * df) {
        // clip f_L (cubic would push f below f_L)
        f_L = 3.0 * f_c - 2.0 * f_R;
    } else if (df * d6 < -df * df) {
        // clip f_R (cubic would push f above f_R)
        f_R = 3.0 * f_c - 2.0 * f_L;
    }
}

__device__ __forceinline__
double ppm_cs_sign(double x) { return (x > 0.0) - (x < 0.0); }

// Eq. (17.3-17.5): Colella-Sekora 2008 extremum-preserving PPM limiter
// (two-stage: face-value correction at smooth extrema + parabolic-
// coefficient limiting with smooth/shock discrimination).
// Inputs: 5-point stencil (fm2, fm1, fc, fp1, fp2) plus unlimited face
// reconstructions fL at i-½ and fR at i+½. Output: (fL, fR) limited.
// Mirrors Athena uniform-mesh path (ppm.cpp:138-279).
__device__ __forceinline__
void ppm_cs_limit(double fm2, double fm1, double fc, double fp1, double fp2,
                  double& fL, double& fR) {
    const double C2 = 1.25;

    // Second-derivative stencils at cell centers.
    double d2_m1 = fm2 + fc  - 2.0 * fm1;   // d2qc_im1
    double d2_c  = fm1 + fp1 - 2.0 * fc;    // d2qc
    double d2_p1 = fc  + fp2 - 2.0 * fp1;   // d2qc_ip1

    // ---- Stage 1a: correct fL (i-½ face) if it violates monotonicity ----
    {
        double qa_tmp = fL  - fm1;    // CD 84a: face - left cell avg
        double qb_tmp = fc  - fL;     // CD 84b: right cell avg - face
        if (qa_tmp * qb_tmp < 0.0) {
            double qa = 3.0 * (fm1 + fc - 2.0 * fL);  // CD 85b: face curvature
            double qb = d2_m1;                         // CD 85a
            double qc = d2_c;                          // CD 85c
            double qd = 0.0;
            double sa = ppm_cs_sign(qa);
            if (sa == ppm_cs_sign(qb) && sa == ppm_cs_sign(qc)) {
                double aqa = fabs(qa), aqb = fabs(qb), aqc = fabs(qc);
                qd = sa * fmin(C2 * aqb, fmin(C2 * aqc, aqa));
            }
            fL = 0.5 * (fm1 + fc) - qd / 6.0;
        }
    }

    // ---- Stage 1b: correct fR (i+½ face) ----
    {
        double qa_tmp = fR  - fc;
        double qb_tmp = fp1 - fR;
        if (qa_tmp * qb_tmp < 0.0) {
            double qa = 3.0 * (fc + fp1 - 2.0 * fR);
            double qb = d2_c;
            double qc = d2_p1;
            double qd = 0.0;
            double sa = ppm_cs_sign(qa);
            if (sa == ppm_cs_sign(qb) && sa == ppm_cs_sign(qc)) {
                double aqa = fabs(qa), aqb = fabs(qb), aqc = fabs(qc);
                qd = sa * fmin(C2 * aqb, fmin(C2 * aqc, aqa));
            }
            fR = 0.5 * (fc + fp1) - qd / 6.0;
        }
    }

    // ---- Stage 2: parabolic coefficient limiting (CS eq 22-23) ----
    double dqf_m = fc - fL;
    double dqf_p = fR - fc;
    double qa_tmp = dqf_m * dqf_p;              // <0 iff local extremum inside cell
    double qb_tmp = (fp1 - fc) * (fc - fm1);    // <0 iff local extremum in 3-pt avg
    double d2qf = 6.0 * (fL + fR - 2.0 * fc);   // -2·a6 of limited parabola

    double qe = 0.0;
    double sd = ppm_cs_sign(d2qf);
    if (sd == ppm_cs_sign(d2_m1) && sd == ppm_cs_sign(d2_c) && sd == ppm_cs_sign(d2_p1)) {
        double aqa = fabs(d2_m1), aqb = fabs(d2_c), aqc = fabs(d2_p1), aqd = fabs(d2qf);
        qe = sd * fmin(fmin(C2 * aqa, C2 * aqb), fmin(C2 * aqc, aqd));
    }

    // Roundoff sensitivity guard.
    double rnorm_a = fmax(fabs(fm1), fabs(fm2));
    double rnorm_b = fmax(fmax(fabs(fc), fabs(fp1)), fabs(fp2));
    double rho = 0.0;
    if (fabs(d2qf) > 1.0e-12 * fmax(rnorm_a, rnorm_b)) {
        rho = qe / d2qf;
    }

    if (qa_tmp <= 0.0 || qb_tmp <= 0.0) {
        // Smooth extremum: scale parabola by rho (CS eq 23). If rho≈1 leave alone.
        if (rho <= 1.0 - 1.0e-12) {
            fL = fc - rho * dqf_m;
            fR = fc + rho * dqf_p;
        }
    } else {
        // No extremum: classical CW overshoot clamp.
        if (fabs(dqf_m) >= 2.0 * fabs(dqf_p)) {
            fL = fc - 2.0 * dqf_p;
        }
        if (fabs(dqf_p) >= 2.0 * fabs(dqf_m)) {
            fR = fc + 2.0 * dqf_m;
        }
    }
}

// ============================================================
// Characteristic-variable PPM (Athena ppm.cpp + characteristic.cpp).
//
// For each cell i we project its 5-point primitive stencil
//   W_k = (ρ, P, vx, vy)_k,  k ∈ {i-2, i-1, i, i+1, i+2}
// into the local characteristic basis using cell i's (ρ_i, a_i),
// where a = √(γ P_i / ρ_i). The 4 wave modes are
//   w0 = ½(P/a² − ρ·vx/a)          left-going acoustic
//   w1 = ρ − P/a²                   entropy
//   w2 = vy                         shear
//   w3 = ½(P/a² + ρ·vx/a)          right-going acoustic
// PPM+CS limiting is performed independently on each wave. Then
// face values are projected back to primitives using the same
// (ρ_i, a_i) basis:
//   ρ_f  = w0_f + w1_f + w3_f
//   vx_f = a·(w3_f − w0_f)/ρ_i
//   vy_f = w2_f
//   P_f  = a²·(w0_f + w3_f)
//
// The y-direction sweep uses vy as the sweep-normal velocity:
// swap the roles of vx and vy in the L/R projection.
//
// Reference: Stone, Gardiner, Teuben, Hawley, Simon 2008 Appendix A
// (equations A3/A4). Implementation mirrors Athena's adiabatic hydro
// non-MHD branch in characteristic.cpp (lines 235-256, 470-492).
// ============================================================

// Eq. (17.7): project primitives (ρ, P, vx, vy) → characteristic
// variables (w0 left-acoustic, w1 entropy, w2 shear, w3 right-acoustic),
// using cell-i basis (ρ_i, a_i). Used for x-direction sweep.
__device__ __forceinline__
void char_project_x(double rho_i, double a_i,
                    double drho, double dP, double dvx, double dvy,
                    double& dw0, double& dw1, double& dw2, double& dw3) {
    double asq   = a_i * a_i;
    double dP_a2 = dP / asq;
    double rho_dvx_a = rho_i * dvx / a_i;
    dw0 = 0.5 * (dP_a2 - rho_dvx_a);
    dw1 = drho - dP_a2;
    dw2 = dvy;
    dw3 = 0.5 * (dP_a2 + rho_dvx_a);
}

// Eq. (17.8): project characteristic → primitives back using the same
// cell-i basis.
__device__ __forceinline__
void char_unproject_x(double rho_i, double a_i,
                      double dw0, double dw1, double dw2, double dw3,
                      double& drho, double& dP, double& dvx, double& dvy) {
    double asq = a_i * a_i;
    drho = dw0 + dw1 + dw3;
    dvx  = a_i * (dw3 - dw0) / rho_i;
    dvy  = dw2;
    dP   = asq * (dw0 + dw3);
}

__global__
void k_cale2_ppm_reconstruct_char(const double* rho_d, const double* P_d,
                                  const double* vx_d,  const double* vy_d,
                                  double* rho_xL, double* rho_xR,
                                  double* rho_yD, double* rho_yU,
                                  double* P_xL,   double* P_xR,
                                  double* P_yD,   double* P_yU,
                                  double* vx_xL,  double* vx_xR,
                                  double* vx_yD,  double* vx_yU,
                                  double* vy_xL,  double* vy_xR,
                                  double* vy_yD,  double* vy_yU,
                                  int nx, int ny, int bc_mode, int limiter_mode,
                                  double gam) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nx*ny) return;
    int ic = flat / ny, jc = flat % ny;
    bool x_per = (bc_mode & 1) != 0;
    bool y_per = (bc_mode & 2) != 0;

    auto idx_x = [&](int i) -> int {
        if (x_per) {
            if (i < 0)   i += nx;
            if (i >= nx) i -= nx;
        } else {
            if (i < 0)   i = 0;
            if (i >= nx) i = nx - 1;
        }
        return i * ny + jc;
    };
    auto idx_y = [&](int j) -> int {
        if (y_per) {
            if (j < 0)   j += ny;
            if (j >= ny) j -= ny;
        } else {
            if (j < 0)   j = 0;
            if (j >= ny) j = ny - 1;
        }
        return ic * ny + j;
    };

    // Eq. (17.1): PPM 4-point face interpolant.
    auto interp_face = [](double fm1, double fc, double fp1, double fp2) {
        return (7.0 * (fc + fp1) - (fm1 + fp2)) / 12.0;
    };

    // Cell-i primitive state and local sound speed.
    double rho_i = rho_d[flat];
    double P_i   = P_d[flat];
    double vx_i  = vx_d[flat];
    double vy_i  = vy_d[flat];
    double a_i   = sqrt(fmax(gam * P_i / fmax(rho_i, 1e-30), 1e-30));

    // -------- X direction --------
    {
        // Load 5-point primitive stencil.
        double r[5], p[5], u[5], v[5];
        r[0] = rho_d[idx_x(ic-2)]; r[1] = rho_d[idx_x(ic-1)];
        r[2] = rho_i;              r[3] = rho_d[idx_x(ic+1)];
        r[4] = rho_d[idx_x(ic+2)];
        p[0] = P_d[idx_x(ic-2)];   p[1] = P_d[idx_x(ic-1)];
        p[2] = P_i;                p[3] = P_d[idx_x(ic+1)];
        p[4] = P_d[idx_x(ic+2)];
        u[0] = vx_d[idx_x(ic-2)];  u[1] = vx_d[idx_x(ic-1)];
        u[2] = vx_i;               u[3] = vx_d[idx_x(ic+1)];
        u[4] = vx_d[idx_x(ic+2)];
        v[0] = vy_d[idx_x(ic-2)];  v[1] = vy_d[idx_x(ic-1)];
        v[2] = vy_i;               v[3] = vy_d[idx_x(ic+1)];
        v[4] = vy_d[idx_x(ic+2)];

        // Project each stencil point to characteristic variables using cell-i basis.
        double w[5][4];
        for (int k = 0; k < 5; ++k) {
            char_project_x(rho_i, a_i, r[k], p[k], u[k], v[k],
                           w[k][0], w[k][1], w[k][2], w[k][3]);
        }

        // PPM reconstruct + CS limit each characteristic field independently.
        double wL[4], wR[4];
        for (int n = 0; n < 4; ++n) {
            double fm2 = w[0][n], fm1 = w[1][n], fc = w[2][n];
            double fp1 = w[3][n], fp2 = w[4][n];
            wL[n] = interp_face(fm2, fm1, fc, fp1);
            wR[n] = interp_face(fm1, fc, fp1, fp2);
            if (limiter_mode == 1) ppm_cs_limit(fm2, fm1, fc, fp1, fp2, wL[n], wR[n]);
            else                   ppm_monotonize(fc, wL[n], wR[n]);
        }

        // Project face values back to primitives.
        double rL, pL, uL, vL, rR, pR, uR, vR;
        char_unproject_x(rho_i, a_i, wL[0], wL[1], wL[2], wL[3], rL, pL, uL, vL);
        char_unproject_x(rho_i, a_i, wR[0], wR[1], wR[2], wR[3], rR, pR, uR, vR);

        rho_xL[flat] = rL;  rho_xR[flat] = rR;
        P_xL  [flat] = pL;  P_xR  [flat] = pR;
        vx_xL [flat] = uL;  vx_xR [flat] = uR;
        vy_xL [flat] = vL;  vy_xR [flat] = vR;
    }

    // -------- Y direction (swap roles: vy is the normal velocity) --------
    {
        double r[5], p[5], u[5], v[5];
        r[0] = rho_d[idx_y(jc-2)]; r[1] = rho_d[idx_y(jc-1)];
        r[2] = rho_i;              r[3] = rho_d[idx_y(jc+1)];
        r[4] = rho_d[idx_y(jc+2)];
        p[0] = P_d[idx_y(jc-2)];   p[1] = P_d[idx_y(jc-1)];
        p[2] = P_i;                p[3] = P_d[idx_y(jc+1)];
        p[4] = P_d[idx_y(jc+2)];
        u[0] = vx_d[idx_y(jc-2)];  u[1] = vx_d[idx_y(jc-1)];
        u[2] = vx_i;               u[3] = vx_d[idx_y(jc+1)];
        u[4] = vx_d[idx_y(jc+2)];
        v[0] = vy_d[idx_y(jc-2)];  v[1] = vy_d[idx_y(jc-1)];
        v[2] = vy_i;               v[3] = vy_d[idx_y(jc+1)];
        v[4] = vy_d[idx_y(jc+2)];

        // For y-sweep the "normal" velocity is vy; reuse char_project_x by
        // passing (vy, vx) in place of (vx, vy) — the shear channel just
        // swaps which component is along/transverse.
        double w[5][4];
        for (int k = 0; k < 5; ++k) {
            char_project_x(rho_i, a_i, r[k], p[k], v[k], u[k],
                           w[k][0], w[k][1], w[k][2], w[k][3]);
        }

        double wL[4], wR[4];
        for (int n = 0; n < 4; ++n) {
            double fm2 = w[0][n], fm1 = w[1][n], fc = w[2][n];
            double fp1 = w[3][n], fp2 = w[4][n];
            wL[n] = interp_face(fm2, fm1, fc, fp1);
            wR[n] = interp_face(fm1, fc, fp1, fp2);
            if (limiter_mode == 1) ppm_cs_limit(fm2, fm1, fc, fp1, fp2, wL[n], wR[n]);
            else                   ppm_monotonize(fc, wL[n], wR[n]);
        }

        double rL, pL, VyL, VxL, rR, pR, VyR, VxR;
        char_unproject_x(rho_i, a_i, wL[0], wL[1], wL[2], wL[3], rL, pL, VyL, VxL);
        char_unproject_x(rho_i, a_i, wR[0], wR[1], wR[2], wR[3], rR, pR, VyR, VxR);

        rho_yD[flat] = rL;  rho_yU[flat] = rR;
        P_yD  [flat] = pL;  P_yU  [flat] = pR;
        vx_yD [flat] = VxL; vx_yU [flat] = VxR;
        vy_yD [flat] = VyL; vy_yU [flat] = VyR;
    }
}

__global__
void k_cale2_ppm_reconstruct(const double* rho_d, const double* rhoE_d,
                             const double* pxd,   const double* pyd,
                             double* rho_xL, double* rho_xR,
                             double* rho_yD, double* rho_yU,
                             double* rhoE_xL, double* rhoE_xR,
                             double* rhoE_yD, double* rhoE_yU,
                             double* pxd_xL, double* pxd_xR,
                             double* pxd_yD, double* pxd_yU,
                             double* pyd_xL, double* pyd_xR,
                             double* pyd_yD, double* pyd_yU,
                             int nx, int ny, int bc_mode, int limiter_mode) {
    // limiter_mode: 0 = classical Colella-Woodward (ppm_monotonize),
    //               1 = Colella-Sekora extremum-preserving (ppm_cs_limit).
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nx*ny) return;
    int ic = flat / ny, jc = flat % ny;
    bool x_per = (bc_mode & 1) != 0;
    bool y_per = (bc_mode & 2) != 0;

    auto idx_x = [&](int i) -> int {
        if (x_per) {
            if (i < 0)   i += nx;
            if (i >= nx) i -= nx;
        } else {
            if (i < 0)   i = 0;
            if (i >= nx) i = nx - 1;
        }
        return i * ny + jc;
    };
    auto idx_y = [&](int j) -> int {
        if (y_per) {
            if (j < 0)   j += ny;
            if (j >= ny) j -= ny;
        } else {
            if (j < 0)   j = 0;
            if (j >= ny) j = ny - 1;
        }
        return ic * ny + j;
    };

    // Interpolate f at face i+1/2 given f_{i-1}, f_i, f_{i+1}, f_{i+2}.
    // Eq. (17.1): PPM 4-point face interpolant.
    auto interp_face = [](double fm1, double fc, double fp1, double fp2) {
        return (7.0 * (fc + fp1) - (fm1 + fp2)) / 12.0;
    };

    auto do_x = [&](const double* f, double& fL, double& fR) {
        double fm2 = f[idx_x(ic-2)];
        double fm1 = f[idx_x(ic-1)];
        double fc  = f[flat];
        double fp1 = f[idx_x(ic+1)];
        double fp2 = f[idx_x(ic+2)];
        // left face: interp at i-1/2 using (i-2, i-1, i, i+1)
        fL = interp_face(fm2, fm1, fc, fp1);
        // right face: interp at i+1/2 using (i-1, i, i+1, i+2)
        fR = interp_face(fm1, fc, fp1, fp2);
        if (limiter_mode == 1) ppm_cs_limit(fm2, fm1, fc, fp1, fp2, fL, fR);
        else                   ppm_monotonize(fc, fL, fR);
    };
    auto do_y = [&](const double* f, double& fD, double& fU) {
        double fm2 = f[idx_y(jc-2)];
        double fm1 = f[idx_y(jc-1)];
        double fc  = f[flat];
        double fp1 = f[idx_y(jc+1)];
        double fp2 = f[idx_y(jc+2)];
        fD = interp_face(fm2, fm1, fc, fp1);
        fU = interp_face(fm1, fc, fp1, fp2);
        if (limiter_mode == 1) ppm_cs_limit(fm2, fm1, fc, fp1, fp2, fD, fU);
        else                   ppm_monotonize(fc, fD, fU);
    };

    do_x(rho_d,  rho_xL[flat],  rho_xR[flat]);
    do_y(rho_d,  rho_yD[flat],  rho_yU[flat]);
    do_x(rhoE_d, rhoE_xL[flat], rhoE_xR[flat]);
    do_y(rhoE_d, rhoE_yD[flat], rhoE_yU[flat]);
    do_x(pxd,    pxd_xL[flat],  pxd_xR[flat]);
    do_y(pxd,    pxd_yD[flat],  pxd_yU[flat]);
    do_x(pyd,    pyd_xL[flat],  pyd_xR[flat]);
    do_y(pyd,    pyd_yD[flat],  pyd_yU[flat]);
}

// Point-evaluate PPM parabolic profile inside a donor cell at relative
// position (sx, sy) ∈ [-1/2, 1/2]² from the cell centroid.
// For direction-split PPM we use:
//   f(sx, sy) = f_bar + s_x·Δ_x + x_6·(1/4 − sx²)   (x part)
//             + s_y·Δ_y + y_6·(1/4 − sy²)            (y part — subtract f_bar)
// where
//   Δ  = (f_R − f_L)                 → slope = (f_R - f_L)
//   f_6 = 6(f_bar − ½(f_L + f_R))    → curvature correction
// Note that adding x-parabola and y-parabola independently and dropping one
// f_bar gives the separable approximation used here.
__device__ __forceinline__
double ppm_eval(double f_bar, double fL, double fR, double s /* ξ - ½, in [-½, ½] */) {
    // Standard PPM parabolic profile at ξ = s + ½ ∈ [0,1]:
    //   f(ξ) = f_L + ξ(Δf + f_6·(1−ξ))
    //        = f_L + ξ·Δf + ξ(1−ξ)·f_6
    // with Δf = f_R − f_L, f_6 = 6·(f_bar − ½(f_L+f_R)).
    double xi = s + 0.5;
    double df = fR - fL;
    double f6 = 6.0 * (f_bar - 0.5 * (fL + fR));
    return fL + xi * df + xi * (1.0 - xi) * f6;
}

__global__
void k_cale2_remap_east_ppm(const double* X0, const double* Y0,
                            const double* X,  const double* Y,
                            const double* rho_d,  const double* rhoE_d,
                            const double* pxd,    const double* pyd,
                            const double* rho_xL, const double* rho_xR,
                            const double* rho_yD, const double* rho_yU,
                            const double* rhoE_xL,const double* rhoE_xR,
                            const double* rhoE_yD,const double* rhoE_yU,
                            const double* pxd_xL, const double* pxd_xR,
                            const double* pxd_yD, const double* pxd_yU,
                            const double* pyd_xL, const double* pyd_xR,
                            const double* pyd_yD, const double* pyd_yU,
                            double* dm_new, double* ie_new,
                            double* px_new, double* py_new,
                            int nx, int ny, double dx_u, double dy_u,
                            int bc_mode) {
    bool x_per = (bc_mode & 1) != 0;
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int n_edges = x_per ? nx * ny : (nx - 1) * ny;
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
    int cR_idx = ic + 1;
    if (x_per && cR_idx >= nx) cR_idx = 0;
    int cR = calc(cR_idx, jc, ny);
    int donor = (As > 0.0) ? cL : cR;
    double V_sweep = fabs(As);

    double cx = 0.25 * (Ax + Anx + Bnx + Bx);
    double cy = 0.25 * (Ay + Any + Bny + By);
    int dic = donor / ny, djc = donor % ny;
    double xd = (dic + 0.5) * dx_u;
    double yd = (djc + 0.5) * dy_u;
    double sx = (cx - xd) / dx_u;  // normalised ∈ [-½, ½]
    double sy = (cy - yd) / dy_u;
    // Wrap sx/sy if donor was folded via periodic BC — swept quad ends up
    // near domain boundary, unwrap relative to donor centroid.
    if (x_per) {
        if (sx >  0.5) sx -= nx;   // -0.5 after wrap
        if (sx < -0.5) sx += nx;
    }
    // Clamp in case of slight overshoot (should be tiny for Eulerian rezone).
    if (sx < -0.5) sx = -0.5; else if (sx > 0.5) sx = 0.5;
    if (sy < -0.5) sy = -0.5; else if (sy > 0.5) sy = 0.5;

    // Direction-split PPM evaluation: x-parabola + y-parabola − f_bar.
    auto eval_2d = [](double f_bar, double fL, double fR,
                      double fD, double fU, double sx, double sy) {
        double fx = ppm_eval(f_bar, fL, fR, sx);
        double fy = ppm_eval(f_bar, fD, fU, sy);
        return fx + fy - f_bar;
    };

    double d_dm = eval_2d(rho_d[donor],  rho_xL[donor],  rho_xR[donor],
                          rho_yD[donor], rho_yU[donor], sx, sy) * V_sweep;
    double d_ie = eval_2d(rhoE_d[donor], rhoE_xL[donor], rhoE_xR[donor],
                          rhoE_yD[donor], rhoE_yU[donor], sx, sy) * V_sweep;
    double d_px = eval_2d(pxd[donor],    pxd_xL[donor],  pxd_xR[donor],
                          pxd_yD[donor], pxd_yU[donor], sx, sy) * V_sweep;
    double d_py = eval_2d(pyd[donor],    pyd_xL[donor],  pyd_xR[donor],
                          pyd_yD[donor], pyd_yU[donor], sx, sy) * V_sweep;

    if (As > 0.0) {
        atomicAdd(&dm_new[cL], -d_dm);  atomicAdd(&ie_new[cL], -d_ie);
        atomicAdd(&px_new[cL], -d_px);  atomicAdd(&py_new[cL], -d_py);
        atomicAdd(&dm_new[cR],  d_dm);  atomicAdd(&ie_new[cR],  d_ie);
        atomicAdd(&px_new[cR],  d_px);  atomicAdd(&py_new[cR],  d_py);
    } else {
        atomicAdd(&dm_new[cR], -d_dm);  atomicAdd(&ie_new[cR], -d_ie);
        atomicAdd(&px_new[cR], -d_px);  atomicAdd(&py_new[cR], -d_py);
        atomicAdd(&dm_new[cL],  d_dm);  atomicAdd(&ie_new[cL],  d_ie);
        atomicAdd(&px_new[cL],  d_px);  atomicAdd(&py_new[cL],  d_py);
    }
}

__global__
void k_cale2_remap_north_ppm(const double* X0, const double* Y0,
                             const double* X,  const double* Y,
                             const double* rho_d,  const double* rhoE_d,
                             const double* pxd,    const double* pyd,
                             const double* rho_xL, const double* rho_xR,
                             const double* rho_yD, const double* rho_yU,
                             const double* rhoE_xL,const double* rhoE_xR,
                             const double* rhoE_yD,const double* rhoE_yU,
                             const double* pxd_xL, const double* pxd_xR,
                             const double* pxd_yD, const double* pxd_yU,
                             const double* pyd_xL, const double* pyd_xR,
                             const double* pyd_yD, const double* pyd_yU,
                             double* dm_new, double* ie_new,
                             double* px_new, double* py_new,
                             int nx, int ny, double dx_u, double dy_u,
                             int bc_mode) {
    bool y_per = (bc_mode & 2) != 0;
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int ny_eff = y_per ? ny : (ny - 1);
    int n_edges = nx * ny_eff;
    if (flat >= n_edges) return;
    int ic = flat / ny_eff, jc = flat % ny_eff;
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
    int cU_idx = jc + 1;
    if (y_per && cU_idx >= ny) cU_idx = 0;
    int cU = calc(ic, cU_idx, ny);
    int donor = (As > 0.0) ? cD : cU;
    double V_sweep = fabs(As);

    double cx = 0.25 * (Ax + Anx + Bnx + Bx);
    double cy = 0.25 * (Ay + Any + Bny + By);
    int dic = donor / ny, djc = donor % ny;
    double xd = (dic + 0.5) * dx_u;
    double yd = (djc + 0.5) * dy_u;
    double sx = (cx - xd) / dx_u;
    double sy = (cy - yd) / dy_u;
    if (y_per) {
        if (sy >  0.5) sy -= ny;
        if (sy < -0.5) sy += ny;
    }
    if (sx < -0.5) sx = -0.5; else if (sx > 0.5) sx = 0.5;
    if (sy < -0.5) sy = -0.5; else if (sy > 0.5) sy = 0.5;

    auto eval_2d = [](double f_bar, double fL, double fR,
                      double fD, double fU, double sx, double sy) {
        double fx = ppm_eval(f_bar, fL, fR, sx);
        double fy = ppm_eval(f_bar, fD, fU, sy);
        return fx + fy - f_bar;
    };

    double d_dm = eval_2d(rho_d[donor],  rho_xL[donor],  rho_xR[donor],
                          rho_yD[donor], rho_yU[donor], sx, sy) * V_sweep;
    double d_ie = eval_2d(rhoE_d[donor], rhoE_xL[donor], rhoE_xR[donor],
                          rhoE_yD[donor], rhoE_yU[donor], sx, sy) * V_sweep;
    double d_px = eval_2d(pxd[donor],    pxd_xL[donor],  pxd_xR[donor],
                          pxd_yD[donor], pxd_yU[donor], sx, sy) * V_sweep;
    double d_py = eval_2d(pyd[donor],    pyd_xL[donor],  pyd_xR[donor],
                          pyd_yD[donor], pyd_yU[donor], sx, sy) * V_sweep;

    if (As > 0.0) {
        atomicAdd(&dm_new[cD], -d_dm);  atomicAdd(&ie_new[cD], -d_ie);
        atomicAdd(&px_new[cD], -d_px);  atomicAdd(&py_new[cD], -d_py);
        atomicAdd(&dm_new[cU],  d_dm);  atomicAdd(&ie_new[cU],  d_ie);
        atomicAdd(&px_new[cU],  d_px);  atomicAdd(&py_new[cU],  d_py);
    } else {
        atomicAdd(&dm_new[cU], -d_dm);  atomicAdd(&ie_new[cU], -d_ie);
        atomicAdd(&px_new[cU], -d_px);  atomicAdd(&py_new[cU], -d_py);
        atomicAdd(&dm_new[cD],  d_dm);  atomicAdd(&ie_new[cD],  d_ie);
        atomicAdd(&px_new[cD],  d_px);  atomicAdd(&py_new[cD],  d_py);
    }
}

// ============================================================
// Primitive-space PPM remap. Input field buffers have semantics:
//   rho_d  = ρ   ;  rhoE_d = P  ;  pxd = vx  ;  pyd = vy   (per-cell averages)
//   *_xL/xR/yD/yU are face reconstructions of those same primitives.
// At the swept centroid we evaluate (ρ̂, P̂, v̂x, v̂y) and convert to
// conserved fluxes:
//     d_dm = ρ̂·V_sweep
//     d_px = ρ̂·v̂x·V_sweep
//     d_py = ρ̂·v̂y·V_sweep
//     d_ie = P̂·V_sweep / (γ-1)        (internal energy only, no KE term)
// Donor-cell subtracts, acceptor-cell adds — discrete conservation is
// exact because both sides reference the same flux value.
// ============================================================

__global__
void k_cale2_remap_east_ppm_prim(const double* X0, const double* Y0,
                                 const double* X,  const double* Y,
                                 const double* rho_d,  const double* P_d,
                                 const double* vx_d,   const double* vy_d,
                                 const double* rho_xL, const double* rho_xR,
                                 const double* rho_yD, const double* rho_yU,
                                 const double* P_xL,   const double* P_xR,
                                 const double* P_yD,   const double* P_yU,
                                 const double* vx_xL,  const double* vx_xR,
                                 const double* vx_yD,  const double* vx_yU,
                                 const double* vy_xL,  const double* vy_xR,
                                 const double* vy_yD,  const double* vy_yU,
                                 double* dm_new, double* ie_new,
                                 double* px_new, double* py_new,
                                 int nx, int ny, double dx_u, double dy_u,
                                 double gam, int bc_mode) {
    bool x_per = (bc_mode & 1) != 0;
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int n_edges = x_per ? nx * ny : (nx - 1) * ny;
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
    int cR_idx = ic + 1;
    if (x_per && cR_idx >= nx) cR_idx = 0;
    int cR = calc(cR_idx, jc, ny);
    int donor = (As > 0.0) ? cL : cR;
    double V_sweep = fabs(As);

    double cx = 0.25 * (Ax + Anx + Bnx + Bx);
    double cy = 0.25 * (Ay + Any + Bny + By);
    int dic = donor / ny, djc = donor % ny;
    double xd = (dic + 0.5) * dx_u;
    double yd = (djc + 0.5) * dy_u;
    double sx = (cx - xd) / dx_u;
    double sy = (cy - yd) / dy_u;
    if (x_per) {
        if (sx >  0.5) sx -= nx;
        if (sx < -0.5) sx += nx;
    }
    if (sx < -0.5) sx = -0.5; else if (sx > 0.5) sx = 0.5;
    if (sy < -0.5) sy = -0.5; else if (sy > 0.5) sy = 0.5;

    auto eval_2d = [](double f_bar, double fL, double fR,
                      double fD, double fU, double sx, double sy) {
        double fx = ppm_eval(f_bar, fL, fR, sx);
        double fy = ppm_eval(f_bar, fD, fU, sy);
        return fx + fy - f_bar;
    };

    double rho_face = eval_2d(rho_d[donor], rho_xL[donor], rho_xR[donor],
                              rho_yD[donor], rho_yU[donor], sx, sy);
    double P_face   = eval_2d(P_d[donor],   P_xL[donor],   P_xR[donor],
                              P_yD[donor],   P_yU[donor],   sx, sy);
    double vx_face  = eval_2d(vx_d[donor],  vx_xL[donor],  vx_xR[donor],
                              vx_yD[donor],  vx_yU[donor],  sx, sy);
    double vy_face  = eval_2d(vy_d[donor],  vy_xL[donor],  vy_xR[donor],
                              vy_yD[donor],  vy_yU[donor],  sx, sy);

    // Positivity safety: clamp ρ, P against non-positive values (PPM
    // in primitive space is robust, but machine-precision overshoot is
    // still possible at sharp interfaces).
    rho_face = fmax(rho_face, 1e-30);
    P_face   = fmax(P_face,   1e-30);

    // Eq. (17.6): primitive → conservative flux at swept centroid.
    double d_dm = rho_face * V_sweep;
    double d_px = d_dm * vx_face;
    double d_py = d_dm * vy_face;
    double d_ie = P_face * V_sweep / (gam - 1.0);

    if (As > 0.0) {
        atomicAdd(&dm_new[cL], -d_dm);  atomicAdd(&ie_new[cL], -d_ie);
        atomicAdd(&px_new[cL], -d_px);  atomicAdd(&py_new[cL], -d_py);
        atomicAdd(&dm_new[cR],  d_dm);  atomicAdd(&ie_new[cR],  d_ie);
        atomicAdd(&px_new[cR],  d_px);  atomicAdd(&py_new[cR],  d_py);
    } else {
        atomicAdd(&dm_new[cR], -d_dm);  atomicAdd(&ie_new[cR], -d_ie);
        atomicAdd(&px_new[cR], -d_px);  atomicAdd(&py_new[cR], -d_py);
        atomicAdd(&dm_new[cL],  d_dm);  atomicAdd(&ie_new[cL],  d_ie);
        atomicAdd(&px_new[cL],  d_px);  atomicAdd(&py_new[cL],  d_py);
    }
}

__global__
void k_cale2_remap_north_ppm_prim(const double* X0, const double* Y0,
                                  const double* X,  const double* Y,
                                  const double* rho_d,  const double* P_d,
                                  const double* vx_d,   const double* vy_d,
                                  const double* rho_xL, const double* rho_xR,
                                  const double* rho_yD, const double* rho_yU,
                                  const double* P_xL,   const double* P_xR,
                                  const double* P_yD,   const double* P_yU,
                                  const double* vx_xL,  const double* vx_xR,
                                  const double* vx_yD,  const double* vx_yU,
                                  const double* vy_xL,  const double* vy_xR,
                                  const double* vy_yD,  const double* vy_yU,
                                  double* dm_new, double* ie_new,
                                  double* px_new, double* py_new,
                                  int nx, int ny, double dx_u, double dy_u,
                                  double gam, int bc_mode) {
    bool y_per = (bc_mode & 2) != 0;
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int ny_eff = y_per ? ny : (ny - 1);
    int n_edges = nx * ny_eff;
    if (flat >= n_edges) return;
    int ic = flat / ny_eff, jc = flat % ny_eff;
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
    int cU_idx = jc + 1;
    if (y_per && cU_idx >= ny) cU_idx = 0;
    int cU = calc(ic, cU_idx, ny);
    int donor = (As > 0.0) ? cD : cU;
    double V_sweep = fabs(As);

    double cx = 0.25 * (Ax + Anx + Bnx + Bx);
    double cy = 0.25 * (Ay + Any + Bny + By);
    int dic = donor / ny, djc = donor % ny;
    double xd = (dic + 0.5) * dx_u;
    double yd = (djc + 0.5) * dy_u;
    double sx = (cx - xd) / dx_u;
    double sy = (cy - yd) / dy_u;
    if (y_per) {
        if (sy >  0.5) sy -= ny;
        if (sy < -0.5) sy += ny;
    }
    if (sx < -0.5) sx = -0.5; else if (sx > 0.5) sx = 0.5;
    if (sy < -0.5) sy = -0.5; else if (sy > 0.5) sy = 0.5;

    auto eval_2d = [](double f_bar, double fL, double fR,
                      double fD, double fU, double sx, double sy) {
        double fx = ppm_eval(f_bar, fL, fR, sx);
        double fy = ppm_eval(f_bar, fD, fU, sy);
        return fx + fy - f_bar;
    };

    double rho_face = eval_2d(rho_d[donor], rho_xL[donor], rho_xR[donor],
                              rho_yD[donor], rho_yU[donor], sx, sy);
    double P_face   = eval_2d(P_d[donor],   P_xL[donor],   P_xR[donor],
                              P_yD[donor],   P_yU[donor],   sx, sy);
    double vx_face  = eval_2d(vx_d[donor],  vx_xL[donor],  vx_xR[donor],
                              vx_yD[donor],  vx_yU[donor],  sx, sy);
    double vy_face  = eval_2d(vy_d[donor],  vy_xL[donor],  vy_xR[donor],
                              vy_yD[donor],  vy_yU[donor],  sx, sy);

    rho_face = fmax(rho_face, 1e-30);
    P_face   = fmax(P_face,   1e-30);

    // Eq. (17.6): primitive → conservative flux at swept centroid.
    double d_dm = rho_face * V_sweep;
    double d_px = d_dm * vx_face;
    double d_py = d_dm * vy_face;
    double d_ie = P_face * V_sweep / (gam - 1.0);

    if (As > 0.0) {
        atomicAdd(&dm_new[cD], -d_dm);  atomicAdd(&ie_new[cD], -d_ie);
        atomicAdd(&px_new[cD], -d_px);  atomicAdd(&py_new[cD], -d_py);
        atomicAdd(&dm_new[cU],  d_dm);  atomicAdd(&ie_new[cU],  d_ie);
        atomicAdd(&px_new[cU],  d_px);  atomicAdd(&py_new[cU],  d_py);
    } else {
        atomicAdd(&dm_new[cU], -d_dm);  atomicAdd(&ie_new[cU], -d_ie);
        atomicAdd(&px_new[cU], -d_px);  atomicAdd(&py_new[cU], -d_py);
        atomicAdd(&dm_new[cD],  d_dm);  atomicAdd(&ie_new[cD],  d_ie);
        atomicAdd(&px_new[cD],  d_px);  atomicAdd(&py_new[cD],  d_py);
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
void k_cale2_snapshot(const double* rho, const double* P, const double* e_int,
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

// Periodic sync of node-centered fields across x and/or y boundaries.
// Two-phase, race-free:
//   phase 1: reduce (sum) all wrapping partners into a canonical master
//   phase 2: broadcast master value back to all partners
// Done via two sequential kernels to avoid atomic races.
//
// Master convention: node with the SMALLEST (in, jn) index in its periodic
// equivalence class. For x-only periodic:
//   (0, j)  is master of (nnx-1, j)
// For y-only periodic:
//   (i, 0)  is master of (i, nny-1)
// For both periodic:
//   (0, 0)  is master of (0, nny-1), (nnx-1, 0), (nnx-1, nny-1)
//
// Simplification: we just AVERAGE the partner values and write back to
// BOTH sides, but only the "master" thread does the write. This
// eliminates write races because each non-master partner is never the
// source of a write — only the master computes and writes to all copies.
//
// mode = 0: copy — writes (average of partners) to all partners. Correct for
//          STATE variables (velocity, positions): periodic duplicates must
//          hold the same value, and averaging cancels FP roundoff between
//          partners that were independently rebuilt.
// mode = 1: sum — writes (sum of partners) to all partners. Correct for
//          FORCE and MASS-like accumulators: each ghost copy only received
//          partial contributions from its local-side cells, so the full
//          physical quantity is the sum across partners. After sync, all
//          partners hold the full quantity (so that F/m in node_update is
//          computed correctly and symmetrically).
__global__
void k_cale2_periodic_sync_node(double* fX, double* fY,
                               int nnx, int nny, int bc_mode, int mode) {
    bool x_per = (bc_mode & 1) != 0;
    bool y_per = (bc_mode & 2) != 0;
    if (!x_per && !y_per) return;

    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nnx * nny;
    if (flat >= n) return;
    int in = flat / nny, jn = flat % nny;

    // Node is a MASTER iff it is NOT a periodic duplicate. Duplicates are
    // at in=nnx-1 (under x-periodic) or jn=nny-1 (under y-periodic); both
    // edges are duplicates under x+y periodic.
    bool x_dup = x_per && (in == nnx - 1);
    bool y_dup = y_per && (jn == nny - 1);
    if (x_dup || y_dup) return;

    // Identify partners in the equivalence class (self + wrap copies).
    int partners[4];
    int np = 0;
    partners[np++] = in * nny + jn;                          // self
    if (x_per && in == 0)
        partners[np++] = (nnx - 1) * nny + jn;               // east duplicate
    if (y_per && jn == 0)
        partners[np++] = in * nny + (nny - 1);               // north duplicate
    if (x_per && y_per && in == 0 && jn == 0)
        partners[np++] = (nnx - 1) * nny + (nny - 1);        // far corner duplicate

    double sx = 0.0, sy = 0.0;
    for (int k = 0; k < np; ++k) { sx += fX[partners[k]]; sy += fY[partners[k]]; }
    if (mode == 0) { sx /= np; sy /= np; }
    for (int k = 0; k < np; ++k) { fX[partners[k]] = sx; fY[partners[k]] = sy; }
}

// Convenience: scalar-field sync (e.g. mass). Same mode semantics as the
// 2-component version above.
__global__
void k_cale2_periodic_sync_scalar(double* f, int nnx, int nny, int bc_mode, int mode) {
    bool x_per = (bc_mode & 1) != 0;
    bool y_per = (bc_mode & 2) != 0;
    if (!x_per && !y_per) return;

    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nnx * nny;
    if (flat >= n) return;
    int in = flat / nny, jn = flat % nny;

    bool x_master = !x_per || (in == 0);
    bool y_master = !y_per || (jn == 0);
    if (!(x_master && y_master)) return;

    int partners[4];
    int np = 0;
    partners[np++] = in * nny + jn;
    if (x_per && in == 0)
        partners[np++] = (nnx - 1) * nny + jn;
    if (y_per && jn == 0)
        partners[np++] = in * nny + (nny - 1);
    if (x_per && y_per && in == 0 && jn == 0)
        partners[np++] = (nnx - 1) * nny + (nny - 1);

    double s = 0.0;
    for (int k = 0; k < np; ++k) s += f[partners[k]];
    if (mode == 0) s /= np;
    for (int k = 0; k < np; ++k) f[partners[k]] = s;
}

// Clamp edge-aligned node velocities AFTER rebuild (reflective walls only).
// bc_mode: see k_cale2_bc_reflective.
__global__
void k_cale2_bc_velocity(double* vX, double* vY, int nnx, int nny, int bc_mode) {
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int n = nnx * nny;
    if (flat >= n) return;
    int in = flat / nny, jn = flat % nny;
    bool x_per = (bc_mode & 1) != 0;
    bool y_per = (bc_mode & 2) != 0;
    if (!x_per && (in == 0 || in == nnx - 1)) vX[flat] = 0.0;
    if (!y_per && (jn == 0 || jn == nny - 1)) vY[flat] = 0.0;
}

// ============================================================
// Passive species tracer X ∈ [0, 1]
// Remap conserves species-mass  mX = X·dm  via donor-cell swept flux.
// Same swept-quad geometry as k_cale2_remap_east/north; we only carry
// ONE extra scalar so the kernels are small.
// ============================================================
// swept_quad_signed() is the __device__ __forceinline__ helper defined
// earlier in this TU (near k_cale2_remap_east); it's already visible.

__global__
void k_cale2_species_init_scratch(const double* mX, double* mX_new, int ncell) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncell) return;
    mX_new[c] = mX[c];
}

__global__
void k_cale2_species_remap_east(const double* X0, const double* Y0,
                                const double* X,  const double* Y,
                                const double* mX, const double* Vol0,
                                double* mX_new,
                                int nx, int ny, int bc_mode) {
    bool x_per = (bc_mode & 1) != 0;
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int n_edges = x_per ? nx * ny : (nx - 1) * ny;
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
    int cL = calc(ic, jc, ny);
    int cR_idx = ic + 1;
    if (x_per && cR_idx >= nx) cR_idx = 0;
    int cR = calc(cR_idx, jc, ny);
    int donor = (As > 0.0) ? cL : cR;
    double V_sweep = fabs(As);
    double V_donor = fmax(Vol0[donor], 1e-30);
    double frac = fmin(V_sweep / V_donor, 0.5);
    double V = frac * V_donor;
    double d_mX = (mX[donor] / V_donor) * V;
    if (As > 0.0) {
        atomicAdd(&mX_new[cL], -d_mX);
        atomicAdd(&mX_new[cR],  d_mX);
    } else {
        atomicAdd(&mX_new[cR], -d_mX);
        atomicAdd(&mX_new[cL],  d_mX);
    }
}

__global__
void k_cale2_species_remap_north(const double* X0, const double* Y0,
                                 const double* X,  const double* Y,
                                 const double* mX, const double* Vol0,
                                 double* mX_new,
                                 int nx, int ny, int bc_mode) {
    bool y_per = (bc_mode & 2) != 0;
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int ny_eff = y_per ? ny : (ny - 1);
    int n_edges = nx * ny_eff;
    if (flat >= n_edges) return;
    int ic = flat / ny_eff, jc = flat % ny_eff;
    int nny = ny + 1;
    int nW = caln(ic,   jc+1, nny);
    int nE = caln(ic+1, jc+1, nny);
    double Ax = X0[nE], Ay = Y0[nE];
    double Anx = X[nE],  Any = Y[nE];
    double Bx = X0[nW], By = Y0[nW];
    double Bnx = X[nW],  Bny = Y[nW];
    double As = swept_quad_signed(Ax, Ay, Anx, Any, Bnx, Bny, Bx, By);
    if (As == 0.0) return;
    int cD = calc(ic, jc, ny);
    int cU_idx = jc + 1;
    if (y_per && cU_idx >= ny) cU_idx = 0;
    int cU = calc(ic, cU_idx, ny);
    int donor = (As > 0.0) ? cD : cU;
    double V_sweep = fabs(As);
    double V_donor = fmax(Vol0[donor], 1e-30);
    double frac = fmin(V_sweep / V_donor, 0.5);
    double V = frac * V_donor;
    double d_mX = (mX[donor] / V_donor) * V;
    if (As > 0.0) {
        atomicAdd(&mX_new[cD], -d_mX);
        atomicAdd(&mX_new[cU],  d_mX);
    } else {
        atomicAdd(&mX_new[cU], -d_mX);
        atomicAdd(&mX_new[cD],  d_mX);
    }
}

// Finalize: mX ← mX_new, X ← mX/dm.  dm is already the post-remap mass.
__global__
void k_cale2_species_finalize(const double* mX_new, const double* dm,
                              double* mX, double* X, int ncell) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncell) return;
    double m = fmax(dm[c], 1e-30);
    mX[c] = mX_new[c];
    X[c]  = fmin(fmax(mX_new[c] / m, 0.0), 1.0);
}
