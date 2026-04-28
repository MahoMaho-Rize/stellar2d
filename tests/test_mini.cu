#include "gpu/lowmach_solver.h"
#include "init/lane_emden.h"
#include <cstdio>
#include <cmath>
#include <vector>
int main() {
    int nr=64, nt=32, ng=2;
    double gamma=5.0/3.0, G=1.0;
    LaneEmdenParams lep; lep.n_poly=1.5; lep.rho_c=1.0; lep.K_poly=1.0; lep.G=G;
    auto sol=solve_lane_emden(lep.n_poly);
    double alpha=std::sqrt((lep.n_poly+1)*lep.K_poly*std::pow(lep.rho_c,1.0/lep.n_poly-1)/(4*M_PI*lep.G));
    Grid grid; grid.init(nr,nt,alpha*sol.xi_1*1.1,2.0);
    EOS eos(gamma);
    State hse; hse.allocate(grid); init_lane_emden(grid,hse,lep,gamma);

    LowMachSolver lm;
    lm.init(grid,eos,G,0.4);
    lm.upload_state(grid,hse);
    
    int tg=(nr+2*ng)*(nt+2*ng);
    std::vector<double> rho_b(tg), rhoE_b(tg);
    cudaMemcpy(rho_b.data(),lm.d_rho,tg*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(rhoE_b.data(),lm.d_rhoE,tg*sizeof(double),cudaMemcpyDeviceToHost);
    
    int k33=(33+ng)*(nt+2*ng)+(0+ng);
    double rho_v=rho_b[k33], rhoE_v=rhoE_b[k33];
    double e_v=rhoE_v/std::fmax(rho_v,1e-20);
    double thresh=1e-15/((gamma-1)*1e-15);
    std::fprintf(stderr,"Before floor: rho=%.15e  rhoE=%.15e  e=%.15e\n", rho_v, rhoE_v, e_v);
    std::fprintf(stderr,"Floor threshold: e < %.15e ? %s\n", thresh, e_v<thresh?"YES":"NO");
    
    lm.apply_floor();
    cudaMemcpy(rho_b.data(),lm.d_rho,tg*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(rhoE_b.data(),lm.d_rhoE,tg*sizeof(double),cudaMemcpyDeviceToHost);
    rho_v=rho_b[k33]; rhoE_v=rhoE_b[k33];
    std::fprintf(stderr,"After floor:  rho=%.15e  rhoE=%.15e\n", rho_v, rhoE_v);
    
    // Check how many cells got floored
    int cnt=0;
    for(int f=0;f<nr*nt;f++){
        int i=f/nt, j=f%nt;
        int k=(i+ng)*(nt+2*ng)+(j+ng);
        if(rhoE_b[k] < 1e-10) cnt++;
    }
    std::fprintf(stderr,"Cells with rhoE<1e-10 after floor: %d / %d\n", cnt, nr*nt);
    
    lm.destroy();
}
