#include "burn/pp_chain.h"
#include "eos.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

struct OneZoneConfig {
    std::string eos_type = "ideal_rad";
    double gamma = 5.0 / 3.0;
    double mu = 1.0;
    double radiation_a = 0.1;

    double rho = 1.0;
    double T0 = 1.0;
    double X_h0 = 0.7;
    double X_he0 = 0.3;

    double lambda0 = 1.0e-2;
    double temperature_ref = 1.0;
    double temperature_exponent = 4.0;
    double q_pp = 1.0;

    double t_end = 100.0;
    double dt_max = 0.1;
    double frac_change_limit = 0.02;
    int output_every = 10;
    std::string output_csv = "onezone_pp.csv";
};

int main(int argc, char** argv) {
    OneZoneConfig cfg;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--eos") == 0 && i + 1 < argc) cfg.eos_type = argv[++i];
        else if (std::strcmp(argv[i], "--gamma") == 0 && i + 1 < argc) cfg.gamma = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--mu") == 0 && i + 1 < argc) cfg.mu = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--radiation-a") == 0 && i + 1 < argc) cfg.radiation_a = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--rho") == 0 && i + 1 < argc) cfg.rho = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--T0") == 0 && i + 1 < argc) cfg.T0 = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--Xh0") == 0 && i + 1 < argc) cfg.X_h0 = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--Xhe0") == 0 && i + 1 < argc) cfg.X_he0 = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--lambda0") == 0 && i + 1 < argc) cfg.lambda0 = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--temperature-ref") == 0 && i + 1 < argc) cfg.temperature_ref = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--temperature-exponent") == 0 && i + 1 < argc) cfg.temperature_exponent = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--qpp") == 0 && i + 1 < argc) cfg.q_pp = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--tend") == 0 && i + 1 < argc) cfg.t_end = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--dt-max") == 0 && i + 1 < argc) cfg.dt_max = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--frac-limit") == 0 && i + 1 < argc) cfg.frac_change_limit = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--output-every") == 0 && i + 1 < argc) cfg.output_every = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--output-csv") == 0 && i + 1 < argc) cfg.output_csv = argv[++i];
    }

    std::unique_ptr<EOS> eos_storage;
    if (cfg.eos_type == "ideal") {
        eos_storage = std::make_unique<IdealGasEOS>(cfg.gamma, cfg.mu);
    } else if (cfg.eos_type == "ideal_rad") {
        eos_storage = std::make_unique<IdealGasRadiationEOS>(cfg.gamma, cfg.mu, cfg.radiation_a);
    } else {
        std::fprintf(stderr, "Unknown EOS type: %s\n", cfg.eos_type.c_str());
        return 1;
    }
    const EOS& eos = *eos_storage;

    OneZoneState state;
    state.rho = cfg.rho;
    state.X_h = cfg.X_h0;
    state.X_he = cfg.X_he0;
    // Set initial internal energy from requested T0 by matching pressure at (rho, T0).
    double R = 1.0 / cfg.mu;
    double P0 = cfg.rho * R * cfg.T0;
    if (cfg.eos_type == "ideal_rad") {
        P0 += cfg.radiation_a * cfg.T0 * cfg.T0 * cfg.T0 * cfg.T0 / 3.0;
    }
    state.e_int = eos.internal_energy_from_rho_p(cfg.rho, P0);

    PPChainParams params;
    params.lambda0 = cfg.lambda0;
    params.temperature_ref = cfg.temperature_ref;
    params.temperature_exponent = cfg.temperature_exponent;
    params.q_pp = cfg.q_pp;

    std::ofstream out(cfg.output_csv);
    out << "step,t,dt,rho,T,P,e_int,X_h,X_he,rate,eps_nuc\n";

    std::printf("onezone_pp - effective pp-chain burner\n");
    std::printf("EOS: %s  gamma=%.3f  mu=%.3f  a_rad=%.3e\n",
                cfg.eos_type.c_str(), cfg.gamma, cfg.mu, cfg.radiation_a);
    std::printf("rho=%.6e  T0=%.6e  Xh0=%.6e  Xhe0=%.6e\n",
                cfg.rho, cfg.T0, cfg.X_h0, cfg.X_he0);
    std::printf("lambda0=%.6e  Tref=%.6e  nu=%.3f  q_pp=%.6e\n",
                cfg.lambda0, cfg.temperature_ref, cfg.temperature_exponent, cfg.q_pp);

    double t = 0.0;
    int step = 0;
    while (t < cfg.t_end) {
        BurnDerivatives rhs = pp_chain_rhs(state, eos, params);
        double dt = std::min(cfg.dt_max, estimate_pp_burn_dt(state, eos, params, cfg.frac_change_limit));
        if (t + dt > cfg.t_end) dt = cfg.t_end - t;

        if (step % cfg.output_every == 0 || step == 0) {
            out << step << "," << t << "," << dt << "," << state.rho << ","
                << rhs.temperature << "," << rhs.pressure << "," << state.e_int << ","
                << state.X_h << "," << state.X_he << "," << rhs.rate << "," << rhs.eps_nuc << "\n";
        }

        advance_pp_chain_rk4(state, eos, params, dt);
        t += dt;
        step++;
    }

    BurnDerivatives rhs = pp_chain_rhs(state, eos, params);
    out << step << "," << t << "," << 0.0 << "," << state.rho << ","
        << rhs.temperature << "," << rhs.pressure << "," << state.e_int << ","
        << state.X_h << "," << state.X_he << "," << rhs.rate << "," << rhs.eps_nuc << "\n";
    out.close();

    std::printf("Final: t=%.6e  T=%.6e  X_h=%.6e  X_he=%.6e  eps=%.6e\n",
                t, rhs.temperature, state.X_h, state.X_he, rhs.eps_nuc);
    std::printf("CSV: %s\n", cfg.output_csv.c_str());
    return 0;
}
