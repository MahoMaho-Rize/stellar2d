#pragma once

#include "../state.h"
#include "../grid.h"

enum class Limiter { MINMOD, VAN_LEER };

double minmod(double a, double b);
double van_leer(double a, double b);
double apply_limiter(double a, double b, Limiter lim);

struct ReconstructedPair {
    PrimitiveVars left;
    PrimitiveVars right;
};

ReconstructedPair muscl_reconstruct_r(
    const PrimitiveVars& wim1, const PrimitiveVars& wi,
    const PrimitiveVars& wip1, const PrimitiveVars& wip2,
    Limiter lim);

ReconstructedPair muscl_reconstruct_theta(
    const PrimitiveVars& wjm1, const PrimitiveVars& wj,
    const PrimitiveVars& wjp1, const PrimitiveVars& wjp2,
    Limiter lim);
