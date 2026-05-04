# AMReX-Astro Microphysics reaction rates — attribution

Our `src/physics/alpha_network.h` reaction rate functions (rate_3a, rate_c12ag,
rate_o16ag, rate_ne20ag, rate_mg24ag, rate_si28ag, rate_s32ag, rate_ar36ag,
rate_ca40ag, rate_ti44ag, rate_cr48ag, rate_fe52ag, rate_o16o16, rate_c12c12,
rate_c12o16) are direct ports of the same-named functions from:

    AMReX-Astro/Microphysics  rates/aprox_rates.H
    https://github.com/AMReX-Astro/Microphysics

That implementation is itself the aprox13 α-chain network developed by
F. X. Timmes (cococubed.com), based on CF88 + Caughlan-Fowler-Hoyle analytic
fits with reverse rates from detailed balance via Iliadis 2007 prefactors.

AMReX-Astro/Microphysics is distributed under BSD-3-Clause.  We include only
the mathematical forms of the rate functions (which are public-domain physics
fits); no source files are bundled.  The C++ port preserves all numerical
constants and branching logic verbatim.

References:
  Caughlan, G. R., & Fowler, W. A. 1988, ADNDT, 40, 283
  Fowler, Caughlan & Zimmerman 1975, ARA&A, 13, 69
  Hashimoto, M., Nomoto, K., & Shigeyama, T. 1989, A&A, 210, L5
  Timmes, F. X., aprox13 network (cococubed.com)
  Iliadis, C. 2007, "Nuclear Physics of Stars" (Wiley)
