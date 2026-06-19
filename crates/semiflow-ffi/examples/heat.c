/*
 * heat.c — C smoke test for semiflow-ffi.
 *
 * Solves the heat equation  du/dt = d^2u/dx^2  on [-10, 10]
 * with initial datum  u0(x) = exp(-x^2)  and compares the FFI result
 * against the closed-form Gaussian heat-kernel oracle:
 *   u(1, x) = (1/sqrt(5)) * exp(-x^2/5)
 * (unit diffusion a=1, t=1: spread factor 1+4*1*1 = 5).
 *
 * Exit 0 on success (sup_error < 5e-4), exit 1 on failure.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/semiflow.h"

#define N        1000
#define XMIN   -10.0
#define XMAX    10.0
#define T        1.0
#define N_STEPS  100

int main(void) {
    double dx = (XMAX - XMIN) / (double)(N - 1);

    /* Build initial condition u0[i] = exp(-x_i^2). */
    double *u0 = malloc(N * sizeof(double));
    if (!u0) { fputs("malloc failed\n", stderr); return 1; }

    for (int i = 0; i < N; i++) {
        double x = XMIN + i * dx;
        u0[i] = exp(-x * x);
    }

    /* Construct FFI state. */
    SemiflowState *state = NULL;
    SemiflowStatus st = smf_state_new_heat_1d_unit(
        XMIN, XMAX, (size_t)N,
        u0, (size_t)N,
        &state
    );
    if (st != Ok) {
        fprintf(stderr, "new_heat_1d_unit failed: %s\n", smf_status_str(st));
        free(u0);
        return 1;
    }

    /* Evolve for t=1.0 with N_STEPS steps. */
    st = smf_evolve(state, T, (size_t)N_STEPS);
    if (st != Ok) {
        fprintf(stderr, "evolve failed: %s\n", smf_status_str(st));
        smf_state_free(state);
        free(u0);
        return 1;
    }

    /* Read back values. */
    size_t sz = smf_state_size(state);
    double *u_ffi = malloc(sz * sizeof(double));
    if (!u_ffi) {
        smf_state_free(state);
        free(u0);
        fputs("malloc failed\n", stderr);
        return 1;
    }
    st = smf_state_values(state, u_ffi, sz);
    if (st != Ok) {
        fprintf(stderr, "state_values failed: %s\n", smf_status_str(st));
        smf_state_free(state);
        free(u_ffi);
        free(u0);
        return 1;
    }

    /* Compare against oracle: u(1,x) = exp(-x^2/5) / sqrt(5). */
    double sup_err = 0.0;
    for (size_t i = 0; i < sz; i++) {
        double x = XMIN + (double)i * dx;
        double oracle = exp(-x * x / 5.0) / sqrt(5.0);
        double err = fabs(u_ffi[i] - oracle);
        if (err > sup_err) sup_err = err;
    }

    printf("sup_error=%.6e  version=%s\n", sup_err, smf_version());

    smf_state_free(state);
    free(u_ffi);
    free(u0);

    if (sup_err >= 5e-4) {
        fprintf(stderr, "FAIL: sup_error %.3e >= 5e-4\n", sup_err);
        return 1;
    }
    puts("PASS");
    return 0;
}
