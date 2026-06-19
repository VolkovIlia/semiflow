/*
 * heat_var_a.c — C smoke test for smf_state_new_with_closure (ADR-0034, S1.2).
 *
 * Demonstrates variable diffusion coefficient a(x) via C function pointers.
 * Uses a constant a = 1.0 delivered through the callback mechanism so the
 * result can be compared directly against the unit-a closed-form oracle
 * (same as heat.c):
 *
 *   u(1, x) = (1/sqrt(5)) * exp(-x^2/5)
 *
 * Exit 0 on success (sup_error_var_a < 5e-4), exit 1 on failure.
 *
 * Callback design:
 *   a(x)   = *((double *)user_data)   -- returns the value stored in user_data
 *   a'(x)  = 0.0                      -- constant a => zero derivative
 *   a''(x) = 0.0                      -- constant a => zero second derivative
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

/* Diffusion coefficient: reads the constant value from user_data. */
static double a_const(double x, void *ud) {
    (void)x;
    return *(const double *)ud;
}

/* Zero function for a' and a'' (constant a). */
static double zero_fn(double x, void *ud) {
    (void)x;
    (void)ud;
    return 0.0;
}

int main(void) {
    double dx = (XMAX - XMIN) / (double)(N - 1);

    /* Build initial condition u0[i] = exp(-x_i^2). */
    double *u0 = malloc(N * sizeof(double));
    if (!u0) { fputs("malloc failed\n", stderr); return 1; }

    for (int i = 0; i < N; i++) {
        double x = XMIN + i * dx;
        u0[i] = exp(-x * x);
    }

    /* user_data: pointer to a constant a = 1.0, kept alive through state lifetime. */
    double a_value = 1.0;

    /* Construct state via the with_closure path. */
    SemiflowState *state = NULL;
    SemiflowStatus st = smf_state_new_with_closure(
        XMIN, XMAX, (size_t)N,
        a_const, zero_fn, zero_fn,
        (void *)&a_value,
        1.0,               /* a_norm_bound */
        u0, (size_t)N,
        &state
    );
    if (st != Ok) {
        fprintf(stderr, "new_with_closure failed: %s\n", smf_status_str(st));
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
    double *u_var = malloc(sz * sizeof(double));
    if (!u_var) {
        smf_state_free(state);
        free(u0);
        fputs("malloc failed\n", stderr);
        return 1;
    }
    st = smf_state_values(state, u_var, sz);
    if (st != Ok) {
        fprintf(stderr, "state_values failed: %s\n", smf_status_str(st));
        smf_state_free(state);
        free(u_var);
        free(u0);
        return 1;
    }

    /* Compare against oracle: u(1,x) = exp(-x^2/5) / sqrt(5). */
    double sup_err = 0.0;
    for (size_t i = 0; i < sz; i++) {
        double x = XMIN + (double)i * dx;
        double oracle = exp(-x * x / 5.0) / sqrt(5.0);
        double err = fabs(u_var[i] - oracle);
        if (err > sup_err) sup_err = err;
    }

    printf("sup_error_var_a=%.6e  version=%s\n", sup_err, smf_version());

    smf_state_free(state);
    free(u_var);
    free(u0);

    if (sup_err >= 5e-4) {
        fprintf(stderr, "FAIL: sup_error_var_a %.3e >= 5e-4\n", sup_err);
        return 1;
    }
    puts("PASS");
    return 0;
}
